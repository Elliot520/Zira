"""FastMetal-QAD 1.3B video worker. Runs in the separate `.venv-fastmetal` environment (Python 3.11 with
FastVideo's pinned libraries and Apple's MLX), started and stopped by app/tools/video_fastmetal.py.

FastMetal-QAD is Wan 2.1 1.3B distilled to 3 DMD steps with an INT8 MLX DiT (FastVideo / Hao AI Lab). The
worker only denoises and decodes: the prompt is encoded by Zira itself (its streamed UMT5, see
app/tools/video.py::streamed_t5_encoder) and handed over as a .npy file, so the 11GB text encoder is never
loaded here. The steps mirror FastVideo's own examples/inference/basic/mlx_wan_prompt_to_video.py (default
release settings: 3-step DMD 1000,757,522, flow shift 8, fp16 MLX, TAEHV decode), reusing its functions.
One difference: TAEHV decodes frame by frame (PyTorch, MPS) instead of FastVideo's all-frames-at-once MLX
decoder - measured on this 16GB Mac, the parallel decode ran the GPU out of memory even for a 45-frame
832x480 clip, after all three denoising steps had finished.

Protocol (one JSON object per line): the worker prints {"ready": true, ...} once the DiT is loaded, then for
every request {"embeds": "/in.npy", "out": "/frames.npy", "width": W, "height": H, "frames": F, "seed": S}
it prints {"step": i, "total": n} after each denoising step and finally {"ok": true, "frames": F,
"seconds": ...} or {"ok": false, "error": "..."}. The output is uint8 frames, F x H x W x 3.
"""

from __future__ import annotations

import json
import os
import sys
import time

# The libraries print progress to stdout; keep the real stdout for protocol lines only.
_protocol = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "fastvideo", "examples", "inference", "basic"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import mlx_wan_prompt_to_video as fastmetal  # noqa: E402 - FastVideo's own example, for its helpers
from fastvideo.mlx_runtime.checkpoint import load_mlx_dit_checkpoint  # noqa: E402
from fastvideo.mlx_runtime.memory import cleanup_mlx  # noqa: E402
from fastvideo.mlx_runtime.refine import plan_refine_resolutions  # noqa: E402
from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, dmd_step  # noqa: E402
from fastvideo.mlx_runtime.wan_vae import ensure_taehv_checkpoint  # noqa: E402
from fastvideo.third_party.taehv import TAEHV  # noqa: E402
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (  # noqa: E402
    FlowMatchEulerDiscreteScheduler,
)

DMD_TIMESTEPS = [1000, 757, 522]
_taehv = None  # loaded on first decode (~22MB), kept for the next clip
FLOW_SHIFT = 8.0
MX_DTYPE = mx.float16


def send(obj: dict) -> None:
    _protocol.write(json.dumps(obj) + "\n")
    _protocol.flush()


def decode_frame_by_frame(latents_np: np.ndarray) -> np.ndarray:
    """TAEHV (Wan 2.1's tiny decoder) in sequential mode: one frame's activations at a time."""
    global _taehv
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    if _taehv is None:
        _taehv = TAEHV(str(ensure_taehv_checkpoint(z_dim=latents_np.shape[1]))).to(device=device, dtype=dtype).eval()
    latents = torch.from_numpy(latents_np).to(device=device, dtype=dtype)
    with torch.no_grad():
        frames = _taehv.decode_video(latents.transpose(1, 2), parallel=False, show_progress_bar=False)  # N T C H W
    video = frames[0].permute(0, 2, 3, 1).float().clamp(0, 1).cpu().numpy()
    del latents, frames
    if device == "mps":
        torch.mps.empty_cache()
    return video


def generate(dit, schedule: MLXDMDSchedule, request: dict, stats: dict) -> int:
    config = dit.config
    width, height, frames, seed = int(request["width"]), int(request["height"]), int(request["frames"]), int(request["seed"])
    plan = plan_refine_resolutions(
        height=height, width=width, num_frames=frames, spatial_scale=1, vae_spatial_compression=8,
        vae_temporal_compression=4, patch_size=tuple(config.get("patch_size", (1, 2, 2))), enabled=False,
    )
    latent_frames, latent_height, latent_width = plan.latent_frames, plan.stage1_latent_height, plan.stage1_latent_width

    mx.random.seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((1, int(config["in_channels"]), latent_frames, latent_height, latent_width),
                        generator=generator, dtype=torch.float32)
    latents = mx.array(noise.numpy()).astype(MX_DTYPE)
    encoder_hidden_states = mx.array(np.load(request["embeds"])).astype(MX_DTYPE)
    freqs_cis = fastmetal.make_rotary_embeddings(
        config, latent_frames=latent_frames, latent_height=latent_height, latent_width=latent_width,
    )

    mx.reset_peak_memory()
    for index, timestep in enumerate(DMD_TIMESTEPS):
        step_started = time.monotonic()
        noise_input = latents.astype(mx.float32)
        pred = dit(latents.astype(MX_DTYPE), encoder_hidden_states, mx.array([float(timestep)]).astype(mx.float32), freqs_cis)
        last = index == len(DMD_TIMESTEPS) - 1
        latents = dmd_step(
            latents=noise_input, noise_input_latent=noise_input, pred_noise=pred.astype(mx.float32),
            schedule=schedule, timestep=float(timestep),
            next_timestep=None if last else float(DMD_TIMESTEPS[index + 1]),
            noise=None if last else mx.random.normal(noise_input.shape).astype(mx.float32),
        ).astype(MX_DTYPE)
        mx.eval(latents)
        send({"step": index + 1, "total": len(DMD_TIMESTEPS), "seconds": round(time.monotonic() - step_started, 1)})
        del noise_input, pred

    denoise_peak = mx.get_peak_memory()
    latents_np = np.array(latents.astype(mx.float32))
    del latents, encoder_hidden_states, freqs_cis
    cleanup_mlx()
    decode_started = time.monotonic()
    video = decode_frame_by_frame(latents_np)  # T x H x W x 3, 0..1
    stats["decode_seconds"] = round(time.monotonic() - decode_started, 1)
    stats["peak_gb"] = round(denoise_peak / 2**30, 2)
    np.save(request["out"], (np.clip(video, 0.0, 1.0) * 255).round().astype(np.uint8))
    return int(video.shape[0])


def main() -> None:
    root = os.environ["FASTMETAL_ROOT"]
    started = time.monotonic()
    try:
        dit = load_mlx_dit_checkpoint(root, compile=True)
        schedule = MLXDMDSchedule.from_torch_scheduler(FlowMatchEulerDiscreteScheduler(shift=FLOW_SHIFT))
    except Exception as exc:  # noqa: BLE001 - reported to the parent, which shows it to the user
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    send({"ready": True, "load_seconds": round(time.monotonic() - started, 1)})

    for line in sys.stdin:
        if not line.strip():
            continue
        started = time.monotonic()
        try:
            stats: dict = {}
            count = generate(dit, schedule, json.loads(line), stats)
            send({"ok": True, "frames": count, "seconds": round(time.monotonic() - started, 1), **stats})
        except Exception as exc:  # noqa: BLE001 - one failed clip must not end the worker
            cleanup_mlx()
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
