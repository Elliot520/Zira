"""HunyuanVideo 1.5 480p image-to-video worker (step-distilled, 8 or 12 steps), run by app/tools/video_hunyuan.py with
Zira's own Python. Same line protocol as the FastMetal workers (third_party/fastmetal/worker5b.py), so the same
client starts it, streams its steps and can end it mid-step (Stop):

  -> {"ready": true, "load_seconds": ...}                          at start (nothing big is loaded yet)
  <- {"prompt", "out", "width", "height", "frames", "seed", "condition": H x W x 3 uint8 .npy (required: I2V)}
  -> {"step", "total", "seconds"} per step, then {"ok": true, "frames", "seconds", "decode_seconds", "peak_gb"}

Memory on a 16GB Mac (the full model is ~35GB): the transformer and the Qwen2.5-VL text encoder are 4-bit SDNQ
copies (third_party/hunyuan/convert.py -> models/hunyuan15-uint4, ~4.5GB each), and they are never in memory
together: per request the text encoder encodes the prompt and is freed, then the transformer, VAE and image encoder
load, make the clip and are freed - so an idle worker holds almost nothing. Attention runs in query chunks: MPS's
attention would otherwise build the full score matrix (tens of GB for a 5s clip's ~49k tokens).
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

# Protocol lines only on the real stdout; library output goes to the log (stderr).
_protocol = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.environ.get("HUNYUAN_ROOT", os.path.join(ROOT, "models", "hunyuan15"))
QUANT = os.environ.get("HUNYUAN_QUANT", os.path.join(ROOT, "models", "hunyuan15-uint4"))
STEPS = int(os.environ.get("HUNYUAN_STEPS", "8"))
ATTENTION_BUDGET = int(os.environ.get("HUNYUAN_ATTENTION_ELEMENTS", str(192 * 1024 * 1024)))  # scores per chunk
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16

_sdpa = F.scaled_dot_product_attention


def _chunked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    """scaled_dot_product_attention over slices of the queries when the score matrix would be large (same result)."""
    q_len, k_len = query.shape[-2], key.shape[-2]
    heads = query.shape[-3] if query.dim() >= 3 else 1
    if is_causal or query.device.type != "mps" or q_len * k_len * heads <= ATTENTION_BUDGET:
        return _sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                     **kwargs)
    chunk = max(64, ATTENTION_BUDGET // (k_len * heads))
    out = []
    for start in range(0, q_len, chunk):
        mask = attn_mask
        if mask is not None and mask.dim() >= 2 and mask.shape[-2] == q_len:
            mask = mask[..., start:start + chunk, :]
        out.append(_sdpa(query[..., start:start + chunk, :], key, value, attn_mask=mask, dropout_p=dropout_p,
                         scale=scale, **kwargs))
    return torch.cat(out, dim=-2)


F.scaled_dot_product_attention = _chunked_sdpa  # before diffusers is imported, so it picks this one up
torch.nn.functional.scaled_dot_product_attention = _chunked_sdpa


def send(obj: dict) -> None:
    _protocol.write(json.dumps(obj) + "\n")


def free() -> None:
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()


def peak_gb() -> float:
    try:
        return round(torch.mps.driver_allocated_memory() / 1e9, 2)
    except Exception:  # noqa: BLE001
        return 0.0


def rebuild_buffers(model):
    """load_sdnq_model builds the model empty and fills in the saved weights; buffers that are computed, never saved
    (Qwen's rotary inv_freq) are left without memory ("Placeholder storage has not been allocated on MPS")."""
    rotary = getattr(model, "rotary_emb", None)
    if rotary is not None:
        model.rotary_emb = type(rotary)(config=model.config, device=DEVICE)
    empty = [name for name, buf in model.named_buffers() if buf.device.type == "meta"]
    if empty:
        raise RuntimeError(f"buffers left empty after loading: {empty[:5]}")
    return model


def encode(prompt: str):
    """The prompt's two embeddings (Qwen2.5-VL + byT5), on the CPU; the text encoders are freed again."""
    from diffusers import FlowMatchEulerDiscreteScheduler, HunyuanVideo15ImageToVideoPipeline
    from diffusers.guiders import ClassifierFreeGuidance
    from sdnq import load_sdnq_model
    from transformers import ByT5Tokenizer, Qwen2_5_VLTextModel, Qwen2TokenizerFast, T5EncoderModel

    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLHunyuanVideo15

    text_encoder = rebuild_buffers(load_sdnq_model(os.path.join(QUANT, "text_encoder"), model_cls=Qwen2_5_VLTextModel,
                                                   dtype=DTYPE, device=DEVICE))
    text_encoder_2 = T5EncoderModel.from_pretrained(os.path.join(SRC, "text_encoder_2"), torch_dtype=DTYPE).to(DEVICE)
    # The pipeline's constructor reads the VAE's settings; an empty stand-in (no weights, no memory) is enough here.
    with init_empty_weights():
        vae_stub = AutoencoderKLHunyuanVideo15.from_config(AutoencoderKLHunyuanVideo15.load_config(SRC, subfolder="vae"))
    pipe = HunyuanVideo15ImageToVideoPipeline(
        text_encoder=text_encoder, tokenizer=Qwen2TokenizerFast.from_pretrained(os.path.join(SRC, "tokenizer")),
        text_encoder_2=text_encoder_2, tokenizer_2=ByT5Tokenizer.from_pretrained(os.path.join(SRC, "tokenizer_2")),
        transformer=None, vae=vae_stub, image_encoder=None, feature_extractor=None,
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(SRC, subfolder="scheduler"),
        guider=ClassifierFreeGuidance.from_pretrained(SRC, subfolder="guider"),
    )
    with torch.no_grad():
        embeds = pipe.encode_prompt(prompt=prompt, device=DEVICE, dtype=DTYPE)
    embeds = tuple(e.cpu() if isinstance(e, torch.Tensor) else e for e in embeds)
    del pipe, text_encoder, text_encoder_2
    free()
    return embeds


def generate(req: dict) -> dict:
    from diffusers import (AutoencoderKLHunyuanVideo15, FlowMatchEulerDiscreteScheduler,
                           HunyuanVideo15ImageToVideoPipeline, HunyuanVideo15Transformer3DModel)
    from diffusers.guiders import ClassifierFreeGuidance
    from PIL import Image
    from sdnq import load_sdnq_model
    from transformers import SiglipImageProcessor, SiglipVisionModel

    started = time.monotonic()
    embeds, embeds_mask, embeds_2, embeds_mask_2 = encode(req["prompt"])
    encoded = time.monotonic()

    transformer = load_sdnq_model(os.path.join(QUANT, "transformer"), model_cls=HunyuanVideo15Transformer3DModel,
                                  dtype=DTYPE, device=DEVICE)
    vae = AutoencoderKLHunyuanVideo15.from_pretrained(SRC, subfolder="vae", torch_dtype=DTYPE).to(DEVICE)
    vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256)  # small tiles: little memory per tile
    image_encoder = SiglipVisionModel.from_pretrained(os.path.join(SRC, "image_encoder"), torch_dtype=DTYPE).to(DEVICE)
    pipe = HunyuanVideo15ImageToVideoPipeline(
        text_encoder=None, tokenizer=None, text_encoder_2=None, tokenizer_2=None,
        transformer=transformer, vae=vae, image_encoder=image_encoder,
        feature_extractor=SiglipImageProcessor.from_pretrained(os.path.join(SRC, "feature_extractor")),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(SRC, subfolder="scheduler"),
        guider=ClassifierFreeGuidance.from_pretrained(SRC, subfolder="guider"),
    )
    pipe.set_progress_bar_config(disable=True)
    steps = int(req.get("steps") or STEPS)
    done = {"n": 0, "peak": 0.0}
    step = pipe.scheduler.step

    def counted_step(*args, **kwargs):
        result = step(*args, **kwargs)
        done["n"] += 1
        done["peak"] = max(done["peak"], peak_gb())
        send({"step": done["n"], "total": steps, "seconds": round(time.monotonic() - started, 1)})
        return result

    pipe.scheduler.step = counted_step
    image = Image.fromarray(np.load(req["condition"]))
    generator = torch.Generator("cpu").manual_seed(int(req.get("seed") or 0))
    denoised = time.monotonic()
    with torch.no_grad():
        latents = pipe(image=image, prompt_embeds=embeds.to(DEVICE), prompt_embeds_mask=embeds_mask.to(DEVICE),
                       prompt_embeds_2=embeds_2.to(DEVICE), prompt_embeds_mask_2=embeds_mask_2.to(DEVICE),
                       num_frames=int(req["frames"]), num_inference_steps=steps, generator=generator,
                       output_type="latent").frames
    finished = time.monotonic()
    # Decode only after the transformer and image encoder are gone: decoding next to them ran a 16GB Mac into
    # swap until it stalled (first real test, 2026-09-28). The latents are saved first, so a failed decode does not
    # lose the minutes of denoising.
    torch.save(latents.cpu(), req["out"] + ".latents.pt")
    pipe.transformer = pipe.image_encoder = None
    del transformer, image_encoder
    free()
    with torch.no_grad():
        latents = latents.to(DEVICE, vae.dtype) / vae.config.scaling_factor
        video = pipe.video_processor.postprocess_video(vae.decode(latents, return_dict=False)[0], output_type="np")[0]
    decoded = time.monotonic()
    frames = (np.clip(video, 0.0, 1.0) * 255).round().astype(np.uint8)
    width, height = int(req.get("width") or frames.shape[2]), int(req.get("height") or frames.shape[1])
    if (frames.shape[2], frames.shape[1]) != (width, height):  # the model renders its own 480p; Zira asks a size
        frames = np.stack([np.asarray(Image.fromarray(f).resize((width, height), Image.Resampling.LANCZOS))
                           for f in frames])
    np.save(req["out"], frames)
    del pipe, vae, video, latents
    free()
    return {"ok": True, "frames": int(frames.shape[0]), "width": int(frames.shape[2]), "height": int(frames.shape[1]),
            "seconds": round(decoded - started, 1), "encode_seconds": round(encoded - started, 1),
            "denoise_seconds": round(finished - denoised, 1), "decode_seconds": round(decoded - finished, 1),
            "peak_gb": max(done["peak"], peak_gb())}


def main() -> None:
    started = time.monotonic()
    try:
        import diffusers  # noqa: F401 - fail early (and clearly) if the environment is broken
        for part in ("transformer", "text_encoder"):
            if not os.path.isdir(os.path.join(QUANT, part)):
                raise RuntimeError(f"{part} not converted yet: run third_party/hunyuan/convert.py")
    except Exception as exc:  # noqa: BLE001
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    send({"ready": True, "load_seconds": round(time.monotonic() - started, 1), "device": DEVICE})
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        try:
            send(generate(req))
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            free()
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
