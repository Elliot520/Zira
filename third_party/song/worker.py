"""ACE-Step 1.5 song worker (MIT, github.com/ace-step/ACE-Step-1.5, cloned in third_party/ace-step), run by
app/tools/song.py with the `.venv-acestep` Python (3.11): ACE-Step needs Python < 3.13 and its own pinned
transformers, and Zira's environment is 3.14.

Protocol: JSON lines. On start it loads the models once and prints
    {"ready": true, "device": ..., "load_seconds": ...}   or   {"ready": false, "error": ...}
Then, per request line {"id": n, "lyrics": ..., "caption": ..., "language": ..., "seconds": ..., "out": <path>}
it prints any number of {"id": n, "progress": 0..1, "desc": ...} lines and then
    {"id": n, "ok": true, "audio_seconds": .., "seconds": ..}   or   {"id": n, "ok": false, "error": ...}
The worker exits when its input closes (Zira ends it after each song, which frees all of its memory).

Settings come from the environment: ACESTEP_ROOT (the cloned repo, with checkpoints/), ACESTEP_DIT (default
acestep-v15-turbo), ACESTEP_LM (default acestep-5Hz-lm-1.7B, "none" to skip the planning model),
ACESTEP_QUANTIZATION (e.g. int8_weight_only, default none), ACESTEP_STEPS (default 8).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(os.environ.get("ACESTEP_ROOT", Path(__file__).resolve().parents[1] / "ace-step"))
DIT = os.environ.get("ACESTEP_DIT", "acestep-v15-turbo")
LM = os.environ.get("ACESTEP_LM", "acestep-5Hz-lm-1.7B")
QUANTIZATION = os.environ.get("ACESTEP_QUANTIZATION") or None
STEPS = int(os.environ.get("ACESTEP_STEPS", "8"))
# Memory on a 16GB Mac (measured 2026-09-28: the defaults ran out at 11.7GB PyTorch + 10.1GB MLX): ACE-Step loads its
# PyTorch models in float32 on MPS and also keeps a converted MLX copy of the song model. Here the MLX copy is off
# and the models are cast to bfloat16 after loading, which roughly halves them.
MLX_DIT = os.environ.get("ACESTEP_MLX_DIT", "0") == "1"
DTYPE = os.environ.get("ACESTEP_DTYPE", "bfloat16")
os.environ.setdefault("ACESTEP_LM_BACKEND", "mlx")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Replies go to the real stdout only; library output goes to stderr (logs/song.log), so it never breaks the protocol.
_OUT = sys.stdout
sys.stdout = sys.stderr


def reply(payload: dict) -> None:
    _OUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _OUT.flush()


def main() -> None:
    started = time.monotonic()
    try:
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        import torch
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        from acestep.llm_inference import LLMHandler

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dit = AceStepHandler()
        status, ok = dit.initialize_service(project_root=str(ROOT), config_path=DIT, device=device,
                                            quantization=QUANTIZATION, use_mlx_dit=MLX_DIT)
        if not ok:
            raise RuntimeError(f"the song model did not load: {status}")
        if DTYPE != "float32" and device == "mps" and QUANTIZATION is None:
            dtype = getattr(torch, DTYPE)
            dit.dtype = dtype
            for name in ("model", "text_encoder"):
                module = getattr(dit, name, None)
                if module is not None:
                    setattr(dit, name, module.to(dtype))
            if getattr(dit, "silence_latent", None) is not None:
                dit.silence_latent = dit.silence_latent.to(dtype)
            torch.mps.empty_cache()
        llm = LLMHandler()
        if LM != "none":
            llm.initialize(checkpoint_dir=str(ROOT / "checkpoints"), lm_model_path=LM, backend="mlx", device=device)
    except Exception as exc:  # noqa: BLE001 - reported to Zira, which shows it to the user
        traceback.print_exc()
        reply({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    reply({"ready": True, "device": device, "dit": DIT, "lm": LM, "dtype": str(dit.dtype), "mlx_dit": MLX_DIT,
           "load_seconds": round(time.monotonic() - started, 1)})

    for line in sys.stdin:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid = req.get("id")

        def progress(value=None, desc=None, *args, **kwargs):
            if isinstance(value, (int, float)):
                reply({"id": rid, "progress": round(float(value), 3), "desc": str(desc or "")})

        began = time.monotonic()
        try:
            params = GenerationParams(
                task_type="text2music",
                caption=str(req.get("caption") or "")[:500],
                lyrics=str(req.get("lyrics") or "")[:4000],
                vocal_language=str(req.get("language") or "unknown"),
                duration=float(req.get("seconds") or -1),
                inference_steps=STEPS,
                shift=3.0,  # the turbo model's recommended timestep shift (docs/en/INFERENCE.md)
                thinking=LM != "none",
            )
            config = GenerationConfig(batch_size=1, audio_format="mp3")
            with tempfile.TemporaryDirectory() as tmp:
                result = generate_music(dit, llm, params, config, save_dir=tmp, progress=progress)
                if not result.success or not result.audios:
                    raise RuntimeError(result.error or "no audio was made")
                made = Path(result.audios[0]["path"])
                out = Path(req["out"])
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(made), out)
            info = {}
            try:
                import soundfile as sf

                info["audio_seconds"] = round(sf.info(str(out)).duration, 1)
            except Exception:  # noqa: BLE001 - the length is only for the log
                pass
            reply({"id": rid, "ok": True, "seconds": round(time.monotonic() - began, 1), **info})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            reply({"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            try:
                torch.mps.empty_cache()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
