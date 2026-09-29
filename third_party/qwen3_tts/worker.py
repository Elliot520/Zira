"""Qwen3-TTS worker using MLX-Audio on Apple Silicon."""

from __future__ import annotations

import json
import os
import sys
import time

_protocol = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def send(message: dict) -> None:
    _protocol.write(json.dumps(message, ensure_ascii=False) + "\n")
    _protocol.flush()


class LoRALinear:
    def __new__(cls, base, rank: int, alpha: float):
        import mlx.core as mx
        import mlx.nn as nn

        class _LoRALinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base
                self.scale = alpha / rank
                self.lora_a = mx.zeros((rank, base.weight.shape[-1]))
                self.lora_b = mx.zeros((base.weight.shape[0], rank))

            def __call__(self, value):
                return self.base(value) + (value @ self.lora_a.T @ self.lora_b.T) * self.scale

        return _LoRALinear()


class QLoRALinear:
    def __new__(cls, base, rank: int, alpha: float):
        import mlx.core as mx
        import mlx.nn as nn

        input_size = base.weight.shape[-1] * (32 // base.bits)
        output_size = base.scales.shape[0]

        class _QLoRALinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base
                self.base.freeze()
                self.scale = alpha / rank
                self.lora_a = mx.zeros((rank, input_size))
                self.lora_b = mx.zeros((output_size, rank))

            def __call__(self, value):
                return self.base(value) + (value @ self.lora_a.T @ self.lora_b.T) * self.scale

        return _QLoRALinear()


def apply_hindi_adapter(model, adapter_path: str) -> tuple[int, int]:
    import mlx.core as mx
    import mlx.nn as nn

    targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    def patch(module) -> int:
        children = module.children()
        if not isinstance(children, dict):
            return 0
        count = 0
        for name, child in children.items():
            if name in targets and isinstance(child, nn.QuantizedLinear):
                setattr(module, name, QLoRALinear(child, rank=8, alpha=16.0))
                count += 1
            elif name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank=8, alpha=16.0))
                count += 1
            elif isinstance(child, nn.Module):
                count += patch(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, nn.Module):
                        count += patch(item)
        return count

    patched = patch(model.talker)
    if patched == 0:
        raise RuntimeError("No Qwen3-TTS talker layers matched the Hindi LoRA targets.")

    adapters = mx.load(adapter_path)
    loaded = 0
    for key, value in adapters.items():
        parts = key.split(".")
        target = model
        for part in parts[:-1]:
            target = target[int(part)] if part.isdigit() else getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise RuntimeError(f"Hindi adapter tensor has no matching model parameter: {key}")
        setattr(target, parts[-1], value)
        loaded += 1
    if loaded != 462:
        raise RuntimeError(f"Expected 462 Hindi adapter tensors, loaded {loaded}.")
    return patched, loaded


def main() -> None:
    model_id = os.environ.get("QWEN3_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit")
    adapter_id = os.environ.get("QWEN3_TTS_ADAPTER", "akashicmarga/qwen3-tts-hindi-lora-grpo")
    try:
        from mlx_audio.audio_io import write as write_audio
        from mlx_audio.tts.utils import load_model
        from huggingface_hub import hf_hub_download
        import numpy as np

        model = load_model(model_id)
        adapter_path = hf_hub_download(repo_id=adapter_id, filename="adapters.safetensors")
        patched, loaded = apply_hindi_adapter(model, adapter_path)
        model.eval()
    except Exception as exc:  # noqa: BLE001 - return startup errors through the worker protocol
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return

    send({"ready": True, "model": model_id, "adapter": adapter_id, "patched_layers": patched,
          "adapter_tensors": loaded, "sample_rate": model.sample_rate})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            started = time.monotonic()
            results = list(model.generate(
                text=request["text"],
                lang_code=request.get("language", "auto").lower(),
                temperature=0.7,
            ))
            if not results:
                raise RuntimeError("The model returned no audio segments.")
            audio = np.concatenate([np.asarray(result.audio).reshape(-1) for result in results])
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError("The model returned empty or invalid audio.")
            write_audio(request["out"], audio, model.sample_rate, format="wav")
            send({"ok": True, "seconds": round(time.monotonic() - started, 2),
                  "audio_seconds": round(audio.size / model.sample_rate, 2)})
        except Exception as exc:  # noqa: BLE001 - a failed synthesis must not kill the text chat
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()