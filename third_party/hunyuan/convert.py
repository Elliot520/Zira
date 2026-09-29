"""One-time conversion of HunyuanVideo 1.5 480p I2V step-distilled (models/hunyuan15, bf16, ~35GB) into 4-bit
(SDNQ uint4) copies of its two big parts, so they fit a 16GB Mac one at a time (user request, 2026-09-28):
  transformer   16.7GB bf16 -> ~4.7GB   (models/hunyuan15-uint4/transformer)
  text_encoder  14.1GB bf16 -> ~4.2GB   (models/hunyuan15-uint4/text_encoder; Qwen2.5-VL 7B's language model)
The small parts (VAE, image encoder, byT5, tokenizers, scheduler) are used as they are from models/hunyuan15.

Run with Zira's Python: .venv/bin/python third_party/hunyuan/convert.py [transformer|text_encoder]
Each part is loaded shard by shard and quantized while loading, then saved; peak memory stays near one shard.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch
from sdnq import SDNQConfig, save_sdnq_model

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "models" / "hunyuan15"
DST = ROOT / "models" / "hunyuan15-uint4"


def convert(part: str) -> None:
    config = SDNQConfig(weights_dtype="uint4", quantization_device="cpu", return_device="cpu")
    started = time.monotonic()
    if part == "transformer":
        from diffusers import HunyuanVideo15Transformer3DModel

        model = HunyuanVideo15Transformer3DModel.from_pretrained(
            SRC, subfolder="transformer", torch_dtype=torch.bfloat16, quantization_config=config)
    elif part == "text_encoder":
        from transformers import Qwen2_5_VLTextModel

        model = Qwen2_5_VLTextModel.from_pretrained(
            SRC / "text_encoder", torch_dtype=torch.bfloat16, quantization_config=config)
    else:
        raise SystemExit(f"unknown part {part!r}: transformer or text_encoder")
    out = DST / part
    out.mkdir(parents=True, exist_ok=True)
    save_sdnq_model(model, str(out), sdnq_config=config)
    size = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
    print(f"{part}: saved {size:.2f} GB to {out} in {time.monotonic() - started:.0f}s", flush=True)
    del model
    gc.collect()


if __name__ == "__main__":
    for part in sys.argv[1:] or ("transformer", "text_encoder"):
        convert(part)
