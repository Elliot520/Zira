"""LTX-Video 2B distilled (0.9.8), the faster alternative to Wan under the video model selector.

Loaded from Lightricks' single-file checkpoint `ltxv-2b-0.9.8-distilled.safetensors` (bf16, transformer + VAE
in one file; its embedded config lists the 8 distilled steps used below). Distilled means no classifier-free
guidance and 8 fixed steps. Text: T5-XXL, which the official repo only has in fp32 (19GB), so like Wan's
UMT5 it is the fp16 file from `comfyanonymous/flux_text_encoders`, streamed block by block (see
app/tools/video.py::streamed_t5_encoder). Pieces continue each other through LTX's own video conditioning
(`LTXVideoCondition`): the previous piece's last frames are given at frame 0 of the next.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger("jarvis.tools.video")

# The distilled schedule from the checkpoint's own config ("allowed_inference_steps", on a 0-1 scale).
DISTILLED_TIMESTEPS = [1000, 994, 988, 981, 975, 909, 725, 422]
TEXT_LEN = 256  # LTX's default prompt length


@dataclass(frozen=True)
class LTXConfig:
    repo: str = "Lightricks/LTX-Video"
    file: str = "ltxv-2b-0.9.8-distilled.safetensors"
    text_encoder_repo: str = "comfyanonymous/flux_text_encoders"
    text_encoder_file: str = "t5xxl_fp16.safetensors"
    width: int = 576
    height: int = 320
    frames: int = 121  # frames per piece, 8k+1 (~5s at 24 fps)
    overlap_frames: int = 9  # 8k+1: frames of the previous piece the next one continues from
    fps: int = 24
    decode_timestep: float = 0.05
    decode_noise_scale: float = 0.025


class LTXBackend:
    def __init__(self, config: LTXConfig, local_first: Callable[..., Any], device: Callable[[], str]) -> None:
        self.config = config
        self._local_first = local_first
        self._device = device
        self._root: str | None = None
        self._weights: str | None = None
        self._text_encoder_file: str | None = None

    @property
    def steps(self) -> int:
        return len(DISTILLED_TIMESTEPS)

    def _snapshot(self) -> str:
        if self._root is None:
            from huggingface_hub import snapshot_download

            self._root = self._local_first(
                snapshot_download, self.config.repo,
                allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
                                "transformer/config.json", "vae/config.json"],
            )
        return self._root

    def _weights_path(self) -> str:
        if self._weights is None:
            from huggingface_hub import hf_hub_download

            self._weights = self._local_first(hf_hub_download, self.config.repo, self.config.file)
        return self._weights

    def _text_encoder_path(self) -> str:
        if self._text_encoder_file is None:
            from huggingface_hub import hf_hub_download

            self._text_encoder_file = self._local_first(
                hf_hub_download, self.config.text_encoder_repo, self.config.text_encoder_file
            )
        return self._text_encoder_file

    def load(self):
        """Transformer + VAE from the single file, tokenizer + scheduler from the repo, no text encoder."""
        import torch
        from diffusers import (
            AutoencoderKLLTXVideo,
            FlowMatchEulerDiscreteScheduler,
            LTXConditionPipeline,
            LTXVideoTransformer3DModel,
        )
        from transformers import T5TokenizerFast

        root = self._snapshot()
        weights = self._weights_path()
        transformer = LTXVideoTransformer3DModel.from_single_file(weights, torch_dtype=torch.bfloat16)
        vae = AutoencoderKLLTXVideo.from_single_file(weights, torch_dtype=torch.bfloat16)
        tokenizer = T5TokenizerFast.from_pretrained(root, subfolder="tokenizer")
        # The distilled model runs a fixed 8-step schedule (DISTILLED_TIMESTEPS); the repo's scheduler config
        # turns on dynamic shifting, which would require re-deriving the schedule from the resolution.
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            root, subfolder="scheduler", use_dynamic_shifting=False
        )
        pipe = LTXConditionPipeline(
            scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=tokenizer, transformer=transformer
        )
        pipe.to(self._device())
        pipe.vae.enable_tiling()
        return pipe

    def encode_prompt(self, pipe, prompt: str, streamed_encoder):
        """prompt -> (embeds, attention mask), with T5 streamed block by block and released after."""
        import torch
        from transformers import T5Config, T5EncoderModel

        from app.tools.video import _mps_cleanup

        device = self._device()
        started = time.monotonic()
        config = T5Config.from_pretrained(self._snapshot(), subfolder="text_encoder")
        with streamed_encoder(config, T5EncoderModel, self._text_encoder_path(), device) as encoder:
            pipe.text_encoder = encoder
            try:
                with torch.no_grad():
                    embeds, mask, _, _ = pipe.encode_prompt(
                        prompt=prompt, do_classifier_free_guidance=False, max_sequence_length=TEXT_LEN,
                        device=device, dtype=torch.bfloat16,
                    )
                    embeds, mask = embeds.clone(), mask.clone()
                    if device == "mps":
                        torch.mps.synchronize()
            finally:
                pipe.text_encoder = None
        del encoder
        gc.collect()
        _mps_cleanup()
        logger.info("Prompt encoded for LTX in %.1fs (T5 streamed block by block, then released)", time.monotonic() - started)
        return embeds, mask

    def run_piece(self, pipe, embeds, width: int, height: int, frames: int, tail, callback) -> Any:
        """One piece as float frames (F x H x W x 3, 0..1). `tail`: the previous piece's last frames to
        continue from (None for the first piece)."""
        import numpy as np
        import torch
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        from PIL import Image

        prompt_embeds, prompt_mask = embeds
        kwargs: dict[str, Any] = {}
        if callback is not None:
            kwargs["callback_on_step_end"] = callback
        if tail is not None:
            keep = [Image.fromarray((np.clip(f, 0, 1) * 255).round().astype(np.uint8)) for f in tail]
            kwargs["conditions"] = [LTXVideoCondition(video=keep, frame_index=0, strength=1.0)]
        with torch.no_grad():
            output = pipe(
                prompt_embeds=prompt_embeds, prompt_attention_mask=prompt_mask, height=height, width=width,
                num_frames=frames, frame_rate=self.config.fps, timesteps=DISTILLED_TIMESTEPS, guidance_scale=1.0,
                decode_timestep=self.config.decode_timestep, decode_noise_scale=self.config.decode_noise_scale,
                output_type="np", **kwargs,
            )
        return np.asarray(output.frames[0], dtype=np.float32)
