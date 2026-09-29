"""IndicF5-Hinglish speech worker. Runs in the separate `.venv-tts` environment (Python 3.11 with the
older libraries F5-TTS needs), started and stopped by app/voice/text_to_speech.py::IndicF5TTS.

Protocol (one JSON object per line): the worker prints {"ready": true, ...} once the model is loaded,
then for every request {"text": "...", "out": "/path.wav"} it writes a 24 kHz WAV and replies
{"ok": true, "seconds": ..., "audio_seconds": ...} or {"ok": false, "error": "..."}.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

# The library prints progress to stdout; keep the real stdout for protocol lines only.
_protocol = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from f5_tts.infer.utils_infer import infer_process, load_vocoder, preprocess_ref_audio_text  # noqa: E402
from f5_tts.model import CFM, DiT  # noqa: E402
from f5_tts.model.utils import get_tokenizer  # noqa: E402

SAMPLE_RATE = 24000
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
CHARS_PER_SECOND = 13.0  # typical speaking rate, used instead of F5's byte-count estimate (see _speed)


def send(obj: dict) -> None:
    _protocol.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _protocol.flush()


def cached(pattern: str) -> str:
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    found = glob.glob(os.path.join(hub, pattern))
    if not found:
        raise FileNotFoundError(f"Not in the Hugging Face cache: {pattern}")
    return found[0]


def load(voice_id: str, device: str):
    vocab_map, vocab_size = get_tokenizer(cached("models--ai4bharat--IndicF5/snapshots/*/checkpoints/vocab.txt"), tokenizer="custom")
    model = CFM(
        transformer=DiT(**MODEL_CFG, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
                             target_sample_rate=SAMPLE_RATE, mel_spec_type="vocos"),
        odeint_kwargs=dict(method="euler"),
        vocab_char_map=vocab_map,
    )
    weights = load_file(cached("models--Saravananravi--indicf5-hinglish/snapshots/*/model.safetensors"))
    weights = {k[len("ema_model."):] if k.startswith("ema_model.") else k: v for k, v in weights.items() if k not in ("initted", "step")}
    result = model.load_state_dict(weights, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("mel_spec.")]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing[:3]} unexpected={result.unexpected_keys[:3]}")
    # fp32 on purpose: measured, fp16 on MPS produced NaN audio.
    model = model.to(device).eval()
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

    voices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
    meta = json.load(open(os.path.join(voices_dir, "voices.json"), encoding="utf-8"))[voice_id]
    ref_audio, ref_text = preprocess_ref_audio_text(
        os.path.join(voices_dir, meta["file"]), meta["transcript"], show_info=lambda *a: None
    )
    return model, vocoder, ref_audio, ref_text


def _speed(ref_audio: str, ref_text: str, text: str) -> float:
    """F5 sizes the output from UTF-8 byte counts, so Latin text (1 byte per letter) next to a Devanagari
    reference (3 bytes per letter) gets about a third of the time it needs and comes out garbled
    (measured). Choose the `speed` that makes F5's own estimate equal len(text) / CHARS_PER_SECOND."""
    ref_seconds = sf.info(ref_audio).duration
    estimated = ref_seconds / max(len(ref_text.encode("utf-8")), 1) * len(text.encode("utf-8"))
    wanted = max(len(text) / CHARS_PER_SECOND, 0.6)
    return float(min(max(estimated / wanted, 0.25), 3.0))


def main() -> None:
    voice_id = os.environ.get("INDICF5_VOICE", "MAR_M_WIKI_00001")
    nfe = int(os.environ.get("INDICF5_NFE", "16"))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    started = time.time()
    try:
        model, vocoder, ref_audio, ref_text = load(voice_id, device)
    except Exception as exc:  # noqa: BLE001 - report and exit; the app shows the reason
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    send({"ready": True, "load_seconds": round(time.time() - started, 1), "device": device, "voice": voice_id})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            t0 = time.time()
            text = request["text"]
            # One call for the whole text: every call re-processes the reference clip, so splitting
            # into sentences here made it ~50% slower (measured). F5 chunks long text itself, and
            # `speed` scales each chunk's length the same way.
            audio, _sr, _ = infer_process(
                ref_audio, ref_text, text, model, vocoder, mel_spec_type="vocos",
                speed=_speed(ref_audio, ref_text, text), device=device, nfe_step=nfe,
                show_info=lambda *a: None,
            )
            audio = np.asarray(audio, dtype=np.float32)
            if not np.isfinite(audio).all():
                raise RuntimeError("the model produced invalid audio")
            sf.write(request["out"], audio, SAMPLE_RATE, subtype="PCM_16")
            send({"ok": True, "seconds": round(time.time() - t0, 1), "audio_seconds": round(len(audio) / SAMPLE_RATE, 1)})
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the worker
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
