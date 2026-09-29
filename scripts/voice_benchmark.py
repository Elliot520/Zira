"""Real Hinglish STT benchmark against a live, running JARVIS server.

Not a pytest test (those are hermetic/mocked, see tests/test_voice.py) - this hits the real
/api/voice/transcribe endpoint, so it needs `./run.sh` already running in another terminal. Uses
macOS `say` to synthesize each test sentence (same technique as README "Testing voice without a
microphone"), so it's repeatable without an actual microphone - it measures the STT engine, not
your voice specifically, but is genuinely useful for comparing engines/models/config on real audio
rather than trusting docs or benchmarks run on someone else's hardware.

Usage:
    ./run.sh                                    # in one terminal
    .venv/bin/python scripts/voice_benchmark.py  # in another

Requires `say` (macOS) and `ffmpeg` (brew install ffmpeg) on PATH.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib import error, request

SERVER = "http://127.0.0.1:8000"

# Phase 2/20 Hinglish code-switching test set: natural mixed Hindi-English sentences a real
# bilingual speaker would say, covering commands, questions, corrections and casual chat - not
# formal/textbook Hindi. "Expected" is what a human would consider a faithful transcript (script
# mixing is fine and expected; translating to pure English is the failure this set watches for).
TEST_SENTENCES = [
    "Haan bhai, aaj kya plan hai?",
    "JARVIS, kal mujhe office jaana hai.",
    "Can you check mera calendar?",
    "Achha, so what should we do now?",
    "Mujhe ek Android app banana hai using Kotlin.",
    "Ye code thoda slow chal raha hai, can you check it?",
    "Tomorrow morning mujhe remind kar dena.",
    "Actually wait, mujhe kal nahi parso jaana hai.",
    "Haan JARVIS, kya scene hai?",
    "Can you remind me tomorrow morning?",
    "Wait, nahi, mera matlab ye nahi tha.",
    "Isko thoda optimize kar sakte ho?",
    "Don't change the existing architecture.",
    "Bas mujhe simple answer do.",
]


def synthesize(text: str, out_dir: Path, index: int) -> Path:
    aiff = out_dir / f"clip{index}.aiff"
    webm = out_dir / f"clip{index}.webm"
    subprocess.run(["say", "-v", "Rishi", "-o", str(aiff), text], check=True, capture_output=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff), "-ar", "16000", "-ac", "1", "-c:a", "libopus", str(webm)],
        check=True, capture_output=True,
    )
    return webm


def transcribe(clip: Path) -> tuple[str, float]:
    """Returns (text, server-reported duration_ms) - the server's own timing (STT + transcript
    cleanup, which is on by default: VOICE_TRANSCRIPT_CLEANUP_ENABLED) rather than wall-clock, so
    results aren't skewed by this script's own localhost network overhead."""
    boundary = "----voicebenchmark"
    with open(clip, "rb") as f:
        audio_bytes = f.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"clip.webm\"\r\n"
        f"Content-Type: audio/webm\r\n\r\n"
    ).encode() + audio_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = request.Request(
        f"{SERVER}/api/voice/transcribe", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            import json

            payload = json.loads(resp.read())
    except error.URLError as exc:
        raise RuntimeError(f"Is ./run.sh running? Could not reach {SERVER}: {exc}") from exc
    return payload.get("text", ""), float(payload.get("duration_ms", 0))


def main() -> None:
    try:
        request.urlopen(f"{SERVER}/api/voice/status", timeout=3)
    except error.URLError:
        print(f"Server not reachable at {SERVER} - start it with ./run.sh first.", file=sys.stderr)
        sys.exit(1)

    print(f"{'Original':<52} {'Transcribed':<52} {'ms':>7}")
    print("-" * 115)
    latencies_ms = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i, sentence in enumerate(TEST_SENTENCES):
            clip = synthesize(sentence, tmp_path, i)
            text, duration_ms = transcribe(clip)
            latencies_ms.append(duration_ms)
            print(f"{sentence:<52} {text:<52} {duration_ms:>7.0f}")

    print("-" * 115)
    print(f"Clips: {len(latencies_ms)}  Avg: {sum(latencies_ms) / len(latencies_ms):.0f}ms  "
          f"Min: {min(latencies_ms):.0f}ms  Max: {max(latencies_ms):.0f}ms")
    print("\nRead the two columns yourself for accuracy - this script deliberately does not compute")
    print("a fake pass/fail score; judging whether code-switching/meaning was preserved needs a human.")


if __name__ == "__main__":
    main()
