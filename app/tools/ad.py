"""Scene-based video ads (user request, 2026-09-28): "make a 20 second ad for ZeroGo" becomes a short commercial -
several different shots with crossfades, a caption on each, a voice-over and background music - in one mp4.

The script (scenes with what is seen, a caption and a spoken line, plus a music style) is written by the chat model
(Planner._plan_ad, or the model calling create_ad itself). Then, one step at a time so memory stays safe:
1. the voice-over lines are spoken by the reply voice (Kokoro, on the CPU) - each scene is made long enough for its
   line;
2. each scene is one clip from the video tool (app/tools/video.py, through SCENE_DIR: its automatic switch-on, Stop
   and fallback all apply) - at most 5 seconds, one clip of the video model;
3. the video model is parked and ACE-Step makes an instrumental jingle (app/tools/song.py's worker);
4. captions are drawn with PIL (this ffmpeg has no drawtext) and one ffmpeg run joins everything.
The chat model is unloaded for the whole ad and loaded again at the end. A missing voice-over or jingle does not
fail the ad (it is made without it); a scene that cannot be made does.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.tools.base import Tool, ToolResult, current_conversation, current_progress_reporter, current_request
from app.tools.image import _slugify, unique_export_path
from app.tools.image_safety import check_minor_safety
from app.tools.video import SCENE_DIR

logger = logging.getLogger("jarvis.tools.ad")

MAX_SCENES = 6
MIN_SCENE_SECONDS = 2.0
MAX_SCENE_SECONDS = 5.0  # one clip of the video model
MAX_AD_SECONDS = 40.0
FADE_SECONDS = 0.4
FPS = 24
DEFAULT_MUSIC = "upbeat modern background music for an advert, bright synths and light drums, no vocals"
_STOPPED = "Ad stopped. Nothing was saved."
_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
)


class AdCancelled(Exception):
    """The user stopped the ad being made (Stop, or typing/saying "stop" - POST /api/ads/cancel)."""


@dataclass
class Scene:
    visual: str
    seconds: float
    text: str = ""
    voiceover: str = ""


@dataclass
class AdScript:
    title: str
    music: str
    language: str
    scenes: list[Scene]

    @property
    def planned_seconds(self) -> float:
        return sum(s.seconds for s in self.scenes) - FADE_SECONDS * max(0, len(self.scenes) - 1)


def _text(value: Any, limit: int) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def parse_script(arguments: dict[str, Any]) -> AdScript | str:
    """The script from the tool's arguments, cleaned and kept within this Mac's limits; a str says what is wrong."""
    raw = arguments.get("scenes")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    if not isinstance(raw, list) or not raw:
        return "create_ad needs 'scenes': a list of {visual, seconds, text, voiceover}."
    scenes: list[Scene] = []
    for item in raw[:MAX_SCENES]:
        if not isinstance(item, dict):
            continue
        visual = _text(item.get("visual") or item.get("prompt"), 600)
        if not visual:
            continue
        try:
            seconds = float(item.get("seconds") or 4)
        except (TypeError, ValueError):
            seconds = 4.0
        scenes.append(Scene(visual, min(max(seconds, MIN_SCENE_SECONDS), MAX_SCENE_SECONDS),
                            _text(item.get("text"), 60), _text(item.get("voiceover"), 200)))
    if not scenes:
        return "create_ad needs at least one scene with a 'visual' description."
    while len(scenes) > 1 and sum(s.seconds for s in scenes) > MAX_AD_SECONDS:
        scenes.pop()
    title = _text(arguments.get("title"), 80) or scenes[0].visual[:40]
    music = _text(arguments.get("music"), 300) or DEFAULT_MUSIC
    language = _text(arguments.get("language"), 8).lower() or "en"
    return AdScript(title, music, language, scenes)


_QUOTED = re.compile(r"[\"“”‘’][^\"“”‘’]{1,60}[\"“”‘’]")
_NO_TEXT = ", no text, no letters, no logos on screen"
_TEXTY = re.compile(r"\b(?:the |a |an |its |some )?(?:logos?|text|texts|lettering|letters|words|signs?|signage|slogans?|"
                    r"captions?|titles?|brand names?)\b", re.I)


def clean_visual(visual: str, script: AdScript) -> str:
    """What the video model is asked to draw, without the words it cannot draw: video models turn a product name
    into garbled letters (the first real ad showed "Zirraagg" on a phone). The name is in the captions and the
    voice-over instead. Unusual words of the title and captions (not in the English dictionary) and quoted text
    are taken out."""
    from app.agent.planner import _english_words

    words = _english_words()
    names = {w for w in re.findall(r"[A-Za-z]{4,}", " ".join([script.title, *(s.text for s in script.scenes)]))
             if words and w.lower() not in words}
    cleaned = _QUOTED.sub("", visual)
    for name in names:
        cleaned = re.sub(rf"\b{re.escape(name)}\b['’]?s?", "", cleaned, flags=re.I)
    # "showing the logo and the text" -> something a video model can draw
    cleaned = _TEXTY.sub("colorful graphics", re.sub(r"\s{2,}", " ", cleaned))
    cleaned = re.sub(r"colorful graphics(\s*(,|and|with|or)\s*colorful graphics)+", "colorful graphics", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.")
    return cleaned + _NO_TEXT


# ---------------------------------------------------------------------------- ffmpeg helpers (plain, testable)
def media_seconds(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, timeout=30).stdout.strip()
    return float(out) if out else 0.0


def video_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0:s=x", str(path)], capture_output=True, text=True, timeout=30).stdout.strip()
    width, height = out.split("x")[:2]
    return int(width), int(height)


def _font(size: int):
    from PIL import ImageFont

    for path in _FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def render_caption(text: str, width: int, height: int, path: Path) -> Path:
    """A transparent frame with the caption on a soft dark band near the bottom (centred, wrapped)."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    size = max(14, int(min(width, height) * 0.075))
    font = _font(size)
    lines: list[str] = []
    for word in text.split():
        if lines and draw.textlength(f"{lines[-1]} {word}", font=font) <= width * 0.88:
            lines[-1] = f"{lines[-1]} {word}"
        else:
            lines.append(word)
    line_height = int(size * 1.25)
    band = line_height * len(lines) + int(size * 0.8)
    top = height - band - int(height * 0.06)
    draw.rectangle([0, top, width, top + band], fill=(0, 0, 0, 155))
    for i, line in enumerate(lines):
        x = (width - draw.textlength(line, font=font)) / 2
        draw.text((x, top + int(size * 0.4) + i * line_height), line, font=font, fill=(255, 255, 255, 255))
    image.save(path)
    return path


def assemble(clips: list[Path], durations: list[float], captions: list[Path | None], music: Path | None,
             voices: list[tuple[Path, float]], out: Path, size: tuple[int, int], fade: float = FADE_SECONDS) -> float:
    """One ffmpeg run: the clips (trimmed to `durations`) with crossfades and captions, the music under the
    voice-over lines (each at its start time). Returns the ad's length in seconds."""
    width, height = size
    n = len(clips)
    fade = min(fade, min(durations) / 2) if n > 1 else 0.0
    args = ["ffmpeg", "-y", "-loglevel", "error"]
    for clip in clips:
        args += ["-i", str(clip)]
    caption_input: dict[int, int] = {}
    for i, caption in enumerate(captions):
        if caption is not None:
            caption_input[i] = n + len(caption_input)
            args += ["-loop", "1", "-t", f"{durations[i]:.3f}", "-i", str(caption)]
    next_input = n + len(caption_input)
    music_input = None
    if music is not None:
        music_input, next_input = next_input, next_input + 1
        args += ["-i", str(music)]
    voice_inputs = []
    for path, start in voices:
        voice_inputs.append((next_input, start))
        next_input += 1
        args += ["-i", str(path)]

    graph = []
    for i in range(n):
        chain = (f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,"
                 f"fps={FPS},trim=0:{durations[i]:.3f},setpts=PTS-STARTPTS,format=yuv420p")
        if i in caption_input:
            graph.append(f"{chain}[b{i}]")
            graph.append(f"[b{i}][{caption_input[i]}:v]overlay=0:0:shortest=1,format=yuv420p,settb=AVTB[v{i}]")
        else:
            graph.append(f"{chain},settb=AVTB[v{i}]")
    last, offset = "v0", 0.0
    for i in range(1, n):
        offset += durations[i - 1] - fade
        graph.append(f"[{last}][v{i}]xfade=transition=fade:duration={fade:.3f}:offset={offset:.3f}[x{i}]")
        last = f"x{i}"
    total = sum(durations) - fade * (n - 1)

    audio = []
    stereo = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    if music_input is not None:
        level = 0.5 if voice_inputs else 0.9  # under the voice-over, still clearly there
        graph.append(f"[{music_input}:a]{stereo},atrim=0:{total:.3f},asetpts=PTS-STARTPTS,volume={level},"
                     f"afade=t=in:d=0.6,afade=t=out:st={max(0.0, total - 1.5):.3f}:d=1.5[music]")
        audio.append("[music]")
    for k, (index, start) in enumerate(voice_inputs):
        delay = int(start * 1000)
        graph.append(f"[{index}:a]{stereo},adelay=delays={delay}:all=1,volume=1.6[voice{k}]")
        audio.append(f"[voice{k}]")
    if audio:
        mixed = audio[0] if len(audio) == 1 else f"{''.join(audio)}amix=inputs={len(audio)}:duration=longest:normalize=0"
        if len(audio) == 1:
            graph.append(f"{mixed}apad,atrim=0:{total:.3f}[aout]")
        else:
            graph.append(f"{mixed},apad,atrim=0:{total:.3f}[aout]")

    args += ["-filter_complex", ";".join(graph), "-map", f"[{last}]"]
    if audio:
        args += ["-map", "[aout]", "-c:a", "aac", "-b:a", "160k"]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-t", f"{total:.3f}", str(out)]
    done = subprocess.run(args, capture_output=True, text=True, timeout=900)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg could not put the ad together: {done.stderr.strip()[-400:]}")
    return total


# ---------------------------------------------------------------------------- the tool
class CreateAdTool(Tool):
    name = "create_ad"
    description = (
        "Make a short video ad (advertisement, commercial, promo) - several different scenes joined with crossfades, "
        "a caption on each, a voice-over and background music, in one video. Use it when the user asks for an ad, "
        "advertisement, commercial or promo video. `scenes`: 3 to 5 items, each {visual: a vivid English description "
        "of what is seen - subject, action, setting, camera, light - with no words or logos in the picture; seconds: "
        "2 to 5; text: a short caption, at most 6 words; voiceover: one short spoken line}. `music`: the background "
        "music style (instrumental). Making an ad takes a long time on this machine (each scene is a video clip) - say so briefly."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "A short name for the ad."},
            "music": {"type": "string", "description": "Background music style, instrumental."},
            "language": {"type": "string", "description": "Language code of the voice-over, e.g. en or hi."},
            "scenes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "visual": {"type": "string"},
                        "seconds": {"type": "number"},
                        "text": {"type": "string"},
                        "voiceover": {"type": "string"},
                    },
                    "required": ["visual"],
                },
            },
        },
        "required": ["scenes"],
    }

    def __init__(self, video_tool: Any, exports_dir: Path, public_url: str, *, generation_lock: asyncio.Lock,
                 tts: Any = None, song_worker: Any = None, llm_memory: Any = None, media: Any = None,
                 video_pipelines: Any = None) -> None:
        self._video = video_tool
        self._exports = Path(exports_dir)
        self._public = public_url.rstrip("/")
        self._lock = generation_lock
        self._tts = tts
        self._song = song_worker
        self._llm_memory = llm_memory
        self._media = media
        self._video_pipelines = video_pipelines
        self._cancelled = False
        self.generating = False

    def describe(self, arguments: dict[str, Any]) -> str:
        title = arguments.get("title")
        return f"Making an ad: {title.strip()[:60]}" if isinstance(title, str) and title.strip() else "Making an ad"

    def cancel(self) -> bool:
        """Stops the ad being made (its scene or its music). False when no ad is being made."""
        if not self.generating:
            return False
        self._cancelled = True
        if self._video_pipelines is not None:
            self._video_pipelines.cancel()
        if self._song is not None:
            self._song.cancel()
        logger.info("Ad: stop requested by the user")
        return True

    def _check(self) -> None:
        if self._cancelled:
            raise AdCancelled()

    async def execute(self, **arguments: Any) -> ToolResult:
        script = parse_script(arguments)
        if isinstance(script, str):
            return ToolResult.failure(script)
        everything = "\n".join([script.title, script.music] + [f"{s.visual}\n{s.text}\n{s.voiceover}" for s in script.scenes])
        refusal = check_minor_safety(everything)  # the existing always-on minor check, as for images and videos
        if refusal:
            return ToolResult.failure(refusal)
        if shutil.which("ffmpeg") is None:
            return ToolResult.failure("ffmpeg is required to put an ad together (brew install ffmpeg).")

        reporter = current_progress_reporter.get()
        started = time.monotonic()

        def report(percent: float, label: str) -> None:
            if reporter is not None:
                reporter({"step": max(0, min(99, round(percent))), "total_steps": 100, "label": label,
                          "elapsed_seconds": round(time.monotonic() - started, 1), "eta_seconds": None})

        n = len(script.scenes)
        self._cancelled = False
        self.generating = True
        tmp = Path(tempfile.mkdtemp(prefix="zira-ad-"))
        try:
            if self._llm_memory is not None:
                await self._llm_memory.release()  # once for the whole ad (the scenes leave it alone)

            # 1. The voice-over, first: each scene is made long enough for its line.
            voice_files: dict[int, Path] = {}
            if self._tts is not None:
                for i, scene in enumerate(script.scenes):
                    if not scene.voiceover:
                        continue
                    report(1 + 3 * i / n, "Recording the voice-over")
                    try:
                        wav = await asyncio.wait_for(self._tts.synthesize(scene.voiceover), timeout=90)
                        path = tmp / f"voice-{i}.wav"
                        path.write_bytes(wav)
                        spoken = await asyncio.to_thread(media_seconds, path)
                        scene.seconds = min(MAX_SCENE_SECONDS, max(scene.seconds, spoken + 0.5))
                        voice_files[i] = path
                    except Exception as exc:  # noqa: BLE001 - the ad is made without this line
                        logger.warning("Ad voice-over line %d not recorded: %s", i + 1, exc)
            self._check()

            # 2. The scenes, one clip each, through the video tool.
            clips: list[Path] = []
            for i, scene in enumerate(script.scenes):
                base, span = 5 + 70 * i / n, 70 / n
                report(base, f"Scene {i + 1} of {n}")

                def scene_progress(info: dict, i=i, base=base, span=span) -> None:
                    if info.get("stage") == "loading":
                        report(base, f"Scene {i + 1} of {n} · setting up the video model")
                    elif info.get("total_steps"):
                        report(base + span * info["step"] / info["total_steps"], f"Scene {i + 1} of {n}")

                tokens = (SCENE_DIR.set(tmp), current_progress_reporter.set(scene_progress), current_request.set(""))
                try:
                    result = await self._video.execute(prompt=clean_visual(scene.visual, script), seconds=scene.seconds,
                                                       title=f"scene-{i + 1}")
                finally:
                    current_request.reset(tokens[2])
                    current_progress_reporter.reset(tokens[1])
                    SCENE_DIR.reset(tokens[0])
                self._check()
                if not result.ok:
                    return ToolResult.failure(f"Scene {i + 1} of the ad could not be made: {result.error}")
                clips.append(Path(result.files[0]["url"]))

            # 3. The jingle: the video model is parked first, so ACE-Step has the memory.
            durations = [min(scene.seconds, await asyncio.to_thread(media_seconds, clip) or scene.seconds)
                         for scene, clip in zip(script.scenes, clips)]
            fade = min(FADE_SECONDS, min(durations) / 2) if n > 1 else 0.0
            total = sum(durations) - fade * (n - 1)
            music = None
            if self._song is not None:
                report(76, "Making the music")
                from app.api.videos import park_video_model
                from app.tools.song import SongCancelled

                await park_video_model(self._video_pipelines, "the ad's music needs the memory")
                try:
                    async with self._lock:
                        await self._song.make(
                            {"lyrics": "[Instrumental]", "caption": script.music, "language": "unknown",
                             "seconds": max(10.0, total + 1.0), "out": str(tmp / "music.mp3")},
                            lambda fraction, _desc: report(76 + 19 * fraction, "Making the music"),
                        )
                    music = tmp / "music.mp3"
                except SongCancelled as exc:
                    raise AdCancelled() from exc
                except Exception as exc:  # noqa: BLE001 - the ad is made without music
                    logger.warning("Ad music not made, putting the ad together without it: %s", exc)
            self._check()

            # 4. Captions, voice timing, one ffmpeg run.
            report(96, "Putting the ad together")
            size = await asyncio.to_thread(video_size, clips[0])
            captions = [
                await asyncio.to_thread(render_caption, scene.text, size[0], size[1], tmp / f"caption-{i}.png")
                if scene.text else None
                for i, scene in enumerate(script.scenes)
            ]
            starts = [sum(durations[:i]) - fade * i for i in range(n)]
            voices = [(voice_files[i], starts[i] + 0.25) for i in sorted(voice_files)]
            self._exports.mkdir(parents=True, exist_ok=True)
            stem = _slugify(script.title)
            stem = stem if stem == "ad" or stem.endswith("-ad") else f"{stem}-ad"  # never "...-ad-ad"
            target = unique_export_path(self._exports, stem, ".mp4")
            length = await asyncio.to_thread(assemble, clips, durations, captions, music, voices, target, size, fade)
        except AdCancelled:
            logger.info("Ad stopped by the user; nothing was saved")
            return ToolResult.failure(_STOPPED)
        except Exception as exc:  # noqa: BLE001 - any other failure maps to one clear error
            logger.exception("Ad failed")
            return ToolResult.failure(f"The ad could not be made: {exc}")
        finally:
            self.generating = False
            if self._llm_memory is not None:
                await self._llm_memory.restore()
            shutil.rmtree(tmp, ignore_errors=True)

        logger.info("Ad made: %s (%d scenes, %.1fs, music=%s, voice lines=%d) in %.0fs", target.name, n, length,
                    music is not None, len(voices), time.monotonic() - started)
        if self._media is not None:
            self._media.record(filename=target.name, kind="video", prompt=json.dumps(
                [{"visual": s.visual, "text": s.text, "voiceover": s.voiceover} for s in script.scenes])[:4000],
                request=current_request.get() or "", model="ad", width=size[0], height=size[1],
                seconds=round(length, 2), conversation_id=current_conversation.get(), source="ad")
        url = f"{self._public}/api/exports/{target.name}"
        return ToolResult.success(f'Made a {length:.0f}-second ad: "{script.title}".', files=[{"title": target.name, "url": url}])
