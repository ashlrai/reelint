"""Thin wrappers around ffprobe/ffmpeg. Everything is decoded through pipes; nothing is written to disk."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Frames are downscaled hard before analysis: we measure change, not detail,
# and a small frame averages away sensor noise and compression grain.
FRAME_W, FRAME_H = 72, 128
AUDIO_RATE = 16000


class MediaError(Exception):
    """The file can't be read, or FFmpeg isn't installed."""


def _tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(f"{name} was not found on PATH. Install FFmpeg: https://ffmpeg.org/download.html")
    return path


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, check=False)


def _rate(value: Optional[str]) -> float:
    if not value or value in ("0/0", "N/A"):
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(value)


def _rotation(stream: dict) -> int:
    rotate = stream.get("tags", {}).get("rotate")
    if rotate is not None:
        return abs(int(float(rotate)))
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            return abs(int(float(side["rotation"])))
    return 0


@dataclass(frozen=True)
class Probe:
    path: str
    duration: float
    width: int  # as displayed, after rotation metadata
    height: int
    fps: float
    has_audio: bool


def probe(path: str | Path) -> Probe:
    path = str(path)
    if not Path(path).is_file():
        raise MediaError(f"File not found: {path}")
    result = _run([_tool("ffprobe"), "-v", "error", "-print_format", "json",
                   "-show_streams", "-show_format", path])
    if result.returncode != 0:
        raise MediaError(f"Could not read {path}: {result.stderr.decode(errors='replace').strip()}")
    data = json.loads(result.stdout or b"{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise MediaError(f"{path} has no video stream")
    width, height = int(video["width"]), int(video["height"])
    if _rotation(video) % 180 == 90:
        width, height = height, width
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return Probe(path=path, duration=duration, width=width, height=height, fps=fps, has_audio=has_audio)


def frames(path: str, fps: float) -> np.ndarray:
    """RGB frames sampled at `fps`, shape (n, FRAME_H, FRAME_W, 3), uint8 (about 27KB per frame)."""
    vf = f"fps={fps},scale={FRAME_W}:{FRAME_H}:flags=area,format=rgb24"
    result = _run([_tool("ffmpeg"), "-v", "error", "-i", path, "-an", "-vf", vf,
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    if result.returncode != 0:
        raise MediaError(f"Could not decode video from {path}: {result.stderr.decode(errors='replace').strip()}")
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    size = FRAME_W * FRAME_H * 3
    count = raw.size // size
    return raw[: count * size].reshape(count, FRAME_H, FRAME_W, 3)


def opening_frames(info: Probe, fps: float, seconds: float, width: int = 540) -> np.ndarray:
    """RGB uint8 frames from the first `seconds`, scaled to `width` keeping the aspect ratio."""
    height = max(2, int(round(width * info.height / info.width / 2)) * 2)
    vf = f"fps={fps},scale={width}:{height}:flags=area,format=rgb24"
    result = _run([_tool("ffmpeg"), "-v", "error", "-i", info.path, "-t", str(seconds), "-an", "-vf", vf,
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    if result.returncode != 0:
        raise MediaError(f"Could not decode video from {info.path}: {result.stderr.decode(errors='replace').strip()}")
    size = width * height * 3
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    count = raw.size // size
    return raw[: count * size].reshape(count, height, width, 3)


def audio(path: str) -> np.ndarray:
    """Mono samples at AUDIO_RATE, values -1..1. Empty if the file has no audio."""
    result = _run([_tool("ffmpeg"), "-v", "error", "-i", path, "-vn", "-ac", "1",
                   "-ar", str(AUDIO_RATE), "-f", "s16le", "-"])
    if result.returncode != 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


_INTEGRATED = re.compile(r"I:\s+(-?[\d.]+|-inf)\s+LUFS")
_PEAK = re.compile(r"Peak:\s+(-?[\d.]+|-inf)\s+dBFS")


def loudness(path: str) -> tuple[Optional[float], Optional[float]]:
    """(integrated LUFS, true peak dBTP) per EBU R128, or (None, None) if unmeasurable."""
    result = _run([_tool("ffmpeg"), "-hide_banner", "-nostats", "-i", path, "-vn",
                   "-af", "ebur128=peak=true", "-f", "null", "-"])
    text = result.stderr.decode(errors="replace")
    summary = text.rsplit("Summary:", 1)[-1] if "Summary:" in text else ""

    def parse(pattern: re.Pattern) -> Optional[float]:
        match = pattern.search(summary)
        if not match:
            return None
        return float("-inf") if match.group(1) == "-inf" else float(match.group(1))

    return parse(_INTEGRATED), parse(_PEAK)
