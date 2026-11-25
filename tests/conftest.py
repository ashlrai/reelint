"""Synthetic test videos, generated once per session with FFmpeg's built-in sources."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

if not shutil.which("ffmpeg"):
    pytest.skip("FFmpeg is required for the test suite", allow_module_level=True)

VERTICAL = "1080x1920"
# 1kHz tone at amplitude 0.3 measures roughly -13 LUFS.
TONE = "0.3*sin(2*PI*1000*t)"


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def _encode(out: Path, video_filter: str, audio: str | None, duration: float) -> Path:
    args = ["-f", "lavfi", "-i", video_filter]
    if audio:
        args += ["-f", "lavfi", "-i", f"aevalsrc='{audio}':s=48000:d={duration}"]
    args += ["-t", str(duration), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-r", "30"]
    args += ["-c:a", "aac", "-shortest"] if audio else ["-an"]
    _ffmpeg(*args, str(out))
    return out


def _colour_cuts(size: str, colours: list[str], shot: float) -> str:
    """A lavfi graph that hard-cuts between solid colours every `shot` seconds."""
    inputs = "".join(f"color=c={c}:s={size}:r=30:d={shot}[v{i}];" for i, c in enumerate(colours))
    labels = "".join(f"[v{i}]" for i in range(len(colours)))
    return f"{inputs}{labels}concat=n={len(colours)}:v=1:a=0"


PALETTE = ["red", "navy", "yellow", "teal", "white", "purple", "orange", "green"]


@pytest.fixture(scope="session")
def videos(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("videos")
    out: dict[str, Path] = {}
    out["good"] = _encode(root / "good.mp4", _colour_cuts(VERTICAL, PALETTE, 1.0), TONE, 8)
    out["static"] = _encode(root / "static.mp4", f"color=c=gray:s={VERTICAL}:r=30:d=8", TONE, 8)
    out["black_open"] = _encode(root / "black_open.mp4",
                                _colour_cuts(VERTICAL, ["black", "black"] + PALETTE[:6], 1.0), TONE, 8)
    out["black_end"] = _encode(root / "black_end.mp4",
                               _colour_cuts(VERTICAL, PALETTE[:6] + ["black", "black"], 1.0), TONE, 8)
    out["navy_end"] = _encode(root / "navy_end.mp4", _colour_cuts(VERTICAL, PALETTE[:5] + ["navy"] * 3, 1.0), TONE, 8)
    out["landscape"] = _encode(root / "landscape.mp4", _colour_cuts("1920x1080", PALETTE[:4], 1.0), TONE, 4)
    out["low_res"] = _encode(root / "low_res.mp4", _colour_cuts("540x960", PALETTE[:4], 1.0), TONE, 4)
    out["no_audio"] = _encode(root / "no_audio.mp4", _colour_cuts(VERTICAL, PALETTE, 1.0), None, 8)
    out["dead_air"] = _encode(root / "dead_air.mp4", _colour_cuts(VERTICAL, PALETTE, 1.0),
                              f"{TONE}*(lt(t,3)+gte(t,5))", 8)
    out["late_sound"] = _encode(root / "late_sound.mp4", _colour_cuts(VERTICAL, PALETTE, 1.0),
                                f"{TONE}*gte(t,1.5)", 8)
    out["silent_tail"] = _encode(root / "silent_tail.mp4", _colour_cuts(VERTICAL, PALETTE, 1.0),
                                 f"{TONE}*lt(t,6)", 8)
    out["quiet"] = _encode(root / "quiet.mp4", _colour_cuts(VERTICAL, PALETTE, 1.0),
                           "0.01*sin(2*PI*1000*t)", 8)
    # Punch-in zooms every 1.5s (+15% each): no hard cuts, but the picture keeps jumping.
    zoom = (f"testsrc2=s={VERTICAL}:r=30:d=8,scale=2160:3840,"
            f"zoompan=z='1+0.15*floor(on/45)':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={VERTICAL}:fps=30")
    out["punch_in"] = _encode(root / "punch_in.mp4", zoom, TONE, 8)
    # One locked-off shot where only a caption-sized box changes every 2s.
    captions = (f"color=c=gray:s={VERTICAL}:r=30:d=8,"
                "drawbox=x=200:y=1300:w=680:h=120:color=white:t=fill:enable='between(t,1,3)',"
                "drawbox=x=150:y=1300:w=780:h=120:color=yellow:t=fill:enable='between(t,3,5)',"
                "drawbox=x=250:y=1300:w=580:h=120:color=white:t=fill:enable='gte(t,5)'")
    out["captions"] = _encode(root / "captions.mp4", captions, TONE, 8)
    # A one-second whip pan across a wide test pattern, static either side.
    whip = "testsrc2=s=3240x1920:r=30:d=8,crop=1080:1920:x='min(max((t-3)*2160,0),2160)':y=0"
    out["whip"] = _encode(root / "whip.mp4", whip, TONE, 8)
    return out
