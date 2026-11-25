"""Thresholds, platform profiles and reelint.toml loading."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the tomli backport covers 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None


@dataclass(frozen=True)
class Config:
    profile: str = "reels"

    # format
    max_duration: float = 180.0  # platform hard limit, seconds
    long_duration: float = 90.0  # past this, completion rate usually suffers
    min_duration: float = 3.0
    min_short_side: int = 1080
    hard_min_short_side: int = 720
    min_fps: float = 24.0
    aspect_tolerance: float = 0.02  # allowed |w/h - 9/16|

    # hook
    hook_window: float = 1.5  # seconds a viewer gives you before swiping
    hook_min_motion: float = 0.04  # max mean-abs-diff from the first frame inside the window
    black_level: float = 0.06  # mean brightness (brightest channel, full range) treated as "black"
    sound_onset_max: float = 0.5  # seconds of silence allowed before the first sound

    # pacing
    cut_threshold: float = 0.18  # frame-to-frame change that counts as a hard cut
    jump_ratio: float = 3.0  # a frame change this many times the local median is a jump (punch-in, text pop)
    jump_floor: float = 0.03  # ...as long as it's at least this big, so grain never counts
    pop_floor: float = 0.06  # change within one grid cell that can count as a pop (caption, sticker)
    beat_threshold: float = 0.10  # drift from the last beat's frame that counts as a new visual beat
    min_beat_gap: float = 0.5  # changes closer together than this are one event (e.g. a whip pan)
    static_warn: float = 3.0  # seconds without a visual beat
    static_error: float = 6.0

    # audio
    silence_db: float = -45.0  # dBFS below which a window is silent
    dead_air_min: float = 1.0  # seconds of mid-video silence worth flagging
    lufs_min: float = -20.0
    lufs_max: float = -9.0
    true_peak_max: float = -1.0

    # ending
    tail_silence_max: float = 0.8

    disabled: frozenset[str] = field(default_factory=frozenset)


PROFILES: dict[str, dict] = {
    # Instagram Reels and YouTube Shorts both allow up to 3 minutes.
    "reels": {},
    "shorts": {},
    # TikTok uploads can run to 10 minutes; the short-form advice still applies.
    "tiktok": {"max_duration": 600.0},
}


def load_config(profile: str = "reels", path: Optional[str | Path] = None,
                disable: Optional[list[str]] = None) -> Config:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Choose from: {', '.join(PROFILES)}")
    config = replace(Config(profile=profile), **PROFILES[profile])

    if path is None and Path("reelint.toml").is_file():
        path = "reelint.toml"
    if path is not None:
        config = _apply_toml(config, Path(path))

    if disable:
        config = replace(config, disabled=config.disabled | frozenset(disable))
    return config


def _apply_toml(config: Config, path: Path) -> Config:
    if tomllib is None:
        raise ValueError("Reading reelint.toml needs tomllib (Python 3.11+) or the tomli package")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f.name: f for f in fields(Config)}
    overrides: dict = {}
    for key, value in data.get("thresholds", {}).items():
        if key not in known or key in ("profile", "disabled"):
            raise ValueError(f"{path}: unknown threshold '{key}'")
        overrides[key] = type(getattr(config, key))(value)
    disabled = config.disabled | frozenset(data.get("disable", []))
    return replace(config, **overrides, disabled=disabled)
