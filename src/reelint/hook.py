"""Profile the opening seconds of a video: the features most likely to drive skip rate.

This doesn't predict a skip rate. It measures the hook consistently so that,
logged next to real skip rates (see `write_csv`), you can see which hook
features go with lower skips for your own content.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from . import media, vision
from .analysis import AUDIO_HOP, Analysis, analyze
from .config import Config

HOOK_FPS = 4.0  # opening frames read for text and faces
MIN_TEXT_HEIGHT = 0.015  # ignore text smaller than this fraction of frame height (code on screens, watermarks)

# Check thresholds
TEXT_BY = 0.5  # text should be on screen by this time
CHANGE_BY = 1.5
SOUND_BY = 0.5
SMALL_TEXT = 0.025  # caption line height below this is hard to read on a phone
READABLE_WORDS_PER_SECOND = 4.0
QUIET_HOOK_DB = 3.0  # hook this many dB quieter than the rest of the video


@dataclass(frozen=True)
class Check:
    id: str
    status: str  # pass | warn | info | unknown
    message: str


@dataclass
class HookProfile:
    file: str
    window: float
    duration: float
    opens_on_black: bool
    first_change_at: Optional[float]
    changes_in_window: int
    has_audio: bool
    sound_starts_at: Optional[float]
    hook_db_vs_rest: Optional[float]
    vision: bool
    text_on_first_frame: Optional[bool] = None
    first_text_at: Optional[float] = None
    hook_text: list[str] = field(default_factory=list)
    hook_words: Optional[int] = None
    largest_text: Optional[float] = None
    first_face_at: Optional[float] = None
    largest_face: Optional[float] = None
    checks: list[Check] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["checks"] = [asdict(c) for c in self.checks]
        return data


def profile(path: str | Path, config: Optional[Config] = None, window: float = 3.0,
            use_vision: bool = True, analysis: Optional[Analysis] = None) -> HookProfile:
    config = config or Config()
    a = analysis or analyze(path, config)
    window = min(window, a.duration) if a.duration else window

    opening = a.brightness[a.frame_times < 0.3]
    beats = [t for t in a.beats if t < window]
    hook = HookProfile(
        file=a.probe.path, window=round(window, 2), duration=round(a.duration, 2),
        opens_on_black=bool(len(opening) and opening.mean() < config.black_level),
        first_change_at=beats[0] if beats else None, changes_in_window=len(beats),
        has_audio=a.audio_db.size > 0, sound_starts_at=None, hook_db_vs_rest=None,
        vision=use_vision and vision.available(),
    )
    _audio_features(hook, a, config, window)
    if hook.vision:
        _vision_features(hook, a.probe, window)
    hook.checks = _checks(hook, a, config)
    return hook


def _power_mean_db(db: np.ndarray) -> Optional[float]:
    if db.size == 0:
        return None
    return float(10 * np.log10(np.mean(10 ** (db / 10)) + 1e-20))


def _audio_features(hook: HookProfile, a: Analysis, config: Config, window: float) -> None:
    if a.audio_db.size == 0:
        return
    loud = np.flatnonzero(a.audio_db > config.silence_db)
    if len(loud):
        hook.sound_starts_at = round(float(loud[0] * AUDIO_HOP), 2)
    split = int(window / AUDIO_HOP)
    head, rest = _power_mean_db(a.audio_db[:split]), _power_mean_db(a.audio_db[split:])
    if head is not None and rest is not None and rest > config.silence_db:
        hook.hook_db_vs_rest = round(head - rest, 1)


def _vision_features(hook: HookProfile, info: media.Probe, window: float) -> None:
    frames = media.opening_frames(info, HOOK_FPS, window)
    lines: list[tuple[str, str]] = []  # (normalised, as shown), in order of first appearance
    largest_text, largest_face = 0.0, 0.0
    for index, frame in enumerate(frames):
        t = round(index / HOOK_FPS, 2)
        boxes = [b for b in vision.read_text(frame) if b.height >= MIN_TEXT_HEIGHT]
        if boxes:
            if hook.first_text_at is None:
                hook.first_text_at = t
            largest_text = max(largest_text, max(b.height for b in boxes))
            for box in sorted(boxes, key=lambda b: b.center_y):
                _add_line(lines, box.text)
        faces = vision.find_faces(frame)
        if faces:
            if hook.first_face_at is None:
                hook.first_face_at = t
            largest_face = max(largest_face, max(f.size for f in faces))
        if index == 0:
            hook.text_on_first_frame = bool(boxes)
    hook.hook_text = [shown for _, shown in lines]
    hook.hook_words = sum(len(shown.split()) for shown in hook.hook_text)
    hook.largest_text = round(largest_text, 3) if largest_text else None
    hook.largest_face = round(largest_face, 3) if largest_face else None


def _add_line(lines: list[tuple[str, str]], text: str) -> None:
    """Keep each caption once. Captions often reveal word by word ("YOU NEED" →
    "YOU NEED THIS"), so a line that extends an earlier one replaces it."""
    norm = vision.normalise(text)
    if not norm:
        return
    for i, (seen, _) in enumerate(lines):
        if norm in seen:
            return
        if seen in norm:
            lines[i] = (norm, text)
            return
    lines.append((norm, text))


def _checks(h: HookProfile, a: Analysis, config: Config) -> list[Check]:
    checks: list[Check] = []

    def add(check_id: str, status: str, message: str) -> None:
        checks.append(Check(check_id, status, message))

    if h.opens_on_black:
        add("open-frame", "warn", "Opens on black: the first frame is blank.")
    else:
        add("open-frame", "pass", "Opens on a live frame.")

    if not h.vision:
        add("text", "unknown", "Text not checked (pip install reelint[vision]).")
    elif h.first_text_at is None:
        add("text", "warn", f"No readable text in the first {h.window:g}s; muted viewers get no hook.")
    elif h.first_text_at <= TEXT_BY:
        add("text", "pass", f"Text on screen from {h.first_text_at:g}s.")
    else:
        add("text", "warn", f"First text appears at {h.first_text_at:g}s; aim for under {TEXT_BY:g}s.")

    if h.vision and h.largest_text is not None and h.largest_text < SMALL_TEXT:
        add("text-size", "warn", f"Largest text is {h.largest_text * 100:.1f}% of the screen height; hard to read on a phone.")

    if h.hook_words is not None and h.hook_words > READABLE_WORDS_PER_SECOND * h.window:
        add("text-length", "warn",
            f"{h.hook_words} words of text in {h.window:g}s; most people read about {READABLE_WORDS_PER_SECOND:g} words a second.")

    moving = a.hook_motion >= config.hook_min_motion
    if h.first_change_at is not None and h.first_change_at <= CHANGE_BY:
        add("change", "pass", f"First visual change at {h.first_change_at:g}s ({h.changes_in_window} in {h.window:g}s).")
    elif moving:
        add("change", "pass", "The shot is moving from the start.")
    elif h.first_change_at is not None:
        add("change", "warn", f"First visual change at {h.first_change_at:g}s; aim for under {CHANGE_BY:g}s.")
    else:
        add("change", "warn", f"Nothing changes on screen in the first {h.window:g}s.")

    if not h.has_audio:
        add("sound", "warn", "No audio track.")
    elif h.sound_starts_at is None:
        add("sound", "warn", "The audio is silent.")
    elif h.sound_starts_at <= SOUND_BY:
        add("sound", "pass", f"Sound from {h.sound_starts_at:g}s.")
    else:
        add("sound", "warn", f"Sound starts at {h.sound_starts_at:g}s; aim for under {SOUND_BY:g}s.")

    if h.hook_db_vs_rest is not None and h.hook_db_vs_rest <= -QUIET_HOOK_DB:
        add("sound-level", "warn", f"The hook is {-h.hook_db_vs_rest:.1f} dB quieter than the rest of the video.")

    if not h.vision:
        pass
    elif h.first_face_at is not None:
        add("face", "info", f"Face on screen from {h.first_face_at:g}s.")
    else:
        add("face", "info", f"No face in the first {h.window:g}s (fine for screen recordings and B-roll).")
    return checks


# ---------------------------------------------------------------- output

SYMBOL = {"pass": "✓", "warn": "!", "info": "·", "unknown": "?"}
STYLE = {"pass": "32", "warn": "33", "info": "2", "unknown": "2"}


def to_text(h: HookProfile, style) -> str:
    lines = [f"{style(h.file, '1')}  ·  hook (first {h.window:g}s)"]
    for c in h.checks:
        lines.append(f"  {style(SYMBOL[c.status], STYLE[c.status])} {c.message}")
    if h.hook_text:
        lines.append("")
        lines.append("  Hook text:  " + style(" / ".join(f'"{t}"' for t in h.hook_text), "1"))
    return "\n".join(lines)


CSV_COLUMNS = ["file", "analyzed", "duration", "opens_on_black", "first_change_at", "changes_in_window",
               "sound_starts_at", "hook_db_vs_rest", "text_on_first_frame", "first_text_at", "hook_words",
               "largest_text_pct", "first_face_at", "hook_text", "skip_rate"]


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def csv_row(h: HookProfile) -> dict[str, str]:
    return {
        "file": str(Path(h.file).resolve()),
        "analyzed": datetime.now().strftime("%Y-%m-%d"),
        "duration": _cell(h.duration),
        "opens_on_black": _cell(h.opens_on_black),
        "first_change_at": _cell(h.first_change_at),
        "changes_in_window": _cell(h.changes_in_window),
        "sound_starts_at": _cell(h.sound_starts_at),
        "hook_db_vs_rest": _cell(h.hook_db_vs_rest),
        "text_on_first_frame": _cell(h.text_on_first_frame),
        "first_text_at": _cell(h.first_text_at),
        "hook_words": _cell(h.hook_words),
        "largest_text_pct": _cell(round(h.largest_text * 100, 1) if h.largest_text else None),
        "first_face_at": _cell(h.first_face_at),
        "hook_text": " / ".join(h.hook_text),
    }


def write_csv(path: str | Path, profiles: list[HookProfile]) -> None:
    """Add or refresh rows in a hook log, keeping anything the user typed in
    (skip_rate, and any extra columns such as views or notes)."""
    path = Path(path)
    rows: dict[str, dict[str, str]] = {}
    columns = list(CSV_COLUMNS)
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in columns:
                    columns.append(name)
            for row in reader:
                rows[row.get("file", "")] = row
    for h in profiles:
        fresh = csv_row(h)
        rows[fresh["file"]] = {**rows.get(fresh["file"], {}), **fresh}
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows.values():
            writer.writerow({c: row.get(c, "") for c in columns})
