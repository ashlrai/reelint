"""The rules. Each one reads an Analysis and returns findings with timestamps and a fix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np

from .analysis import AUDIO_HOP, Analysis, runs
from .config import Config

ERROR, WARNING, INFO = "error", "warning", "info"
SEVERITY_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    message: str
    hint: str = ""
    start: Optional[float] = None  # seconds; None means the whole video
    end: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    summary: str
    check: Callable[[Analysis, Config], list[Finding]]


RULES: dict[str, Rule] = {}


def rule(rule_id: str, summary: str):
    def register(fn: Callable[[Analysis, Config], list[Finding]]):
        RULES[rule_id] = Rule(rule_id, rule_id.split("-", 1)[0], summary, fn)
        return fn
    return register


def lint(analysis: Analysis, config: Config) -> list[Finding]:
    findings: list[Finding] = []
    for rule_id, entry in RULES.items():
        if rule_id in config.disabled or entry.category in config.disabled:
            continue
        findings.extend(entry.check(analysis, config))
    return sorted(findings, key=lambda f: (f.start if f.start is not None else -1.0, SEVERITY_ORDER[f.severity]))


def fmt_time(seconds: float) -> str:
    minutes, secs = divmod(max(0.0, seconds), 60)
    return f"{int(minutes)}:{secs:04.1f}"


# ---------------------------------------------------------------- format

@rule("format-aspect", "Video should be vertical 9:16")
def _aspect(a: Analysis, c: Config) -> list[Finding]:
    w, h = a.probe.width, a.probe.height
    ratio = w / h
    if abs(ratio - 9 / 16) <= c.aspect_tolerance:
        return []
    if ratio >= 1:
        return [Finding("format-aspect", ERROR,
                        f"Video is {w}x{h} ({'landscape' if ratio > 1 else 'square'}); short-form feeds are full-screen 9:16.",
                        "Reframe to 1080x1920. A landscape video plays as a small strip with bars above and below.")]
    return [Finding("format-aspect", WARNING,
                    f"Video is {w}x{h} (ratio {ratio:.3f}); 9:16 is {9 / 16:.3f}.",
                    "Crop or pad to 1080x1920 so the app doesn't crop it for you.")]


@rule("format-resolution", "Short side should be at least 1080px")
def _resolution(a: Analysis, c: Config) -> list[Finding]:
    short = min(a.probe.width, a.probe.height)
    if short >= c.min_short_side:
        return []
    severity = ERROR if short < c.hard_min_short_side else WARNING
    return [Finding("format-resolution", severity,
                    f"Resolution is {a.probe.width}x{a.probe.height}; below {c.min_short_side}p looks soft after the platform re-encodes it.",
                    "Export at 1080x1920.")]


@rule("format-fps", "Frame rate should be at least 24fps")
def _fps(a: Analysis, c: Config) -> list[Finding]:
    if a.probe.fps <= 0 or a.probe.fps >= c.min_fps:
        return []
    return [Finding("format-fps", WARNING, f"Frame rate is {a.probe.fps:.1f}fps; motion will look choppy.",
                    "Export at 30fps (or the frame rate you shot at).")]


@rule("format-duration", "Duration should fit the platform and hold attention")
def _duration(a: Analysis, c: Config) -> list[Finding]:
    d = a.duration
    if d > c.max_duration:
        return [Finding("format-duration", ERROR,
                        f"Video is {fmt_time(d)}; the {c.profile} limit is {fmt_time(c.max_duration)}.",
                        "Trim it, or split it into a series.")]
    if d < c.min_duration:
        return [Finding("format-duration", WARNING, f"Video is only {d:.1f}s long.",
                        "Very short clips rarely carry enough to be shared or saved.")]
    if d > c.long_duration:
        return [Finding("format-duration", INFO,
                        f"Video is {fmt_time(d)}. Past {int(c.long_duration)}s, fewer viewers reach the end.",
                        "Check that every section earns its time, or cut to the strongest part.")]
    return []


# ---------------------------------------------------------------- hook

@rule("hook-black-open", "First frame should not be black")
def _black_open(a: Analysis, c: Config) -> list[Finding]:
    opening = a.brightness[a.frame_times < 0.3]
    if len(opening) == 0 or float(opening.mean()) >= c.black_level:
        return []
    lit = np.flatnonzero(a.brightness >= c.black_level)
    until = float(a.frame_times[lit[0]]) if len(lit) else a.duration
    return [Finding("hook-black-open", ERROR,
                    f"Opens on black for {until:.1f}s. The first frame is also the default cover and what people see mid-scroll.",
                    "Cut the fade-in and open on your strongest image.", 0.0, until)]


@rule("hook-static", "Something should move or change in the first 1.5s")
def _hook_static(a: Analysis, c: Config) -> list[Finding]:
    if a.hook_motion >= c.hook_min_motion or any(t < c.hook_window for t in a.beats):
        return []
    return [Finding("hook-static", WARNING,
                    f"Almost nothing changes on screen in the first {c.hook_window:g}s.",
                    "Open with motion: a cut, a push-in, a gesture or a caption popping in.", 0.0, c.hook_window)]


@rule("hook-silent-start", "Sound should start within 0.5s")
def _silent_start(a: Analysis, c: Config) -> list[Finding]:
    if a.audio_db.size == 0:
        return []
    loud = np.flatnonzero(a.audio_db > c.silence_db)
    onset = float(loud[0] * AUDIO_HOP) if len(loud) else a.duration
    if onset <= c.sound_onset_max:
        return []
    return [Finding("hook-silent-start", WARNING,
                    f"No sound for the first {onset:.1f}s.",
                    "Start the voice, music or sound effect on frame one; silence reads as a dead video.", 0.0, onset)]


# ---------------------------------------------------------------- pacing

@rule("pacing-static", "Something should change on screen every few seconds")
def _static(a: Analysis, c: Config) -> list[Finding]:
    marks = [0.0] + a.beats + [a.duration]
    findings = []
    for start, end in zip(marks, marks[1:]):
        gap = end - start
        if gap <= c.static_warn:
            continue
        severity = ERROR if gap > c.static_error else WARNING
        findings.append(Finding("pacing-static", severity,
                                f"{gap:.1f}s with no cut or visual change.",
                                "Add a pattern interrupt: a cut, a zoom, B-roll, an on-screen caption or a graphic.",
                                start, end))
    return findings


# ---------------------------------------------------------------- audio

@rule("audio-missing", "Video should have an audio track")
def _audio_missing(a: Analysis, c: Config) -> list[Finding]:
    if a.probe.has_audio:
        return []
    return [Finding("audio-missing", WARNING, "The video has no audio track.",
                    "Most viewers watch with sound on. Add voice-over or music, or plan to add a sound in the app.")]


@rule("audio-dead-air", "No long silences mid-video")
def _dead_air(a: Analysis, c: Config) -> list[Finding]:
    if a.audio_db.size == 0:
        return []
    silent = a.audio_db <= c.silence_db
    loud = np.flatnonzero(~silent)
    if len(loud) == 0:
        return [Finding("audio-dead-air", WARNING, "The audio track is silent the whole way through.",
                        "Add voice-over or music, or remove the empty track.")]
    findings = []
    for start, end in runs(silent):
        if start <= loud[0] or end >= loud[-1]:
            continue  # leading and trailing silence have their own rules
        seconds = (end - start) * AUDIO_HOP
        if seconds >= c.dead_air_min:
            findings.append(Finding("audio-dead-air", WARNING, f"{seconds:.1f}s of silence.",
                                    "Tighten the pause, or cover it with music or a sound effect.",
                                    start * AUDIO_HOP, end * AUDIO_HOP))
    return findings


@rule("audio-loudness", "Loudness near -14 LUFS without clipping")
def _loudness(a: Analysis, c: Config) -> list[Finding]:
    if a.lufs is None or a.lufs < -60:
        return []
    findings = []
    if a.lufs < c.lufs_min:
        findings.append(Finding("audio-loudness", WARNING,
                                f"Audio is quiet ({a.lufs:.1f} LUFS); it will sound weak next to other posts.",
                                "Normalize to about -14 LUFS, e.g. ffmpeg -af loudnorm=I=-14:TP=-1.5"))
    elif a.lufs > c.lufs_max:
        findings.append(Finding("audio-loudness", WARNING,
                                f"Audio is very loud ({a.lufs:.1f} LUFS); the platform will turn it down and it may sound squashed.",
                                "Normalize to about -14 LUFS, e.g. ffmpeg -af loudnorm=I=-14:TP=-1.5"))
    if a.true_peak is not None and a.true_peak > c.true_peak_max:
        findings.append(Finding("audio-loudness", WARNING,
                                f"Audio peaks at {a.true_peak:.1f} dBTP and may distort after re-encoding.",
                                "Keep true peak at or below -1 dBTP (loudnorm's TP option does this)."))
    return findings


# ---------------------------------------------------------------- ending

@rule("ending-black", "Last frame should not be black")
def _ending_black(a: Analysis, c: Config) -> list[Finding]:
    closing = a.brightness[a.frame_times >= a.frame_times[-1] - 0.3]
    if len(closing) == 0 or float(closing.mean()) >= c.black_level:
        return []
    lit = np.flatnonzero(a.brightness >= c.black_level)
    since = float(a.frame_times[lit[-1]]) + 0.1 if len(lit) else 0.0
    return [Finding("ending-black", WARNING,
                    "Ends on black. A fade-out tells viewers it's over, so they swipe instead of watching it loop.",
                    "End on a live frame, ideally one that flows back into the opening.", since, a.duration)]


@rule("ending-silent-tail", "Sound should run to the last frame")
def _silent_tail(a: Analysis, c: Config) -> list[Finding]:
    if a.audio_db.size == 0:
        return []
    loud = np.flatnonzero(a.audio_db > c.silence_db)
    if len(loud) == 0:
        return []
    last = (loud[-1] + 1) * AUDIO_HOP
    tail = a.duration - last
    if tail <= c.tail_silence_max:
        return []
    return [Finding("ending-silent-tail", WARNING, f"The last {tail:.1f}s are silent.",
                    "Trim the tail so the video loops straight back into sound.", last, a.duration)]
