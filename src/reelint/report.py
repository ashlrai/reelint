"""Scoring and output: a terminal report for people, JSON for scripts and agents."""

from __future__ import annotations

import os
import sys

import numpy as np

from .analysis import AUDIO_HOP, Analysis
from .rules import ERROR, INFO, WARNING, Finding, fmt_time

PENALTY = {ERROR: 15, WARNING: 5, INFO: 0}
RULE_PENALTY_CAP = 25  # one noisy rule can't sink the whole score
BLOCKS = " ▁▂▃▄▅▆▇█"
TIMELINE_WIDTH = 60


def score(findings: list[Finding]) -> int:
    per_rule: dict[str, int] = {}
    for f in findings:
        per_rule[f.rule] = per_rule.get(f.rule, 0) + PENALTY[f.severity]
    return max(0, 100 - sum(min(p, RULE_PENALTY_CAP) for p in per_rule.values()))


def stats(a: Analysis) -> dict:
    shots = [0.0] + a.cuts + [a.duration]
    lengths = np.diff(shots)
    return {
        "cuts": len(a.cuts),
        "beats": len(a.beats),
        "avg_shot_seconds": round(float(lengths.mean()), 2) if len(lengths) else None,
        "beats_per_10s": round(len(a.beats) / a.duration * 10, 1) if a.duration else None,
        "lufs": None if a.lufs is None or a.lufs == float("-inf") else round(a.lufs, 1),
        "true_peak_dbtp": None if a.true_peak is None or a.true_peak == float("-inf") else round(a.true_peak, 1),
    }


def to_json(a: Analysis, findings: list[Finding]) -> dict:
    return {
        "file": a.probe.path,
        "duration": round(a.duration, 2),
        "width": a.probe.width,
        "height": a.probe.height,
        "fps": round(a.probe.fps, 2),
        "score": score(findings),
        "counts": {s: sum(f.severity == s for f in findings) for s in (ERROR, WARNING, INFO)},
        "stats": stats(a),
        "findings": [f.to_dict() for f in findings],
    }


class Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    @classmethod
    def for_stream(cls, stream) -> "Style":
        if os.environ.get("NO_COLOR") or not getattr(stream, "isatty", lambda: False)():
            return cls(False)
        if sys.platform == "win32":
            os.system("")  # switches the Windows console into ANSI mode
        return cls(True)


SEVERITY_STYLE = {ERROR: "1;31", WARNING: "33", INFO: "36"}


def _span(f: Finding) -> str:
    if f.start is None:
        return "whole video"
    if f.end is None or f.end - f.start < 0.05:
        return fmt_time(f.start)
    return f"{fmt_time(f.start)}–{fmt_time(f.end)}"


def _bucket(values: np.ndarray, times: np.ndarray, duration: float, reduce) -> list[float]:
    edges = np.linspace(0, duration, TIMELINE_WIDTH + 1)
    out = []
    for lo, hi in zip(edges, edges[1:]):
        chunk = values[(times >= lo) & (times < hi)]
        out.append(float(reduce(chunk)) if chunk.size else np.nan)
    return out


def _spark(levels: list[float]) -> str:
    chars = []
    for level in levels:
        if np.isnan(level):
            chars.append(" ")
        else:
            chars.append(BLOCKS[int(round(min(1.0, max(0.0, level)) * (len(BLOCKS) - 1)))])
    return "".join(chars)


def timeline(a: Analysis, findings: list[Finding], style: Style) -> list[str]:
    d = a.duration or 1.0
    picture = _bucket(a.diff, a.frame_times, d, np.max)
    lines = ["  picture " + style(_spark([v / 0.2 for v in picture]), "32")]
    if a.audio_db.size:
        audio_times = np.arange(a.audio_db.size) * AUDIO_HOP
        sound = _bucket(a.audio_db, audio_times, d, np.mean)
        lines.append("  sound   " + style(_spark([(v + 60) / 50 for v in sound]), "34"))
    issues = [" "] * TIMELINE_WIDTH
    for f in findings:
        if f.start is None or f.severity == INFO:
            continue
        lo = int(f.start / d * TIMELINE_WIDTH)
        hi = max(lo + 1, int(np.ceil((f.end if f.end is not None else f.start) / d * TIMELINE_WIDTH)))
        mark = "E" if f.severity == ERROR else "W"
        for i in range(max(0, lo), min(TIMELINE_WIDTH, hi)):
            if issues[i] != "E":
                issues[i] = mark
    marked = "".join(style(ch, SEVERITY_STYLE[ERROR if ch == "E" else WARNING]) if ch != " " else ch for ch in issues)
    lines.append("  issues  " + marked)
    lines.append("          " + style(f"0:00{fmt_time(d).rjust(TIMELINE_WIDTH - 4)}", "2"))
    return lines


def to_text(a: Analysis, findings: list[Finding], style: Style) -> str:
    s = stats(a)
    value = score(findings)
    colour = "1;32" if value >= 85 else "1;33" if value >= 60 else "1;31"
    header = (f"{style(a.probe.path, '1')}  ·  {fmt_time(a.duration)}  ·  "
              f"{a.probe.width}x{a.probe.height} @ {a.probe.fps:.0f}fps  ·  score {style(f'{value}/100', colour)}")
    facts = [f"{s['cuts']} cut{'' if s['cuts'] == 1 else 's'}", f"{s['beats_per_10s']} visual beats per 10s"]
    if s["avg_shot_seconds"] is not None:
        facts.append(f"avg shot {s['avg_shot_seconds']}s")
    if s["lufs"] is not None:
        facts.append(f"{s['lufs']} LUFS")
    lines = [header, style("  " + "  ·  ".join(facts), "2"), ""]
    lines += timeline(a, findings, style)
    lines.append("")

    if not findings:
        lines.append(style("  No issues found.", "32"))
    width = max((len(_span(f)) for f in findings), default=0)
    rule_width = max((len(f.rule) for f in findings), default=0)
    for f in findings:
        sev = style(f.severity.ljust(7), SEVERITY_STYLE[f.severity])
        lines.append(f"  {_span(f).ljust(width)}  {sev}  {style(f.rule.ljust(rule_width), '2')}  {f.message}")
        if f.hint:
            lines.append(" " * (width + rule_width + 15) + style("→ " + f.hint, "2"))

    counts = {sev: sum(f.severity == sev for f in findings) for sev in (ERROR, WARNING, INFO)}
    lines += ["", f"  {counts[ERROR]} errors, {counts[WARNING]} warnings, {counts[INFO]} info"]
    return "\n".join(lines)
