"""Turn a video file into the time series the rules read."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import media
from .config import Config

ANALYSIS_FPS = 10.0
AUDIO_HOP = 0.05  # seconds per loudness window


@dataclass
class Analysis:
    probe: media.Probe
    frame_times: np.ndarray  # seconds, one per sampled frame
    brightness: np.ndarray  # mean of the brightest channel per pixel, 0..1
    diff: np.ndarray  # mean abs change from the previous frame (diff[0] = 0)
    spot: np.ndarray  # change in the most-changed grid cell from the previous frame
    hook_motion: float  # largest drift from the first frame inside the hook window
    cuts: list[float]  # hard cuts
    beats: list[float]  # cuts plus gradual changes (zooms, pans, text, B-roll)
    audio_db: np.ndarray  # dBFS per AUDIO_HOP window; empty without audio
    lufs: Optional[float]
    true_peak: Optional[float]

    @property
    def duration(self) -> float:
        return self.probe.duration or (float(self.frame_times[-1]) + 1 / ANALYSIS_FPS if len(self.frame_times) else 0.0)

    def audio_time(self, index: int) -> float:
        return index * AUDIO_HOP


def analyze(path: str | Path, config: Optional[Config] = None) -> Analysis:
    config = config or Config()
    info = media.probe(path)
    rgb = media.frames(info.path, ANALYSIS_FPS)
    if len(rgb) == 0:
        raise media.MediaError(f"No frames could be decoded from {info.path}")

    times = np.arange(len(rgb)) / ANALYSIS_FPS
    # Brightness = the brightest channel per pixel, so saturated dark colours
    # (navy, deep red) aren't mistaken for black the way luma would.
    brightness = rgb.max(axis=3).mean(axis=(1, 2)) / 255.0
    # Change is measured on all three channels, so a cut between two shots of
    # similar brightness but different colour still registers.
    diff = np.zeros(len(rgb), dtype=np.float32)
    spot = np.zeros(len(rgb), dtype=np.float32)
    for i in range(1, len(rgb)):
        diff[i], spot[i] = change(rgb[i], rgb[i - 1]), spot_change(rgb[i], rgb[i - 1])

    hook_count = int(np.count_nonzero(times < config.hook_window))
    hook_motion = max((change(rgb[i], rgb[0]) for i in range(hook_count)), default=0.0)

    cuts, beats = _beats(rgb, diff, spot, times, config)

    if info.has_audio:
        samples = media.audio(info.path)
        audio_db = _window_db(samples)
        lufs, true_peak = media.loudness(info.path)
    else:
        audio_db, lufs, true_peak = np.zeros(0), None, None

    return Analysis(probe=info, frame_times=times, brightness=brightness, diff=diff, spot=spot,
                    hook_motion=hook_motion, cuts=cuts, beats=beats, audio_db=audio_db, lufs=lufs,
                    true_peak=true_peak)


GRID_COLS, GRID_ROWS = 4, 8  # cells of 18x16 px on the 72x128 analysis frame


def change(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two uint8 frames, 0..1."""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) / 255.0


def spot_change(a: np.ndarray, b: np.ndarray) -> float:
    """Change in the most-changed grid cell, 0..1.

    A caption popping in alters a small part of the frame strongly; averaged
    over the whole frame it disappears, so we look cell by cell.
    """
    per_pixel = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=2)
    h, w = per_pixel.shape
    cells = per_pixel.reshape(GRID_ROWS, h // GRID_ROWS, GRID_COLS, w // GRID_COLS).mean(axis=(1, 3))
    return float(cells.max()) / 255.0


def _local_median(series: np.ndarray, i: int, half: int) -> float:
    neighbours = np.concatenate([series[max(1, i - half):i], series[i + 1:i + 1 + half]])
    return float(np.median(neighbours)) if neighbours.size else 0.0


def _beats(frames: np.ndarray, diff: np.ndarray, spot: np.ndarray, times: np.ndarray,
           config: Config) -> tuple[list[float], list[float]]:
    """A beat is anything a viewer would register as "something changed".

    Four ways to qualify:
    - cut: a large frame-to-frame change.
    - jump: a whole-frame change that spikes well above the motion around it
      (a punch-in zoom, a cut between similar shots).
    - pop: one region of the frame changes sharply against its usual motion
      (a caption or sticker appearing, a new line of text).
    - drift: gradual change (a pan, a slow zoom) that has added up to a clear
      difference from the frame of the last beat.

    Jumps and pops are judged against the local median, so a busy handheld
    shot doesn't count every frame. Qualifying frames less than min_beat_gap
    apart belong to one event (a whip pan triggers on every frame it spans),
    so a new beat needs a pause in the action first.
    """
    half = int(round(1.5 * ANALYSIS_FPS))
    cuts: list[float] = []
    beats: list[float] = []
    anchor = frames[0]
    last_trigger = float("-inf")
    for i in range(1, len(frames)):
        t = float(times[i])
        is_cut = diff[i] >= config.cut_threshold
        is_jump = diff[i] >= config.jump_floor and diff[i] >= config.jump_ratio * _local_median(diff, i, half)
        is_pop = spot[i] >= config.pop_floor and spot[i] >= config.jump_ratio * _local_median(spot, i, half)
        if not (is_cut or is_jump or is_pop or change(frames[i], anchor) >= config.beat_threshold):
            continue
        anchor = frames[i]
        same_event = t - last_trigger < config.min_beat_gap
        last_trigger = t
        if same_event:
            continue
        beats.append(t)
        if is_cut:
            cuts.append(t)
    return cuts, beats


def _window_db(samples: np.ndarray) -> np.ndarray:
    hop = int(media.AUDIO_RATE * AUDIO_HOP)
    count = len(samples) // hop
    if count == 0:
        return np.zeros(0)
    windows = samples[: count * hop].reshape(count, hop)
    rms = np.sqrt((windows.astype(np.float64) ** 2).mean(axis=1))
    return 20 * np.log10(rms + 1e-10)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask is True."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))
