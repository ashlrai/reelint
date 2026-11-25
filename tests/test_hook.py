from __future__ import annotations

import csv
import subprocess

import numpy as np
import pytest

from reelint import hook, vision
from reelint.cli import main

needs_vision = pytest.mark.skipif(not vision.available(), reason="needs reelint[vision]")


@pytest.fixture(scope="session")
def text_hook(tmp_path_factory):
    """Gray background; "WAIT FOR IT" appears at 1.0s, a second caption at 2.0s; sound from 0.8s."""
    cv2 = pytest.importorskip("cv2")
    out = tmp_path_factory.mktemp("hook") / "text_hook.mp4"
    fps, seconds, w, h = 30, 4, 1080, 1920
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
         "-i", "-", "-f", "lavfi", "-i", f"aevalsrc='0.3*sin(2*PI*1000*t)*gte(t,0.8)':s=48000:d={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
        stdin=subprocess.PIPE)
    for i in range(fps * seconds):
        t = i / fps
        frame = np.full((h, w, 3), 90, dtype=np.uint8)
        if t >= 1.0:
            cv2.putText(frame, "WAIT FOR IT", (120, 700), cv2.FONT_HERSHEY_DUPLEX, 3.2, (255, 255, 255), 8)
        if t >= 2.0:
            cv2.putText(frame, "THIS IS HUGE", (110, 900), cv2.FONT_HERSHEY_DUPLEX, 3.2, (255, 255, 0), 8)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    return out


@needs_vision
def test_reads_hook_text_and_timing(text_hook):
    h = hook.profile(text_hook)
    assert h.text_on_first_frame is False
    assert h.first_text_at == pytest.approx(1.0, abs=0.25)
    assert [vision.normalise(t) for t in h.hook_text] == ["wait for it", "this is huge"]
    assert h.hook_words == 6
    assert h.first_change_at == pytest.approx(1.0, abs=0.15)
    assert h.sound_starts_at == pytest.approx(0.8, abs=0.06)
    statuses = {c.id: c.status for c in h.checks}
    assert statuses["text"] == "warn"  # text at 1.0s is later than the 0.5s target
    assert statuses["sound"] == "warn"
    assert statuses["change"] == "pass"


def test_without_vision_text_is_unknown(text_hook):
    h = hook.profile(text_hook, use_vision=False)
    assert h.vision is False and h.hook_text == [] and h.first_text_at is None
    assert {c.id: c.status for c in h.checks}["text"] == "unknown"


def test_black_open_is_flagged(videos):
    h = hook.profile(videos["black_open"], use_vision=False)
    assert h.opens_on_black
    assert {c.id: c.status for c in h.checks}["open-frame"] == "warn"


def test_word_by_word_reveal_keeps_the_full_line():
    lines: list[tuple[str, str]] = []
    for text in ["YOU NEED", "YOU NEED THIS", "you need", "AI TOOL"]:
        hook._add_line(lines, text)
    assert [shown for _, shown in lines] == ["YOU NEED THIS", "AI TOOL"]


def test_csv_log_keeps_what_the_user_typed(text_hook, tmp_path):
    log = tmp_path / "hooks.csv"
    profile = hook.profile(text_hook, use_vision=False)
    hook.write_csv(log, [profile])

    with log.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["skip_rate"] = "41"
    with log.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) + ["views"])
        writer.writeheader()
        writer.writerow({**rows[0], "views": "12000"})

    hook.write_csv(log, [profile])  # re-run
    with log.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["skip_rate"] == "41" and rows[0]["views"] == "12000"
    assert rows[0]["sound_starts_at"] == "0.8"


def test_cli_hook_json(text_hook, capsys):
    assert main(["hook", str(text_hook), "--format", "json", "--no-vision"]) == 0
    import json
    data = json.loads(capsys.readouterr().out)
    assert data["sound_starts_at"] == pytest.approx(0.8, abs=0.06)
    assert {c["id"] for c in data["checks"]} >= {"open-frame", "text", "change", "sound"}
