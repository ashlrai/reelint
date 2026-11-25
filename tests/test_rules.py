from __future__ import annotations

import json

import pytest

from reelint import analyze, lint, load_config
from reelint.cli import main
from reelint.report import score


def findings_for(video, **kwargs):
    config = load_config(**kwargs)
    return lint(analyze(video, config), config)


def rules_hit(video, **kwargs) -> set[str]:
    return {f.rule for f in findings_for(video, **kwargs)}


def test_well_cut_video_is_clean(videos):
    findings = findings_for(videos["good"])
    assert findings == [], findings
    assert score(findings) == 100


def test_detects_hard_cuts(videos):
    analysis = analyze(videos["good"])
    assert len(analysis.cuts) == 7
    assert analysis.cuts[0] == pytest.approx(1.0, abs=0.15)


def test_static_shot(videos):
    findings = findings_for(videos["static"])
    rules = {f.rule for f in findings}
    assert "hook-static" in rules
    stretch = next(f for f in findings if f.rule == "pacing-static")
    assert stretch.severity == "error"
    assert stretch.end - stretch.start == pytest.approx(8, abs=0.2)


def test_punch_ins_count_as_beats_not_cuts(videos):
    analysis = analyze(videos["punch_in"])
    assert analysis.cuts == []
    assert [round(b, 1) for b in analysis.beats[:4]] == [1.5, 3.0, 4.5, 6.0]
    assert "pacing-static" not in rules_hit(videos["punch_in"])


def test_black_open(videos):
    finding = next(f for f in findings_for(videos["black_open"]) if f.rule == "hook-black-open")
    assert finding.severity == "error"
    assert finding.end == pytest.approx(2.0, abs=0.15)


def test_black_end(videos):
    finding = next(f for f in findings_for(videos["black_end"]) if f.rule == "ending-black")
    assert finding.start == pytest.approx(6.0, abs=0.15)


def test_landscape(videos):
    finding = next(f for f in findings_for(videos["landscape"]) if f.rule == "format-aspect")
    assert finding.severity == "error"


def test_low_resolution(videos):
    finding = next(f for f in findings_for(videos["low_res"]) if f.rule == "format-resolution")
    assert finding.severity == "error"


def test_no_audio(videos):
    assert rules_hit(videos["no_audio"]) == {"audio-missing"}


def test_dead_air(videos):
    finding = next(f for f in findings_for(videos["dead_air"]) if f.rule == "audio-dead-air")
    assert finding.start == pytest.approx(3.0, abs=0.1)
    assert finding.end == pytest.approx(5.0, abs=0.1)


def test_late_sound(videos):
    finding = next(f for f in findings_for(videos["late_sound"]) if f.rule == "hook-silent-start")
    assert finding.end == pytest.approx(1.5, abs=0.1)


def test_silent_tail(videos):
    finding = next(f for f in findings_for(videos["silent_tail"]) if f.rule == "ending-silent-tail")
    assert finding.start == pytest.approx(6.0, abs=0.1)


def test_quiet_audio(videos):
    assert "audio-loudness" in rules_hit(videos["quiet"])


def test_disable_rule_and_category(videos):
    assert "pacing-static" not in rules_hit(videos["static"], disable=["pacing-static"])
    assert not {r for r in rules_hit(videos["static"], disable=["hook"]) if r.startswith("hook-")}


def test_toml_thresholds(videos, tmp_path):
    config = tmp_path / "reelint.toml"
    config.write_text('disable = ["hook-static"]\n[thresholds]\nstatic_warn = 10\nstatic_error = 20\n')
    assert rules_hit(videos["static"], path=config) == set()


def test_cli_json_and_exit_codes(videos, capsys):
    assert main(["check", str(videos["good"]), "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["score"] == 100
    assert report["stats"]["cuts"] == 7

    assert main(["check", str(videos["static"]), "--format", "json"]) == 1
    capsys.readouterr()
    assert main(["check", str(videos["static"]), "--fail-on", "never"]) == 0
    assert "pacing-static" in capsys.readouterr().out


def test_cli_missing_file(capsys):
    assert main(["check", "does-not-exist.mp4"]) == 2
    assert "File not found" in capsys.readouterr().err


def test_dark_colour_is_not_black(videos):
    # Navy is darker than black's threshold in luma, but it's a real shot, not a fade.
    assert "ending-black" not in rules_hit(videos["navy_end"])


def test_caption_pops_count_as_beats(videos):
    analysis = analyze(videos["captions"])
    assert [round(b, 1) for b in analysis.beats] == [1.0, 3.0, 5.0]
    assert analysis.cuts == []
    rules = rules_hit(videos["captions"])
    assert "hook-static" not in rules  # the caption at 1.0s is the hook


def test_whip_pan_is_one_event_not_many(videos):
    beats_in_pan = [b for b in analyze(videos["whip"]).beats if 3.0 <= b <= 4.1]
    assert 1 <= len(beats_in_pan) <= 2, beats_in_pan
