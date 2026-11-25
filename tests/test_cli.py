"""The wordmark, and the CLI contracts around it."""

from __future__ import annotations

import pytest

from reelint import banner
from reelint.cli import main
from reelint.report import Style
from reelint.rules import RULES


def test_tagline_counts_the_real_rules():
    assert f"{len(RULES)} rules" in banner.tagline()


def test_render_is_plain_without_colour():
    text = banner.render(Style(False))
    assert "\033[" not in text
    assert "██████╗" in text and banner.tagline() in text


def test_render_colours_every_wordmark_line():
    lines = banner.render(Style(True)).splitlines()
    coloured = [line for line in lines if line.startswith("\033[36m")]
    assert len(coloured) == len(banner.WORDMARK.splitlines())


def test_sprocket_matches_the_wordmark_width():
    widest = max(len(line) for line in banner.WORDMARK.splitlines())
    assert len(banner.SPROCKET) == widest


def test_version_is_plain_when_piped(capsys):
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines() == ["reelint 0.1.0"]


def test_bare_command_prints_help_not_art(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage: reelint" in out
    assert "██" not in out


def test_unknown_command_still_fails(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code == 2
