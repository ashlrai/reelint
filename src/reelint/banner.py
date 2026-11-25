"""The reelint wordmark, shown for a bare `reelint` and for `reelint --version`."""

from __future__ import annotations

from .report import Style
from .rules import RULES

WORDMARK = """\
██████╗ ███████╗███████╗██╗     ██╗███╗   ██╗████████╗
██╔══██╗██╔════╝██╔════╝██║     ██║████╗  ██║╚══██╔══╝
██████╔╝█████╗  █████╗  ██║     ██║██╔██╗ ██║   ██║
██╔══██╗██╔══╝  ██╔══╝  ██║     ██║██║╚██╗██║   ██║
██║  ██║███████╗███████╗███████╗██║██║ ╚████║   ██║
╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝   ╚═╝"""

# Sprocket holes down a filmstrip, sized to the wordmark above.
SPROCKET = "▐▌░░" * 13 + "▐▌"


def tagline() -> str:
    return f"a linter for short-form video · {len(RULES)} rules · runs locally"


def render(style: Style) -> str:
    """The wordmark as one block of text, coloured for a terminal that wants colour."""
    strip = style(SPROCKET, "2")
    body = "\n".join(style(line, "36") for line in WORDMARK.splitlines())
    return f"{strip}\n\n{body}\n\n{strip}\n{style('  ' + tagline(), '2')}"
