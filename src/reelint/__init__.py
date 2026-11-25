"""reelint: a linter for short-form video."""

from .analysis import Analysis, analyze
from .config import Config, load_config
from .rules import RULES, Finding, lint

__version__ = "0.1.0"
__all__ = ["Analysis", "Config", "Finding", "RULES", "analyze", "lint", "load_config", "__version__"]
