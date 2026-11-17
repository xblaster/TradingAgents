"""Trading Desk: a multi-agent trading desk driven by the Claude Code CLI."""

from .config import DeskSettings
from .pipeline import DeskRun, TradingDesk

__all__ = ["DeskSettings", "DeskRun", "TradingDesk"]
__version__ = "0.1.0"
