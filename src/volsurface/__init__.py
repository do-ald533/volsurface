"""Volsurface Desk: implied-volatility surface workstation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("volsurface")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
