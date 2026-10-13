"""The application layer's vocabulary for choosing a market-data source.

The user interface depends on this facade rather than on the market-data adapters themselves.
"""

from volsurface.marketdata.factory import KINDS, SourceKind, SourceSpec

__all__ = ["KINDS", "SourceKind", "SourceSpec"]
