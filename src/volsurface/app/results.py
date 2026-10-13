"""Value objects produced by the processing pipeline and consumed by the user interface."""

from dataclasses import dataclass, field

from volsurface.analytics.alerts import Alert, MarketSummary
from volsurface.analytics.gex import GexProfile
from volsurface.domain.types import BoolArray, ChainSnapshot, FloatArray
from volsurface.surface.surface import VolSurface


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Wall-clock seconds spent in each pipeline stage."""

    filter_s: float = 0.0
    iv_s: float = 0.0
    surface_s: float = 0.0
    gex_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.filter_s + self.iv_s + self.surface_s + self.gex_s

    def items(self) -> tuple[tuple[str, float], ...]:
        return (
            ("filter", self.filter_s),
            ("implied_vol", self.iv_s),
            ("surface", self.surface_s),
            ("gex", self.gex_s),
        )


@dataclass(frozen=True, slots=True, eq=False)
class SessionUpdate:
    """Everything the user interface needs to draw one frame.

    ``implied_vol`` holds our own volatility for each chain row (``nan`` where the row was
    filtered out), next to ``chain.vendor_iv`` so the two can be compared.
    """

    seq: int
    source: str
    chain: ChainSnapshot
    usable: BoolArray
    implied_vol: FloatArray
    surface: VolSurface | None
    gex: GexProfile | None
    summary: MarketSummary | None
    alerts: tuple[Alert, ...]
    timings: StageTimings
    warnings: tuple[str, ...] = field(default=())
