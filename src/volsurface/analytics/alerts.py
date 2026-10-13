"""Alert rules on GEX and volatility, as small composable strategy objects.

Three rules (the same as the notebook this project re-implements) compare the current market
summary of an underlying with the previous one. Tickers whose GEX never exceeds ``min_gex`` anywhere
on the grid are too small to be worth an alert and are skipped by the engine, not by each rule.
"""

import enum
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from volsurface.analytics.gex import GexProfile, atm_implied_vol
from volsurface.domain.types import ChainSnapshot, FloatArray


class AlertKind(enum.StrEnum):
    GEX_SIGN_CHANGE = "gex_sign_change"
    NEAR_FLIP = "near_flip"
    IV_JUMP = "iv_jump"


@dataclass(frozen=True, slots=True)
class MarketSummary:
    """The numbers the alert rules look at, for one underlying at one instant."""

    underlying: str
    asof: datetime
    spot: float
    gex_at_spot: float
    flip: float
    distance_to_flip_pct: float
    atm_iv_pct: float
    gex_size: float

    @classmethod
    def from_profile(
        cls, chain: ChainSnapshot, profile: GexProfile, implied_vol: FloatArray
    ) -> "MarketSummary":
        return cls(
            underlying=chain.underlying,
            asof=chain.asof,
            spot=chain.spot,
            gex_at_spot=profile.gex_at_spot,
            flip=profile.flip,
            distance_to_flip_pct=profile.distance_to_flip_pct,
            atm_iv_pct=atm_implied_vol(chain, implied_vol),
            gex_size=profile.size,
        )


@dataclass(frozen=True, slots=True)
class Alert:
    kind: AlertKind
    underlying: str
    asof: datetime
    message: str
    value: float


class AlertRule(Protocol):
    """One rule: what changed between ``previous`` (possibly absent) and ``current``?"""

    def evaluate(
        self, previous: MarketSummary | None, current: MarketSummary
    ) -> Iterable[Alert]: ...


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


class GexSignChange:
    """GEX at spot changed sign since the previous update."""

    def evaluate(self, previous: MarketSummary | None, current: MarketSummary) -> Iterable[Alert]:
        if previous is None or _sign(previous.gex_at_spot) == _sign(current.gex_at_spot):
            return
        direction = "negative" if current.gex_at_spot < 0 else "positive"
        yield Alert(
            AlertKind.GEX_SIGN_CHANGE,
            current.underlying,
            current.asof,
            f"{current.underlying} GEX flipped {direction} ({current.gex_at_spot:+.3f} bn$/1%)",
            current.gex_at_spot,
        )


class NearFlip:
    """Spot moved into the band around the flip level (reported at first sight too)."""

    def __init__(self, band_pct: float = 0.5) -> None:
        self._band = band_pct

    def _near(self, summary: MarketSummary | None) -> bool:
        return summary is not None and abs(summary.distance_to_flip_pct) < self._band

    def evaluate(self, previous: MarketSummary | None, current: MarketSummary) -> Iterable[Alert]:
        if self._near(current) and not self._near(previous):
            yield Alert(
                AlertKind.NEAR_FLIP,
                current.underlying,
                current.asof,
                f"{current.underlying} within {self._band}% of its flip "
                f"({current.distance_to_flip_pct:+.2f}%)",
                current.distance_to_flip_pct,
            )


class IvJump:
    """At-the-money implied vol moved by more than ``points`` volatility points."""

    def __init__(self, points: float = 2.0) -> None:
        self._points = points

    def evaluate(self, previous: MarketSummary | None, current: MarketSummary) -> Iterable[Alert]:
        if previous is None:
            return
        move = current.atm_iv_pct - previous.atm_iv_pct
        if math.isfinite(move) and abs(move) > self._points:
            yield Alert(
                AlertKind.IV_JUMP,
                current.underlying,
                current.asof,
                f"{current.underlying} ATM vol moved {move:+.2f} points to "
                f"{current.atm_iv_pct:.2f}%",
                move,
            )


class AlertEngine:
    """Applies every rule to each update and gates out underlyings with negligible gamma."""

    def __init__(self, rules: Sequence[AlertRule], *, min_gex: float = 0.05) -> None:
        self._rules = tuple(rules)
        self._min_gex = min_gex

    @classmethod
    def default(
        cls, *, near_flip_pct: float = 0.5, iv_jump_points: float = 2.0, min_gex: float = 0.05
    ) -> "AlertEngine":
        return cls(
            [GexSignChange(), NearFlip(near_flip_pct), IvJump(iv_jump_points)], min_gex=min_gex
        )

    def evaluate(self, previous: MarketSummary | None, current: MarketSummary) -> tuple[Alert, ...]:
        if not current.gex_size > self._min_gex:
            return ()
        return tuple(alert for rule in self._rules for alert in rule.evaluate(previous, current))
