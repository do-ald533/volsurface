"""Quote filters built with the Specification pattern.

Each rule is a small object that turns a :class:`~volsurface.domain.types.ChainSnapshot` into a
boolean mask. Rules compose with ``&``, ``|`` and ``~`` so a policy reads like the sentence that
describes it::

    spec = TwoSided() & MinTimeValue(0.05) & MaxDaysToExpiry(120)
    usable = snapshot.select(spec.mask(snapshot))

Adding a rule means adding a class; nothing that already exists changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from volsurface.domain.calendar import MARKET_TZ
from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import BoolArray, ChainSnapshot


class QuoteSpec(ABC):
    """A predicate over the rows of a chain."""

    @abstractmethod
    def mask(self, chain: ChainSnapshot) -> BoolArray:
        """True for rows that satisfy the rule."""

    def __and__(self, other: "QuoteSpec") -> "QuoteSpec":
        return _AllOf((self, other))

    def __or__(self, other: "QuoteSpec") -> "QuoteSpec":
        return _AnyOf((self, other))

    def __invert__(self) -> "QuoteSpec":
        return _Not(self)


@dataclass(frozen=True, slots=True)
class _AllOf(QuoteSpec):
    parts: tuple[QuoteSpec, ...]

    def mask(self, chain: ChainSnapshot) -> BoolArray:
        result = np.ones(chain.size, dtype=np.bool_)
        for part in self.parts:
            result &= part.mask(chain)
        return result


@dataclass(frozen=True, slots=True)
class _AnyOf(QuoteSpec):
    parts: tuple[QuoteSpec, ...]

    def mask(self, chain: ChainSnapshot) -> BoolArray:
        result = np.zeros(chain.size, dtype=np.bool_)
        for part in self.parts:
            result |= part.mask(chain)
        return result


@dataclass(frozen=True, slots=True)
class _Not(QuoteSpec):
    inner: QuoteSpec

    def mask(self, chain: ChainSnapshot) -> BoolArray:
        return ~self.inner.mask(chain)


@dataclass(frozen=True, slots=True)
class AcceptAll(QuoteSpec):
    """Null Object: keeps every row."""

    def mask(self, chain: ChainSnapshot) -> BoolArray:
        return np.ones(chain.size, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class TwoSided(QuoteSpec):
    """Both bid and ask are positive."""

    def mask(self, chain: ChainSnapshot) -> BoolArray:
        return chain.is_two_sided


@dataclass(frozen=True, slots=True)
class HasOpenInterest(QuoteSpec):
    """Open interest is known and at least ``minimum`` contracts."""

    minimum: float = 1.0

    def mask(self, chain: ChainSnapshot) -> BoolArray:
        with np.errstate(invalid="ignore"):
            return np.asarray(chain.open_interest >= self.minimum)


@dataclass(frozen=True, slots=True)
class MinTimeValue(QuoteSpec):
    """Mid price exceeds intrinsic value by more than ``minimum`` dollars.

    Options priced at intrinsic value carry no information about volatility. Intrinsic value is
    measured against the spot price (undiscounted), as in the notebook this rule comes from.
    """

    minimum: float = 0.05

    def mask(self, chain: ChainSnapshot) -> BoolArray:
        intrinsic = np.maximum(chain.right_sign * (chain.spot - chain.strike), 0.0)
        with np.errstate(invalid="ignore"):
            return np.asarray(chain.mid - intrinsic > self.minimum)


@dataclass(frozen=True, slots=True)
class MaxDaysToExpiry(QuoteSpec):
    """Expires within ``days`` calendar days of the snapshot's New York date."""

    days: int

    def __post_init__(self) -> None:
        if self.days < 0:
            raise InvalidInputError("days must be non-negative")

    def mask(self, chain: ChainSnapshot) -> BoolArray:
        today = np.datetime64(chain.asof.astimezone(MARKET_TZ).date(), "D")
        remaining = (chain.expiry - today).astype("timedelta64[D]").astype(np.int64)
        return (remaining >= 0) & (remaining <= self.days)


@dataclass(frozen=True, slots=True)
class MoneynessBand(QuoteSpec):
    """Strike within ``[low, high] * spot`` (for example 0.9 to 1.1)."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not 0 < self.low < self.high:
            raise InvalidInputError("need 0 < low < high")

    def mask(self, chain: ChainSnapshot) -> BoolArray:
        ratio = chain.strike / chain.spot
        return (ratio >= self.low) & (ratio <= self.high)


def surface_quote_spec(max_days: int = 120, min_time_value: float = 0.05) -> QuoteSpec:
    """Default rule for implied-volatility work: two-sided, informative and not too far out."""
    return TwoSided() & MinTimeValue(min_time_value) & MaxDaysToExpiry(max_days)


def gex_quote_spec(max_days: int = 120, min_time_value: float = 0.05) -> QuoteSpec:
    """Default rule for GEX: the surface rule plus known, positive open interest."""
    return surface_quote_spec(max_days, min_time_value) & HasOpenInterest(1.0)
