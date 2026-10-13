"""A fitted volatility surface: SVI slices glued together by interpolation in total variance."""

from dataclasses import dataclass
from datetime import date, datetime
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import FloatArray
from volsurface.surface.arbitrage import ArbitrageReport, build_report
from volsurface.surface.svi import SviParams, total_variance
from volsurface.surface.svi_fit import SviFit


@dataclass(frozen=True, slots=True)
class SurfaceSlice:
    """One fitted expiry."""

    expiry: date
    t: float
    forward: float
    fit: SviFit
    k_min: float
    k_max: float

    @property
    def params(self) -> SviParams:
        return self.fit.params

    def total_variance(self, k: ArrayLike) -> FloatArray:
        return total_variance(k, self.params)

    def atm_vol(self) -> float:
        return float(np.sqrt(self.total_variance(0.0) / self.t))


@dataclass(frozen=True, slots=True)
class SkippedExpiry:
    """An expiry that could not be fitted, and why."""

    expiry: date
    reason: str


@dataclass(frozen=True, slots=True)
class VolSurface:
    """Slices ordered by maturity plus the market context they were fitted in.

    Between two slices total variance is interpolated linearly in maturity at fixed log-moneyness.
    That preserves calendar monotonicity: if two slices are ordered, so is every blend of them.
    Outside the fitted range total variance is scaled in proportion to maturity (a flat
    term-structure of implied volatility), which is an extrapolation, not a model.
    """

    underlying: str
    asof: datetime
    spot: float
    rate: float
    dividend_yield: float
    slices: tuple[SurfaceSlice, ...]
    skipped: tuple[SkippedExpiry, ...] = ()

    def __post_init__(self) -> None:
        maturities = [s.t for s in self.slices]
        if any(b <= a for a, b in pairwise(maturities)):
            raise InvalidInputError("slices must be strictly increasing in maturity")

    @property
    def maturities(self) -> FloatArray:
        return np.array([s.t for s in self.slices])

    def forward(self, t: ArrayLike) -> FloatArray:
        tt = np.asarray(t, dtype=np.float64)
        return np.asarray(self.spot * np.exp((self.rate - self.dividend_yield) * tt))

    def total_variance(self, k: ArrayLike, t: ArrayLike) -> FloatArray:
        """``w(k, t)`` with ``k`` and ``t`` broadcast against each other."""
        if not self.slices:
            raise InvalidInputError("the surface has no slices")
        kk, tt = np.broadcast_arrays(
            np.asarray(k, dtype=np.float64), np.asarray(t, dtype=np.float64)
        )
        times = self.maturities
        w_nodes = np.stack([s.total_variance(kk) for s in self.slices])
        if len(times) == 1:
            return np.asarray(w_nodes[0] * tt / times[0])
        upper = np.clip(np.searchsorted(times, tt), 1, len(times) - 1)
        lower = upper - 1
        w_lower = np.take_along_axis(w_nodes, lower[None, ...], axis=0)[0]
        w_upper = np.take_along_axis(w_nodes, upper[None, ...], axis=0)[0]
        weight = (tt - times[lower]) / (times[upper] - times[lower])
        blended = (1.0 - weight) * w_lower + weight * w_upper
        extrapolated_low = w_nodes[0] * tt / times[0]
        extrapolated_high = w_nodes[-1] * tt / times[-1]
        return np.asarray(
            np.where(
                tt < times[0],
                extrapolated_low,
                np.where(tt > times[-1], extrapolated_high, blended),
            )
        )

    def implied_vol(self, strike: ArrayLike, t: ArrayLike) -> FloatArray:
        """Implied volatility at absolute ``strike`` and maturity ``t`` in years."""
        tt = np.asarray(t, dtype=np.float64)
        k = np.log(np.asarray(strike, dtype=np.float64) / self.forward(tt))
        return np.sqrt(np.maximum(self.total_variance(k, tt), 0.0) / tt)

    def iv_grid(self, k_axis: ArrayLike, t_axis: ArrayLike) -> FloatArray:
        """Implied vols on a grid; rows follow ``t_axis`` and columns follow ``k_axis``."""
        k = np.asarray(k_axis, dtype=np.float64)
        t = np.asarray(t_axis, dtype=np.float64)
        w = self.total_variance(k[None, :], t[:, None])
        return np.sqrt(np.maximum(w, 0.0) / t[:, None])

    def arbitrage_report(self) -> ArbitrageReport:
        return build_report([(s.t, s.params, s.k_min, s.k_max) for s in self.slices])
