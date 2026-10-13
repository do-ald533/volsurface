"""Gamma exposure (GEX) and its flip level.

Convention (taken from the GEX notebook this project re-implements; conventions differ between
data providers): calls count positively and puts negatively, as if dealers were long calls and
short puts. Each option contributes ``sign * open_interest * multiplier * 0.01 / 1e9 * gamma * S^2``
at underlying price ``S``, so GEX is expressed in billions of dollars per 1 % move. Gamma comes from
the American CRR tree, so it reflects early exercise and dividends.

The profile is evaluated on a grid of spot levels around the current price. The **flip level** is
the grid price nearest to spot where the profile changes sign, located by linear interpolation
between the two grid points.
"""

from dataclasses import dataclass

import numpy as np

from volsurface.domain.calendar import years_to_expiry
from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import ChainSnapshot, FloatArray
from volsurface.pricing.backends import PricingBackend
from volsurface.pricing.quote_filter import QuoteSpec

DEFAULT_GRID_LOW = 0.90
DEFAULT_GRID_HIGH = 1.10
DEFAULT_GRID_POINTS = 81
DOLLARS_PER_BILLION = 1e9


@dataclass(frozen=True, slots=True)
class GexProfile:
    """GEX of one underlying across a grid of spot levels."""

    underlying: str
    spot: float
    levels: FloatArray
    gex: FloatArray
    n_options: int

    @property
    def gex_at_spot(self) -> float:
        return float(self.gex[self.gex.size // 2])

    @property
    def flip(self) -> float:
        return flip_level(self.levels, self.gex, self.spot)

    @property
    def distance_to_flip_pct(self) -> float:
        """How far spot is above (+) or below (-) the flip level, in percent of the flip."""
        flip = self.flip
        return (self.spot / flip - 1.0) * 100.0 if np.isfinite(flip) else float("nan")

    @property
    def size(self) -> float:
        """Largest absolute GEX anywhere on the grid; small values mean "not worth alerting"."""
        return float(np.max(np.abs(self.gex))) if self.gex.size else 0.0


def flip_level(levels: FloatArray, gex: FloatArray, spot: float) -> float:
    """Price where GEX changes sign nearest to ``spot``; ``nan`` if it never does."""
    if levels.shape != gex.shape or levels.size < 2:
        raise InvalidInputError("levels and gex must be equal-length arrays of at least 2 points")
    crossings = np.flatnonzero(np.sign(gex[:-1]) != np.sign(gex[1:]))
    if crossings.size == 0:
        return float("nan")
    i = int(crossings[np.argmin(np.abs(levels[crossings] - spot))])
    return float(levels[i] - gex[i] * (levels[i + 1] - levels[i]) / (gex[i + 1] - gex[i]))


def default_grid(
    low: float = DEFAULT_GRID_LOW,
    high: float = DEFAULT_GRID_HIGH,
    points: int = DEFAULT_GRID_POINTS,
) -> FloatArray:
    """Multipliers of spot at which GEX is evaluated (odd length so that 1.0 is included)."""
    if points < 3 or points % 2 == 0:
        raise InvalidInputError("points must be an odd number of at least 3")
    if not 0 < low < 1 < high:
        raise InvalidInputError("need 0 < low < 1 < high")
    return np.linspace(low, high, points)


def compute_gex(
    chain: ChainSnapshot,
    implied_vol: FloatArray,
    backend: PricingBackend,
    *,
    spec: QuoteSpec,
    grid: FloatArray | None = None,
    steps: int = 100,
) -> GexProfile:
    """GEX profile of ``chain`` using the volatilities in ``implied_vol`` (aligned with its rows).

    Rows rejected by ``spec`` (for example no open interest) or without a finite volatility do not
    contribute. The chain must carry open interest for the result to mean anything.
    """
    if implied_vol.shape != (chain.size,):
        raise InvalidInputError("implied_vol must have one value per chain row")
    multipliers = default_grid() if grid is None else grid
    keep = spec.mask(chain) & np.isfinite(implied_vol)
    levels = chain.spot * multipliers
    if not keep.any():
        return GexProfile(chain.underlying, chain.spot, levels, np.zeros_like(levels), 0)

    t = years_to_expiry(chain.asof, chain.expiry[keep])
    gamma = backend.american_gamma_grid(
        chain.spot,
        chain.strike[keep],
        t,
        chain.rate,
        chain.dividend_yield,
        implied_vol[keep],
        chain.is_call[keep],
        multipliers,
        steps=steps,
    )
    weight = chain.right_sign[keep] * chain.open_interest[keep] * chain.multiplier * 0.01
    weight = weight / DOLLARS_PER_BILLION
    gamma = np.nan_to_num(gamma, nan=0.0)
    profile = (weight[:, None] * gamma * (levels[None, :] ** 2)).sum(axis=0)
    return GexProfile(chain.underlying, chain.spot, levels, profile, int(keep.sum()))


def atm_implied_vol(
    chain: ChainSnapshot, implied_vol: FloatArray, *, min_days: float = 1.0
) -> float:
    """Implied vol of the strike nearest spot in the nearest expiry at least ``min_days`` away.

    Returned in volatility points (20.0 means 20 %); ``nan`` if no such option has a volatility.
    """
    t = years_to_expiry(chain.asof, chain.expiry)
    eligible = np.isfinite(implied_vol) & (t >= min_days / 365.0)
    if not eligible.any():
        return float("nan")
    nearest = eligible & (t == t[eligible].min())
    candidates = np.flatnonzero(nearest)
    best = candidates[int(np.argmin(np.abs(chain.strike[candidates] - chain.spot)))]
    return float(implied_vol[best] * 100.0)
