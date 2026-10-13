"""Builder for :class:`~volsurface.surface.surface.VolSurface`.

A surface needs a chain, an implied volatility for each row and a handful of fitting choices.
The builder collects them step by step, validates the combination once in :meth:`build`, and keeps
the long procedure (maturity bucketing, OTM selection, weighting, per-expiry fitting with
calendar awareness) out of callers::

    surface = (
        SurfaceBuilder()
        .with_chain(chain)
        .with_implied_vols(iv)
        .with_warm_start(previous_surface)
        .build()
    )
"""

from datetime import date
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike

from volsurface.domain.calendar import years_to_expiry
from volsurface.domain.errors import ConfigurationError, InvalidInputError, PricingError
from volsurface.domain.types import ChainSnapshot, FloatArray
from volsurface.pricing.black_scholes import bs_greeks
from volsurface.surface.surface import SkippedExpiry, SurfaceSlice, VolSurface
from volsurface.surface.svi import SliceFunction, SviParams, total_variance
from volsurface.surface.svi_fit import fit_svi_slice

Weighting = Literal["vega", "uniform"]
_VEGA_FLOOR = 0.05


def _floor_of(prior: SviParams) -> SliceFunction:
    def floor(grid: FloatArray) -> FloatArray:
        return total_variance(grid, prior)

    return floor


class SurfaceBuilder:
    """Fluent builder; every ``with_*`` method returns the builder."""

    def __init__(self) -> None:
        self._chain: ChainSnapshot | None = None
        self._iv: FloatArray | None = None
        self._min_points = 6
        self._weighting: Weighting = "vega"
        self._warm_start: VolSurface | None = None
        self._calendar_aware = True
        self._otm_only = True
        self._enforce_no_arbitrage = True

    def with_chain(self, chain: ChainSnapshot) -> Self:
        self._chain = chain
        return self

    def with_implied_vols(self, iv: ArrayLike) -> Self:
        self._iv = np.asarray(iv, dtype=np.float64)
        return self

    def with_min_points(self, n: int) -> Self:
        if n < 5:
            raise InvalidInputError("an SVI slice needs at least 5 points")
        self._min_points = n
        return self

    def with_weighting(self, weighting: Weighting) -> Self:
        self._weighting = weighting
        return self

    def with_warm_start(self, previous: VolSurface | None) -> Self:
        self._warm_start = previous
        return self

    def with_calendar_awareness(self, enabled: bool) -> Self:
        self._calendar_aware = enabled
        return self

    def with_otm_only(self, enabled: bool) -> Self:
        self._otm_only = enabled
        return self

    def with_arbitrage_penalties(self, enabled: bool) -> Self:
        self._enforce_no_arbitrage = enabled
        return self

    def build(self) -> VolSurface:
        chain, iv = self._require_inputs()
        t_all = years_to_expiry(chain.asof, chain.expiry)
        forward = chain.spot * np.exp((chain.rate - chain.dividend_yield) * t_all)
        log_moneyness = np.log(chain.strike / forward)
        usable = np.isfinite(iv) & (iv > 0) & (t_all > 0)
        if self._otm_only:
            usable &= chain.is_call == (chain.strike >= forward)

        fitted: list[SurfaceSlice] = []
        skipped: list[SkippedExpiry] = []
        previous_params: SviParams | None = None
        previous_t = 0.0
        for expiry in np.unique(chain.expiry):
            rows = usable & (chain.expiry == expiry)
            day = expiry.astype(object)
            if int(rows.sum()) < self._min_points:
                skipped.append(
                    SkippedExpiry(
                        day, f"only {int(rows.sum())} usable quotes (< {self._min_points})"
                    )
                )
                continue
            t = float(t_all[rows][0])
            if t <= previous_t:
                skipped.append(SkippedExpiry(day, "not later than the previous fitted expiry"))
                continue
            k = log_moneyness[rows]
            sigma = iv[rows]
            w = sigma * sigma * t
            lower_slice = (
                _floor_of(previous_params)
                if self._calendar_aware and previous_params is not None
                else None
            )
            warm = self._warm_slice(day)
            try:
                fit = fit_svi_slice(
                    k,
                    w,
                    weights=self._weights(chain, rows, sigma, t),
                    t=t,
                    initial=warm.params if warm else None,
                    initial_rmse=warm.fit.rmse_variance if warm else None,
                    enforce_no_arbitrage=self._enforce_no_arbitrage,
                    lower_slice=lower_slice,
                )
            except (InvalidInputError, PricingError) as exc:
                skipped.append(SkippedExpiry(day, f"fit failed: {exc}"))
                continue
            fitted.append(
                SurfaceSlice(
                    expiry=day,
                    t=t,
                    forward=float(forward[rows][0]),
                    fit=fit,
                    k_min=float(k.min()),
                    k_max=float(k.max()),
                )
            )
            previous_params, previous_t = fit.params, t

        if not fitted:
            raise PricingError("no expiry had enough usable quotes to fit a slice")
        return VolSurface(
            underlying=chain.underlying,
            asof=chain.asof,
            spot=chain.spot,
            rate=chain.rate,
            dividend_yield=chain.dividend_yield,
            slices=tuple(fitted),
            skipped=tuple(skipped),
        )

    def _require_inputs(self) -> tuple[ChainSnapshot, FloatArray]:
        if self._chain is None:
            raise ConfigurationError("SurfaceBuilder needs a chain: call with_chain()")
        if self._iv is None:
            raise ConfigurationError("SurfaceBuilder needs implied vols: call with_implied_vols()")
        if self._iv.shape != (self._chain.size,):
            raise InvalidInputError(
                f"implied vols have shape {self._iv.shape}, chain has {self._chain.size} rows"
            )
        return self._chain, self._iv

    def _warm_slice(self, expiry: date) -> SurfaceSlice | None:
        if self._warm_start is None:
            return None
        for s in self._warm_start.slices:
            if s.expiry == expiry:
                return s
        return None

    def _weights(
        self, chain: ChainSnapshot, rows: np.ndarray, sigma: FloatArray, t: float
    ) -> FloatArray | None:
        if self._weighting == "uniform":
            return None
        vega = bs_greeks(
            chain.spot,
            chain.strike[rows],
            t,
            chain.rate,
            chain.dividend_yield,
            sigma,
            chain.is_call[rows],
        ).vega
        weight = vega / (sigma * t)
        top = float(np.nanmax(weight))
        if not np.isfinite(top) or top <= 0:
            return None
        return np.maximum(weight / top, _VEGA_FLOOR)
