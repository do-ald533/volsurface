"""The processing pipeline: one chain snapshot in, one ``SessionUpdate`` out.

Stages, each timed separately:

1. **filter**: drop quotes that carry no volatility information (``pricing.quote_filter``);
2. **implied_vol**: invert the mid price on the CRR tree (American) or Black-Scholes (European);
3. **surface**: fit an SVI slice per expiry, warm-started from the previous surface;
4. **gex**: gamma-exposure profile and the alert rules, when open interest is available.

``process`` is synchronous and CPU-bound, so callers run it on a worker thread. The object keeps
the previous surface and market summary between calls, which makes it stateful but lets every
update after the first reuse the last fit.
"""

import time
from collections.abc import Callable

import numpy as np

from volsurface.analytics.alerts import Alert, AlertEngine, MarketSummary
from volsurface.analytics.gex import GexProfile, compute_gex, default_grid
from volsurface.app.config import PipelineSettings
from volsurface.app.results import SessionUpdate, StageTimings
from volsurface.domain.calendar import years_to_expiry
from volsurface.domain.errors import InvalidInputError, PricingError
from volsurface.domain.types import ChainSnapshot, ExerciseStyle, FloatArray
from volsurface.pricing.backends import PricingBackend
from volsurface.pricing.implied_vol import implied_vol_bs
from volsurface.pricing.quote_filter import gex_quote_spec, surface_quote_spec
from volsurface.surface.builder import SurfaceBuilder
from volsurface.surface.surface import VolSurface

type Timer = Callable[[], float]


class SurfacePipeline:
    def __init__(
        self,
        backend: PricingBackend,
        settings: PipelineSettings | None = None,
        *,
        alerts: AlertEngine | None = None,
        timer: Timer = time.perf_counter,
    ) -> None:
        self._backend = backend
        self._settings = settings or PipelineSettings()
        cfg = self._settings
        self._alerts = alerts or AlertEngine.default(
            near_flip_pct=cfg.near_flip_pct, iv_jump_points=cfg.iv_jump_points, min_gex=cfg.min_gex
        )
        self._surface_spec = surface_quote_spec(cfg.max_days_to_expiry, cfg.min_time_value)
        self._gex_spec = gex_quote_spec(cfg.max_days_to_expiry, cfg.min_time_value)
        self._grid = default_grid(cfg.flip_grid_low, cfg.flip_grid_high, cfg.flip_grid_points)
        self._timer = timer
        self._underlying: str | None = None
        self._previous_surface: VolSurface | None = None
        self._previous_summary: MarketSummary | None = None

    def reset(self) -> None:
        self._underlying = None
        self._previous_surface = None
        self._previous_summary = None

    def process(self, chain: ChainSnapshot, *, seq: int = 0, source: str = "") -> SessionUpdate:
        if chain.underlying != self._underlying:
            self.reset()
            self._underlying = chain.underlying
        warnings: list[str] = []

        t0 = self._timer()
        usable = self._surface_spec.mask(chain)
        t1 = self._timer()
        implied_vol = self._implied_vols(chain, usable)
        t2 = self._timer()
        surface = self._surface(chain, implied_vol, warnings)
        t3 = self._timer()
        gex, summary, alerts = self._gex(chain, implied_vol, warnings)
        t4 = self._timer()

        if surface is not None:
            self._previous_surface = surface
        self._previous_summary = summary or self._previous_summary
        return SessionUpdate(
            seq=seq,
            source=source,
            chain=chain,
            usable=usable,
            implied_vol=implied_vol,
            surface=surface,
            gex=gex,
            summary=summary,
            alerts=alerts,
            timings=StageTimings(t1 - t0, t2 - t1, t3 - t2, t4 - t3),
            warnings=tuple(warnings),
        )

    def _implied_vols(self, chain: ChainSnapshot, usable: np.ndarray) -> FloatArray:
        out = np.full(chain.size, np.nan)
        if not usable.any():
            return out
        t = years_to_expiry(chain.asof, chain.expiry[usable])
        args = (
            chain.mid[usable],
            chain.spot,
            chain.strike[usable],
            t,
            chain.rate,
            chain.dividend_yield,
            chain.is_call[usable],
        )
        if chain.exercise is ExerciseStyle.AMERICAN:
            out[usable] = self._backend.american_implied_vol(*args, steps=self._settings.tree_steps)
        else:
            out[usable] = implied_vol_bs(*args)
        return out

    def _surface(
        self, chain: ChainSnapshot, implied_vol: FloatArray, warnings: list[str]
    ) -> VolSurface | None:
        if not np.isfinite(implied_vol).any():
            warnings.append("no usable quotes after filtering")
            return None
        try:
            return (
                SurfaceBuilder()
                .with_chain(chain)
                .with_implied_vols(implied_vol)
                .with_min_points(self._settings.min_points_per_expiry)
                .with_warm_start(self._previous_surface)
                .build()
            )
        except (PricingError, InvalidInputError) as exc:
            warnings.append(f"surface not fitted: {exc}")
            return None

    def _gex(
        self, chain: ChainSnapshot, implied_vol: FloatArray, warnings: list[str]
    ) -> tuple[GexProfile | None, MarketSummary | None, tuple[Alert, ...]]:
        if not self._settings.compute_gex:
            return None, None, ()
        if not np.isfinite(chain.open_interest).any():
            warnings.append("source provides no open interest: GEX skipped")
            return None, None, ()
        profile = compute_gex(
            chain,
            implied_vol,
            self._backend,
            spec=self._gex_spec,
            grid=self._grid,
            steps=self._settings.gex_tree_steps,
        )
        summary = MarketSummary.from_profile(chain, profile, implied_vol)
        alerts = self._alerts.evaluate(self._previous_summary, summary)
        return profile, summary, alerts
