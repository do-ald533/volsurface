from collections.abc import Iterable
from dataclasses import replace

import numpy as np
import pytest

from helpers import T0, small_chain
from volsurface.analytics.alerts import Alert, AlertEngine, AlertKind, MarketSummary
from volsurface.app.config import PipelineSettings
from volsurface.app.pipeline import SurfacePipeline
from volsurface.domain.types import ChainSnapshot, ExerciseStyle
from volsurface.pricing.backends import NumbaCpuBackend

SETTINGS = PipelineSettings(tree_steps=60, gex_tree_steps=40, max_days_to_expiry=120)


def _pipeline(**kwargs: object) -> SurfacePipeline:
    return SurfacePipeline(NumbaCpuBackend(), SETTINGS, **kwargs)  # type: ignore[arg-type]


class AlwaysAlert:
    def evaluate(self, previous: MarketSummary | None, current: MarketSummary) -> Iterable[Alert]:
        yield Alert(AlertKind.IV_JUMP, current.underlying, current.asof, "test", 1.0)


def test_processes_a_chain_into_surface_gex_and_timings() -> None:
    chain = small_chain()
    update = _pipeline().process(chain, seq=7, source="synthetic")
    assert update.seq == 7
    assert update.source == "synthetic"
    assert update.chain is chain
    assert update.usable.shape == (chain.size,)
    assert update.implied_vol.shape == (chain.size,)
    assert np.isfinite(update.implied_vol[update.usable]).mean() > 0.9
    assert np.isnan(update.implied_vol[~update.usable]).all()
    assert update.surface is not None
    assert len(update.surface.slices) == 3
    assert update.gex is not None
    assert update.summary is not None
    assert update.summary.underlying == "SYN"
    assert all(seconds > 0 for _, seconds in update.timings.items())
    assert update.timings.total_s == pytest.approx(sum(s for _, s in update.timings.items()))
    assert update.warnings == ()


def test_second_update_warm_starts_from_the_first() -> None:
    pipeline = _pipeline()
    first = pipeline.process(small_chain(tick=0))
    second = pipeline.process(small_chain(tick=1))
    assert first.surface is not None
    assert second.surface is not None
    cold_effort = sum(s.fit.nfev for s in first.surface.slices)
    warm_effort = sum(s.fit.nfev for s in second.surface.slices)
    assert warm_effort < cold_effort


def test_chain_without_open_interest_skips_gex_with_a_warning() -> None:
    chain = small_chain()
    no_oi = replace(chain, open_interest=np.full(chain.size, np.nan))
    update = _pipeline().process(no_oi)
    assert update.surface is not None
    assert update.gex is None
    assert update.summary is None
    assert any("open interest" in w for w in update.warnings)


def test_gex_can_be_disabled_by_settings() -> None:
    settings = SETTINGS.model_copy(update={"compute_gex": False})
    update = SurfacePipeline(NumbaCpuBackend(), settings).process(small_chain())
    assert update.gex is None
    assert update.warnings == ()


def test_chain_with_no_usable_quotes_gives_no_surface() -> None:
    chain = small_chain()
    dead = replace(chain, bid=np.zeros(chain.size), ask=np.zeros(chain.size))
    update = _pipeline().process(dead)
    assert update.surface is None
    assert not update.usable.any()
    assert any("no usable quotes" in w for w in update.warnings)


def test_european_chains_use_black_scholes() -> None:
    chain = small_chain()
    european = replace(chain, exercise=ExerciseStyle.EUROPEAN)
    update = _pipeline().process(european)
    assert update.surface is not None
    assert np.isfinite(update.implied_vol).any()


def test_alerts_flow_through_the_engine() -> None:
    engine = AlertEngine([AlwaysAlert()], min_gex=0.0)
    pipeline = _pipeline(alerts=engine)
    update = pipeline.process(small_chain())
    assert [a.kind for a in update.alerts] == [AlertKind.IV_JUMP]


def test_changing_underlying_resets_the_stored_state() -> None:
    pipeline = _pipeline()
    pipeline.process(small_chain())
    other = replace(small_chain(tick=1), underlying="OTHER")
    update = pipeline.process(other)
    assert update.surface is not None
    cold = sum(s.fit.nfev for s in update.surface.slices)
    assert cold > len(update.surface.slices) * 20


def test_injected_timer_makes_timings_deterministic() -> None:
    ticks = iter(range(100))
    pipeline = _pipeline(timer=lambda: float(next(ticks)))
    update = pipeline.process(small_chain())
    assert update.timings.items() == (
        ("filter", 1.0),
        ("implied_vol", 1.0),
        ("surface", 1.0),
        ("gex", 1.0),
    )


def test_processing_does_not_mutate_the_input_chain() -> None:
    chain: ChainSnapshot = small_chain()
    before = chain.bid.copy()
    _pipeline().process(chain)
    np.testing.assert_array_equal(chain.bid, before)
    assert chain.asof == T0
