from dataclasses import replace

import pytest

from helpers import small_chain, wait_until
from volsurface.app.commands import Freeze, Resume, StartSession, StopSession
from volsurface.app.config import PipelineSettings, Settings
from volsurface.app.container import open_container, provider_options
from volsurface.app.diagnostics import DiagnosticsCollector
from volsurface.app.events import ComputeFailed, SessionStateChanged, StageTimed, SurfaceComputed
from volsurface.app.pipeline import SurfacePipeline
from volsurface.app.session_state import SessionState
from volsurface.domain.errors import ConfigurationError
from volsurface.domain.events import HandlerFailed
from volsurface.eventing import AsyncEventBus
from volsurface.marketdata.factory import ProviderFactory, SourceSpec
from volsurface.marketdata.synthetic import SyntheticConfig, SyntheticProvider
from volsurface.pricing.backends import NumbaCpuBackend


def _settings() -> Settings:
    return Settings(pipeline=PipelineSettings(tree_steps=50, gex_tree_steps=30))


def _factory() -> ProviderFactory:
    config = SyntheticConfig(expiry_days=(14, 30, 60), strike_step=2.0, interval_s=0.02, seed=4)
    factory = ProviderFactory()
    factory.register("synthetic", lambda spec, options: SyntheticProvider(config))
    return factory


async def test_commands_drive_the_whole_application_end_to_end() -> None:
    async with open_container(_settings(), factory=_factory(), warmup=False) as app:
        await app.dispatcher.dispatch(StartSession(SourceSpec.parse("synthetic")))
        await wait_until(lambda: app.diagnostics.snapshot().updates >= 2)
        snap = app.diagnostics.snapshot(app.bus.stats())
        assert snap.state is SessionState.STREAMING
        assert snap.options_per_update > 100
        assert snap.update_rate_hz > 0
        assert {"filter", "implied_vol", "surface", "gex"} <= set(snap.stage_seconds)
        assert "american_implied_vol" in snap.backend_seconds
        assert snap.total_seconds > 0
        assert any(s.name == "session.pipeline" for s in snap.subscriptions)

        await app.dispatcher.dispatch(Freeze())
        assert app.session.state is SessionState.FROZEN
        await app.dispatcher.dispatch(Resume())
        assert app.session.state is SessionState.STREAMING
        await app.dispatcher.dispatch(StopSession())
        await app.bus.drain()
        assert app.diagnostics.snapshot().state is SessionState.STOPPED


async def test_leaving_the_context_stops_a_running_session_cleanly() -> None:
    async with open_container(_settings(), factory=_factory(), warmup=False) as app:
        await app.dispatcher.dispatch(StartSession(SourceSpec.parse("synthetic")))
        await wait_until(lambda: app.diagnostics.snapshot().updates >= 1)
    assert app.session.state is SessionState.STOPPED


async def test_unknown_backend_name_is_a_configuration_error() -> None:
    settings = Settings(backend="quantum")
    with pytest.raises(ConfigurationError, match="unknown pricing backend"):
        async with open_container(settings, warmup=False):
            pass


def test_provider_options_map_the_ibkr_settings() -> None:
    settings = Settings(ibkr={"port": 4001, "max_lines": 60, "rate": 0.05})
    options = provider_options(settings)
    assert options.ibkr.port == 4001
    assert options.ibkr.max_lines == 60
    assert options.ibkr.rate == 0.05


class TestDiagnosticsCollector:
    async def test_smooths_stage_times_and_counts_failures(self) -> None:
        async with AsyncEventBus() as bus:
            ticks = iter([0.0, 0.5, 1.0, 1.5])
            collector = DiagnosticsCollector(bus, clock=lambda: next(ticks))
            for seconds in (1.0, 1.0, 2.0):
                bus.publish(StageTimed(stage="surface", seconds=seconds, count=10))
            bus.publish(StageTimed(stage="backend.american_price_gamma", seconds=0.2))
            bus.publish(ComputeFailed(source="x", error="boom"))
            bus.publish(HandlerFailed(event_type="E", handler="h", error="bad"))
            bus.publish(
                SessionStateChanged(previous=SessionState.IDLE, state=SessionState.STREAMING)
            )
            await bus.drain()
            snap = collector.snapshot()
        assert snap.stage_seconds["surface"] == pytest.approx(
            0.8 * (0.8 * 1.0 + 0.2 * 1.0) + 0.2 * 2.0
        )
        assert snap.backend_seconds == {"american_price_gamma": pytest.approx(0.2)}
        assert snap.compute_failures == 1
        assert snap.handler_failures == 1
        assert snap.state is SessionState.STREAMING
        assert "h: bad" in snap.last_error

    async def test_update_rate_is_measured_from_arrival_times(self) -> None:
        update = SurfacePipeline(
            NumbaCpuBackend(), PipelineSettings(tree_steps=30, gex_tree_steps=20)
        ).process(small_chain())
        async with AsyncEventBus() as bus:
            ticks = iter([10.0, 10.5, 11.0, 11.5, 12.0])
            collector = DiagnosticsCollector(bus, clock=lambda: next(ticks))
            for _ in range(5):
                bus.publish(SurfaceComputed(update=replace(update)))
            await bus.drain()
            snap = collector.snapshot()
        assert snap.updates == 5
        assert snap.update_rate_hz == pytest.approx(2.0)
        assert snap.options_per_update == update.chain.size
