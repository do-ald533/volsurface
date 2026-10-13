import asyncio
import time
from dataclasses import replace

import pytest

from helpers import wait_until
from volsurface.analytics.alerts import AlertEngine, IvJump
from volsurface.app.config import PipelineSettings
from volsurface.app.events import (
    AlertRaised,
    ComputeFailed,
    SessionStateChanged,
    StageTimed,
    SurfaceComputed,
)
from volsurface.app.pipeline import SurfacePipeline
from volsurface.app.session import MarketSession
from volsurface.app.session_state import SessionState
from volsurface.domain.errors import IllegalTransitionError, ProviderError
from volsurface.eventing import AsyncEventBus
from volsurface.marketdata.factory import ProviderFactory, ProviderOptions, SourceSpec
from volsurface.marketdata.synthetic import SyntheticConfig, SyntheticProvider
from volsurface.pricing.backends import NumbaCpuBackend

SETTINGS = PipelineSettings(tree_steps=50, gex_tree_steps=30)
SYN_SPEC = SourceSpec.parse("synthetic")


def _factory(config: SyntheticConfig) -> ProviderFactory:
    factory = ProviderFactory()
    factory.register("synthetic", lambda spec, options: SyntheticProvider(config))
    return factory


def _config(**overrides: object) -> SyntheticConfig:
    base = SyntheticConfig(expiry_days=(14, 30, 60), strike_step=2.0, interval_s=0.01, seed=2)
    return replace(base, **overrides)


def _session(bus: AsyncEventBus, config: SyntheticConfig, pipeline: SurfacePipeline | None = None):  # type: ignore[no-untyped-def]
    pipe = pipeline or SurfacePipeline(NumbaCpuBackend(), SETTINGS)
    return MarketSession(bus, pipe, _factory(config), ProviderOptions())


async def test_streaming_publishes_updates_timings_and_state_changes() -> None:
    updates: list[SurfaceComputed] = []
    stages: list[StageTimed] = []
    states: list[SessionState] = []
    async with AsyncEventBus() as bus:
        bus.subscribe(SurfaceComputed, updates.append)
        bus.subscribe(StageTimed, stages.append)
        bus.subscribe(SessionStateChanged, lambda e: states.append(e.state))
        session = _session(bus, _config())
        await session.start(SYN_SPEC)
        await wait_until(lambda: len(updates) >= 2)
        assert session.state is SessionState.STREAMING
        await session.stop()
        await bus.drain()
    assert states[:2] == [SessionState.CONNECTING, SessionState.STREAMING]
    assert states[-1] is SessionState.STOPPED
    assert updates[0].update.seq == 1
    assert updates[0].update.source == "synthetic"
    assert {s.stage for s in stages} >= {"filter", "implied_vol", "surface", "gex"}
    assert all(s.seconds >= 0 and s.count > 0 for s in stages)


async def test_freeze_pauses_processing_and_resume_continues() -> None:
    updates: list[SurfaceComputed] = []
    async with AsyncEventBus() as bus:
        bus.subscribe(SurfaceComputed, updates.append)
        session = _session(bus, _config())
        await session.start(SYN_SPEC)
        await wait_until(lambda: len(updates) >= 1)
        await session.freeze()
        await bus.drain()
        frozen_at = len(updates)
        await asyncio.sleep(0.15)
        assert len(updates) == frozen_at
        assert session.state is SessionState.FROZEN
        await session.resume()
        await wait_until(lambda: len(updates) > frozen_at)
        await session.stop()


async def test_a_finite_source_ends_the_session() -> None:
    updates: list[SurfaceComputed] = []
    async with AsyncEventBus() as bus:
        bus.subscribe(SurfaceComputed, updates.append)
        session = _session(bus, _config(max_ticks=2, interval_s=0.0))
        await session.start(SYN_SPEC)
        await wait_until(lambda: session.state is SessionState.STOPPED)
    assert len(updates) >= 1


async def test_starting_twice_is_rejected() -> None:
    async with AsyncEventBus() as bus:
        session = _session(bus, _config())
        await session.start(SYN_SPEC)
        with pytest.raises(IllegalTransitionError):
            await session.start(SYN_SPEC)
        await session.stop()


async def test_a_session_can_be_restarted_after_stopping() -> None:
    updates: list[SurfaceComputed] = []
    async with AsyncEventBus() as bus:
        bus.subscribe(SurfaceComputed, updates.append)
        session = _session(bus, _config())
        await session.start(SYN_SPEC)
        await wait_until(lambda: len(updates) >= 1)
        await session.stop()
        await bus.drain()
        count = len(updates)
        await session.start(SYN_SPEC)
        await wait_until(lambda: len(updates) > count)
        assert updates[-1].update.seq == 1
        await session.stop()


async def test_provider_start_failure_marks_the_session_failed() -> None:
    class Broken:
        name = "broken"

        async def start(self, publisher: object) -> None:
            raise RuntimeError("boom")

        async def stop(self) -> None:
            return None

    factory = ProviderFactory({"synthetic": lambda spec, options: Broken()})  # type: ignore[dict-item,return-value]
    async with AsyncEventBus() as bus:
        session = MarketSession(
            bus, SurfacePipeline(NumbaCpuBackend(), SETTINGS), factory, ProviderOptions()
        )
        with pytest.raises(ProviderError, match="boom"):
            await session.start(SYN_SPEC)
        assert session.state is SessionState.FAILED


async def test_a_failing_pipeline_is_reported_and_the_session_survives() -> None:
    class ExplodingOnce(SurfacePipeline):
        calls = 0

        def process(self, chain, *, seq=0, source=""):  # type: ignore[no-untyped-def]
            type(self).calls += 1
            if type(self).calls == 1:
                raise RuntimeError("bad chain")
            return super().process(chain, seq=seq, source=source)

    failures: list[ComputeFailed] = []
    updates: list[SurfaceComputed] = []
    async with AsyncEventBus() as bus:
        bus.subscribe(ComputeFailed, failures.append)
        bus.subscribe(SurfaceComputed, updates.append)
        pipeline = ExplodingOnce(NumbaCpuBackend(), SETTINGS)
        session = _session(bus, _config(), pipeline)
        await session.start(SYN_SPEC)
        await wait_until(lambda: len(updates) >= 1)
        await session.stop()
    assert len(failures) == 1
    assert "bad chain" in failures[0].error
    assert session.state is SessionState.STOPPED


async def test_alerts_are_republished_as_events() -> None:
    alerts: list[AlertRaised] = []
    async with AsyncEventBus() as bus:
        bus.subscribe(AlertRaised, alerts.append)
        engine = AlertEngine([IvJump(0.0001)], min_gex=0.0)
        pipeline = SurfacePipeline(NumbaCpuBackend(), SETTINGS, alerts=engine)
        session = _session(bus, _config(vol_shock=0.05), pipeline)
        await session.start(SYN_SPEC)
        await wait_until(lambda: len(alerts) >= 1, timeout=10)
        await session.stop()
    assert alerts[0].alert.underlying == "SYN"


async def test_conflation_keeps_only_the_latest_chain_when_the_pipeline_is_slow() -> None:
    seen: list[int] = []

    class Slow(SurfacePipeline):
        def process(self, chain, *, seq=0, source=""):  # type: ignore[no-untyped-def]
            time.sleep(0.15)
            seen.append(seq)
            return super().process(chain, seq=seq, source=source)

    async with AsyncEventBus() as bus:
        pipeline = Slow(NumbaCpuBackend(), SETTINGS)
        session = _session(bus, _config(interval_s=0.0, max_ticks=8), pipeline)
        await session.start(SYN_SPEC)
        await wait_until(lambda: session.state is SessionState.STOPPED, timeout=10)
        await bus.drain()
        dropped = {s.name: s.dropped for s in bus.stats()}["session.pipeline"]
    assert dropped > 0
    assert len(seen) < 8
