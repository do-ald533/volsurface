import asyncio
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.events import ChainReceived, ConnectionState, ConnectionStateChanged
from volsurface.domain.ports import MarketDataProvider
from volsurface.eventing import AsyncEventBus
from volsurface.eventing.testing import RecordingPublisher
from volsurface.marketdata.synthetic import (
    SyntheticChain,
    SyntheticConfig,
    SyntheticProvider,
    default_ssvi,
)

ASOF = datetime(2026, 10, 2, 14, 0, tzinfo=UTC)


class TestSyntheticChain:
    def test_is_deterministic_for_a_seed_and_tick(self) -> None:
        a = SyntheticChain(SyntheticConfig(seed=3)).snapshot(ASOF, 100.0, 1.0, tick=4)
        b = SyntheticChain(SyntheticConfig(seed=3)).snapshot(ASOF, 100.0, 1.0, tick=4)
        np.testing.assert_array_equal(a.bid, b.bid)
        np.testing.assert_array_equal(a.open_interest, b.open_interest)

    def test_different_ticks_differ(self) -> None:
        gen = SyntheticChain(SyntheticConfig(seed=3))
        a = gen.snapshot(ASOF, 100.0, 1.0, tick=1)
        b = gen.snapshot(ASOF, 100.0, 1.0, tick=2)
        assert not np.array_equal(a.bid, b.bid)

    def test_shape_and_quote_sanity(self) -> None:
        cfg = SyntheticConfig()
        chain = SyntheticChain(cfg).snapshot(ASOF, 100.0, 1.0)
        n_strikes = len(np.unique(chain.strike))
        assert chain.size == len(cfg.expiry_days) * n_strikes * 2
        assert (chain.ask >= chain.bid).all()
        assert (chain.bid >= 0).all()
        assert np.isfinite(chain.open_interest).all()
        assert np.isnan(chain.vendor_iv).all()
        assert chain.is_two_sided.mean() > 0.5

    def test_open_interest_has_calls_above_and_puts_below_spot(self) -> None:
        chain = SyntheticChain(SyntheticConfig()).snapshot(ASOF, 100.0, 1.0)
        calls_above = chain.open_interest[chain.is_call & (chain.strike > 100)].sum()
        calls_below = chain.open_interest[chain.is_call & (chain.strike < 100)].sum()
        puts_above = chain.open_interest[~chain.is_call & (chain.strike > 100)].sum()
        puts_below = chain.open_interest[~chain.is_call & (chain.strike < 100)].sum()
        assert calls_above > calls_below
        assert puts_below > puts_above

    def test_higher_vol_scale_means_richer_options(self) -> None:
        gen = SyntheticChain(SyntheticConfig(iv_noise=0.0))
        low = gen.snapshot(ASOF, 100.0, 0.8)
        high = gen.snapshot(ASOF, 100.0, 1.2)
        assert (
            high.mid[high.is_two_sided & low.is_two_sided]
            >= low.mid[high.is_two_sided & low.is_two_sided] - 0.011
        ).all()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"start": datetime(2026, 10, 2)},
            {"spot0": -1.0},
            {"moneyness_range": (1.1, 1.2)},
            {"expiry_days": ()},
            {"expiry_days": (0, 30)},
            {"max_ticks": 0},
            {"interval_s": -1.0},
        ],
    )
    def test_config_validation(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(InvalidInputError):
            SyntheticConfig(**kwargs)  # type: ignore[arg-type]

    def test_default_truth_is_arbitrage_free(self) -> None:
        assert default_ssvi().conditions().arbitrage_free


class TestSyntheticProvider:
    async def test_streams_the_requested_number_of_chains_then_disconnects(self) -> None:
        provider = SyntheticProvider(
            SyntheticConfig(max_ticks=3, interval_s=0.0, expiry_days=(14, 30), seed=1)
        )
        assert isinstance(provider, MarketDataProvider)
        recorder = RecordingPublisher()
        await provider.start(recorder)
        await provider.wait_finished()
        await provider.stop()

        chains = recorder.of_type(ChainReceived)
        states = recorder.of_type(ConnectionStateChanged)
        assert len(chains) == 3
        asofs = [c.snapshot.asof for c in chains]
        assert asofs == [ASOF + timedelta(seconds=30 * i) for i in range(3)]
        assert [s.state for s in states] == [
            ConnectionState.CONNECTED,
            ConnectionState.DISCONNECTED,
        ]
        assert all(c.source == "synthetic" for c in chains)

    async def test_scripted_spot_path_is_followed(self) -> None:
        path = (100.0, 101.5, 99.0, 98.0)
        provider = SyntheticProvider(
            SyntheticConfig(spot_path=path, max_ticks=4, interval_s=0.0, expiry_days=(30,))
        )
        recorder = RecordingPublisher()
        await provider.start(recorder)
        await provider.wait_finished()
        await provider.stop()
        assert [c.snapshot.spot for c in recorder.of_type(ChainReceived)] == list(path)

    async def test_runs_are_reproducible(self) -> None:
        async def run() -> list[float]:
            provider = SyntheticProvider(
                SyntheticConfig(seed=9, max_ticks=4, interval_s=0.0, expiry_days=(30,))
            )
            recorder = RecordingPublisher()
            await provider.start(recorder)
            await provider.wait_finished()
            await provider.stop()
            return [c.snapshot.spot for c in recorder.of_type(ChainReceived)]

        assert await run() == await run()

    async def test_works_through_the_real_event_bus_and_stops_cleanly(self) -> None:
        seen: list[ChainReceived] = []
        provider = SyntheticProvider(SyntheticConfig(interval_s=0.01, expiry_days=(30,)))
        async with AsyncEventBus() as bus:
            bus.subscribe(ChainReceived, seen.append)
            await provider.start(bus)
            while len(seen) < 3:
                await bus.drain()
                await asyncio.sleep(0.01)
            await provider.stop()
            await provider.stop()
        assert len(seen) >= 3

    async def test_start_twice_is_idempotent(self) -> None:
        provider = SyntheticProvider(
            SyntheticConfig(max_ticks=1, interval_s=0.0, expiry_days=(30,))
        )
        recorder = RecordingPublisher()
        await provider.start(recorder)
        await provider.start(recorder)
        await provider.wait_finished()
        await provider.stop()
        assert len(recorder.of_type(ChainReceived)) == 1
