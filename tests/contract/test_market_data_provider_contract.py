"""One contract, every provider: the same assertions run against each MarketDataProvider.

If a new provider satisfies these tests it can replace any other one in the session, which is
the Liskov substitution principle made executable.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from helpers import FakeGateway, fake_options, small_chain, wait_until
from volsurface.domain.clock import ManualClock
from volsurface.domain.events import ChainReceived, ConnectionState, ConnectionStateChanged
from volsurface.domain.ports import MarketDataProvider
from volsurface.domain.types import ChainSnapshot
from volsurface.eventing.testing import RecordingPublisher
from volsurface.marketdata.ibkr import IbkrProvider
from volsurface.marketdata.recording_store import RecordingStore
from volsurface.marketdata.replay import ReplayProvider
from volsurface.marketdata.synthetic import SyntheticConfig, SyntheticProvider

type Maker = Callable[[Path], MarketDataProvider]


def _synthetic(_: Path) -> MarketDataProvider:
    return SyntheticProvider(
        SyntheticConfig(max_ticks=3, interval_s=0.0, expiry_days=(14, 30), strike_step=2.0)
    )


def _replay(tmp: Path) -> MarketDataProvider:
    RecordingStore(tmp / "rec").write_all([small_chain(tick=i) for i in range(3)])
    return ReplayProvider(tmp / "rec", speed=0)


def _ibkr(_: Path) -> MarketDataProvider:
    return IbkrProvider(
        "SPY",
        fake_options(),
        gateway_factory=lambda symbol, options: FakeGateway(),
        clock=ManualClock(small_chain().asof),
        poll_interval_s=0.0,
        warmup_s=0.0,
    )


MAKERS = [
    pytest.param(_synthetic, id="synthetic"),
    pytest.param(_replay, id="replay"),
    pytest.param(_ibkr, id="ibkr-fake-gateway"),
]


def _assert_valid_chain(chain: ChainSnapshot) -> None:
    assert chain.asof.tzinfo is not None
    assert chain.spot > 0
    assert chain.size > 0
    assert chain.underlying
    for column in (chain.bid, chain.ask, chain.open_interest, chain.vendor_iv, chain.is_call):
        assert column.shape == chain.strike.shape
    assert (chain.strike > 0).all()
    assert np.isfinite(chain.rate)
    assert np.isfinite(chain.dividend_yield)
    assert chain.is_two_sided.any()


@pytest.mark.parametrize("make", MAKERS)
class TestProviderContract:
    async def _run(self, make: Maker, tmp_path: Path, minimum: int = 1) -> RecordingPublisher:
        provider = make(tmp_path)
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(lambda: len(out.of_type(ChainReceived)) >= minimum)
        await provider.stop()
        return out

    def test_implements_the_port(self, make: Maker, tmp_path: Path) -> None:
        provider = make(tmp_path)
        assert isinstance(provider, MarketDataProvider)
        assert isinstance(provider.name, str)
        assert provider.name

    async def test_publishes_valid_chains(self, make: Maker, tmp_path: Path) -> None:
        out = await self._run(make, tmp_path, minimum=2)
        for event in out.of_type(ChainReceived):
            assert event.source
            _assert_valid_chain(event.snapshot)

    async def test_announces_a_connection_before_the_first_chain(
        self, make: Maker, tmp_path: Path
    ) -> None:
        out = await self._run(make, tmp_path)
        first_connected = next(
            i
            for i, e in enumerate(out.events)
            if isinstance(e, ConnectionStateChanged) and e.state is ConnectionState.CONNECTED
        )
        first_chain = next(i for i, e in enumerate(out.events) if isinstance(e, ChainReceived))
        assert first_connected < first_chain

    async def test_chain_times_never_go_backwards(self, make: Maker, tmp_path: Path) -> None:
        out = await self._run(make, tmp_path, minimum=2)
        times = [e.snapshot.asof for e in out.of_type(ChainReceived)]
        assert times == sorted(times)

    async def test_stop_is_idempotent_and_safe_without_start(
        self, make: Maker, tmp_path: Path
    ) -> None:
        provider = make(tmp_path)
        await provider.stop()
        out = RecordingPublisher()
        await provider.start(out)
        await provider.stop()
        await provider.stop()

    async def test_start_twice_does_not_duplicate_the_stream(
        self, make: Maker, tmp_path: Path
    ) -> None:
        provider = make(tmp_path)
        out = RecordingPublisher()
        await provider.start(out)
        await provider.start(out)
        await wait_until(lambda: len(out.of_type(ChainReceived)) >= 1)
        await provider.stop()
        connected = [
            e for e in out.of_type(ConnectionStateChanged) if e.state is ConnectionState.CONNECTED
        ]
        assert len(connected) == 1
