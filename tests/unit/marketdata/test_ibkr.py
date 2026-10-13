import math
from datetime import date, timedelta

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from helpers import T0, FakeGateway, fake_options, listed_chain, wait_until
from volsurface.domain.clock import ManualClock
from volsurface.domain.errors import ConfigurationError
from volsurface.domain.events import ChainReceived, ConnectionState, ConnectionStateChanged
from volsurface.domain.types import Right
from volsurface.eventing.testing import RecordingPublisher
from volsurface.marketdata.ibkr import (
    BrokerQuote,
    BrokerSnapshot,
    IbkrProvider,
    OptionChainInfo,
    OptionSpec,
    build_snapshot,
    select_contracts,
)
from volsurface.marketdata.options import IbkrOptions

TODAY = T0.date()
INFO = listed_chain()


def _select(**overrides: object) -> list[OptionSpec]:
    kwargs: dict[str, object] = {
        "today": TODAY,
        "n_expiries": 3,
        "band": 0.1,
        "budget": 30,
    }
    kwargs.update(overrides)
    return select_contracts(INFO.expirations, INFO.strikes, 100.0, **kwargs)  # type: ignore[arg-type]


class TestSelectContracts:
    def test_respects_the_budget_and_the_expiry_count(self) -> None:
        specs = _select()
        assert 0 < len(specs) <= 30
        assert len({s.expiry for s in specs}) == 3
        assert {s.expiry for s in specs} == set(sorted(INFO.expirations)[:3])

    def test_only_the_out_of_the_money_side_is_requested(self) -> None:
        for spec in _select():
            assert (spec.right is Right.CALL) == (spec.strike >= 100.0)

    def test_strikes_stay_inside_the_band_and_prefer_spot(self) -> None:
        specs = _select(budget=21, n_expiries=1)
        strikes = sorted(s.strike for s in specs)
        assert all(90.0 <= k <= 110.0 for k in strikes)
        assert 100.0 in strikes
        assert max(strikes) - 100.0 <= 11
        assert 100.0 - min(strikes) <= 11

    def test_tight_budgets_still_cover_every_expiry(self) -> None:
        specs = _select(budget=3, n_expiries=3)
        assert len(specs) == 3
        assert len({s.expiry for s in specs}) == 3

    def test_past_expiries_and_empty_chains(self) -> None:
        past = [TODAY - timedelta(days=1)]
        assert (
            select_contracts(past, [100.0], 100.0, today=TODAY, n_expiries=2, band=0.1, budget=5)
            == []
        )
        assert (
            select_contracts(
                INFO.expirations, [], 100.0, today=TODAY, n_expiries=2, band=0.1, budget=5
            )
            == []
        )

    def test_zero_budget_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError):
            _select(budget=0)

    @given(
        budget=st.integers(min_value=1, max_value=120),
        n_expiries=st.integers(min_value=1, max_value=7),
        spot=st.floats(min_value=75.0, max_value=125.0),
        band=st.floats(min_value=0.01, max_value=0.4),
    )
    def test_never_exceeds_the_budget_for_any_inputs(
        self, budget: int, n_expiries: int, spot: float, band: float
    ) -> None:
        specs = select_contracts(
            INFO.expirations,
            INFO.strikes,
            spot,
            today=TODAY,
            n_expiries=n_expiries,
            band=band,
            budget=budget,
        )
        assert len(specs) <= budget
        assert len(set(specs)) == len(specs)
        assert all(spot * (1 - band) <= s.strike <= spot * (1 + band) for s in specs)


class TestBuildSnapshot:
    def _info(self) -> OptionChainInfo:
        return OptionChainInfo((TODAY,), (100.0,), multiplier=100)

    def _quote(
        self, strike: float, bid: float, ask: float, oi: float = 10.0, iv: float = 0.2
    ) -> BrokerQuote:
        right = Right.CALL if strike >= 100 else Right.PUT
        return BrokerQuote(OptionSpec(date(2026, 11, 20), strike, right), bid, ask, oi, iv)

    def test_translates_and_sanitises_broker_values(self) -> None:
        broker = BrokerSnapshot(
            100.0,
            (
                self._quote(105.0, 1.0, 1.2, oi=math.nan, iv=math.nan),
                self._quote(95.0, -1.0, 2.0),
                self._quote(100.0, 3.0, 3.2),
            ),
        )
        chain = build_snapshot(
            "SPY", broker, self._info(), IbkrOptions(rate=0.05, dividend_yield=0.01), T0
        )
        assert chain is not None
        assert chain.underlying == "SPY"
        assert chain.rate == 0.05
        assert chain.dividend_yield == 0.01
        assert chain.strike.tolist() == [95.0, 100.0, 105.0]
        assert math.isnan(chain.bid[0])
        assert math.isnan(chain.open_interest[2])
        assert math.isnan(chain.vendor_iv[2])
        assert chain.vendor_iv[1] == pytest.approx(0.2)
        assert chain.is_call.tolist() == [False, True, True]

    @pytest.mark.parametrize(
        "broker",
        [
            BrokerSnapshot(math.nan, ()),
            BrokerSnapshot(100.0, ()),
            BrokerSnapshot(
                -1.0,
                (BrokerQuote(OptionSpec(date(2026, 11, 20), 100.0, Right.CALL), 1, 2, 1, 0.2),),
            ),
            BrokerSnapshot(
                100.0,
                (BrokerQuote(OptionSpec(date(2026, 11, 20), 100.0, Right.CALL), -1, -1, 1, 0.2),),
            ),
        ],
    )
    def test_returns_none_when_nothing_is_usable(self, broker: BrokerSnapshot) -> None:
        assert build_snapshot("SPY", broker, self._info(), IbkrOptions(), T0) is None


def _provider(gateway: FakeGateway, **kwargs: object) -> tuple[IbkrProvider, ManualClock]:
    clock = ManualClock(T0)
    options = kwargs.pop("options", fake_options())
    provider = IbkrProvider(
        "SPY",
        options,  # type: ignore[arg-type]
        gateway_factory=lambda symbol, opts: gateway,
        clock=clock,
        poll_interval_s=0.0,
        warmup_s=0.0,
        backoff_s=0.0,
        **kwargs,  # type: ignore[arg-type]
    )
    return provider, clock


class TestIbkrProvider:
    async def test_publishes_connected_then_chains(self) -> None:
        gateway = FakeGateway()
        provider, _ = _provider(gateway)
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(lambda: len(out.of_type(ChainReceived)) >= 3)
        await provider.stop()
        states = [s.state for s in out.of_type(ConnectionStateChanged)]
        assert states[:2] == [ConnectionState.CONNECTING, ConnectionState.CONNECTED]
        chain = out.of_type(ChainReceived)[0].snapshot
        assert chain.underlying == "SPY"
        assert chain.asof == T0
        assert chain.size == len(gateway.subscriptions[0])
        assert gateway.disconnects >= 1

    async def test_subscriptions_respect_the_line_budget(self) -> None:
        gateway = FakeGateway()
        provider, _ = _provider(gateway, options=fake_options(max_lines=21, expiries=2))
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(lambda: len(gateway.subscriptions) >= 1)
        await provider.stop()
        assert len(gateway.subscriptions[0]) <= 20

    async def test_strikes_are_reselected_when_spot_drifts(self) -> None:
        gateway = FakeGateway(spot_after_reads={3: 108.0})
        provider, _ = _provider(gateway, options=fake_options(band=0.1))
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(lambda: len(gateway.subscriptions) >= 2)
        await provider.stop()
        centre_before = np.mean([s.strike for s in gateway.subscriptions[0]])
        centre_after = np.mean([s.strike for s in gateway.subscriptions[1]])
        assert centre_after > centre_before

    async def test_reconnects_with_backoff_after_connection_failures(self) -> None:
        gateway = FakeGateway(connect_failures=2)
        provider, _ = _provider(gateway)
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(lambda: len(out.of_type(ChainReceived)) >= 1)
        await provider.stop()
        states = [s.state for s in out.of_type(ConnectionStateChanged)]
        assert states.count(ConnectionState.RECONNECTING) >= 2
        assert ConnectionState.CONNECTED in states
        assert ConnectionState.FAILED not in states

    async def test_gives_up_after_too_many_failures(self) -> None:
        gateway = FakeGateway(connect_failures=100)
        provider, _ = _provider(gateway, max_reconnects=2)
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(
            lambda: any(
                s.state is ConnectionState.FAILED for s in out.of_type(ConnectionStateChanged)
            )
        )
        await provider.stop()
        failed = [
            s for s in out.of_type(ConnectionStateChanged) if s.state is ConnectionState.FAILED
        ]
        assert "gave up" in failed[0].detail
        assert out.of_type(ChainReceived) == []

    async def test_a_dropped_connection_triggers_a_reconnect(self) -> None:
        gateway = FakeGateway(drop_after_reads=3)
        provider, _ = _provider(gateway)
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(
            lambda: any(
                s.state is ConnectionState.RECONNECTING for s in out.of_type(ConnectionStateChanged)
            )
        )
        await provider.stop()
        assert len(out.of_type(ChainReceived)) >= 2

    async def test_stop_is_idempotent_and_disconnects(self) -> None:
        gateway = FakeGateway()
        provider, _ = _provider(gateway)
        await provider.start(RecordingPublisher())
        await provider.stop()
        await provider.stop()
        assert not gateway.connected
