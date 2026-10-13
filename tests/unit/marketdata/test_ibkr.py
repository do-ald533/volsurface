import asyncio
import math
from datetime import date, timedelta

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from helpers import T0, FakeGateway, fake_options, listed_chain, wait_until
from volsurface.domain.clock import ManualClock
from volsurface.domain.errors import (
    ConfigurationError,
    ConnectionLostError,
    MarketDataUnavailableError,
)
from volsurface.domain.events import ChainReceived, ConnectionState, ConnectionStateChanged
from volsurface.domain.types import Right
from volsurface.eventing.testing import RecordingPublisher
from volsurface.marketdata.ibkr import (
    MAX_PARALLEL_LISTINGS,
    BrokerQuote,
    BrokerSnapshot,
    IbkrProvider,
    OptionChainInfo,
    OptionSpec,
    build_snapshot,
    explain_ib_error,
    select_contracts,
    spread_expiries,
    spread_strikes,
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


def _spy_like_expirations(today: date) -> list[date]:
    daily = [today + timedelta(days=i) for i in range(10)]
    weekly = [today + timedelta(days=7 * i) for i in range(2, 14)]
    monthly = [today + timedelta(days=30 * i) for i in range(4, 15)]
    return sorted(set(daily + weekly + monthly))


class TestSpreadSelection:
    def test_expiries_are_spread_over_the_maturities_not_clustered_at_the_front(self) -> None:
        expirations = _spy_like_expirations(TODAY)
        chosen = spread_expiries(expirations, today=TODAY, count=6, min_days=3, max_days=120)
        days = [(e - TODAY).days for e in chosen]
        assert len(chosen) == 6 and chosen == sorted(chosen)
        assert days[0] <= 5 and days[-1] >= 100
        ratios = np.diff(np.log(days))
        assert ratios.min() > 0.3

    def test_the_maturity_window_is_respected(self) -> None:
        expirations = _spy_like_expirations(TODAY)
        chosen = spread_expiries(expirations, today=TODAY, count=6, min_days=10, max_days=60)
        assert all(10 <= (e - TODAY).days <= 60 for e in chosen)

    def test_few_eligible_expiries_are_all_used(self) -> None:
        expirations = [TODAY + timedelta(days=d) for d in (5, 40, 300)]
        chosen = spread_expiries(expirations, today=TODAY, count=6, min_days=3, max_days=120)
        assert [(e - TODAY).days for e in chosen] == [5, 40]

    def test_outside_the_window_falls_back_to_the_nearest_dates(self) -> None:
        expirations = [TODAY + timedelta(days=d) for d in (1, 2, 400, 500)]
        chosen = spread_expiries(expirations, today=TODAY, count=2, min_days=3, max_days=120)
        assert [(e - TODAY).days for e in chosen] == [1, 2]

    def test_past_expiries_are_ignored(self) -> None:
        expirations = [TODAY - timedelta(days=3), TODAY + timedelta(days=30)]
        chosen = spread_expiries(expirations, today=TODAY, count=3, min_days=0, max_days=120)
        assert chosen == [TODAY + timedelta(days=30)]

    def test_strikes_span_the_whole_band_even_with_one_dollar_spacing(self) -> None:
        strikes = [float(k) for k in range(500, 1000)]
        chosen = spread_strikes(strikes, 777.4, band=0.08, count=13)
        assert chosen == sorted(chosen) and len(set(chosen)) == len(chosen) == 13
        assert chosen[0] / 777.4 - 1 == pytest.approx(-0.08, abs=0.005)
        assert chosen[-1] / 777.4 - 1 == pytest.approx(0.08, abs=0.005)
        assert min(abs(k - 777.4) for k in chosen) < 6.0
        assert np.diff(chosen).max() < 2.5 * np.diff(chosen).min()

    def test_few_listed_strikes_are_all_used(self) -> None:
        assert spread_strikes([90.0, 100.0, 110.0, 200.0], 100.0, band=0.15, count=9) == [
            90.0,
            100.0,
            110.0,
        ]

    def test_the_real_spy_case_from_a_live_run(self) -> None:
        expirations = _spy_like_expirations(TODAY)
        strikes = [float(k) for k in range(500, 1000)]
        specs = select_contracts(
            expirations,
            strikes,
            777.4,
            today=TODAY,
            n_expiries=6,
            band=0.08,
            budget=79,
            min_days=3,
            max_days=120,
        )
        assert len(specs) <= 79
        days = sorted({(s.expiry - TODAY).days for s in specs})
        assert len(days) == 6 and days[-1] - days[0] > 90
        strike_values = sorted({s.strike for s in specs})
        assert strike_values[-1] - strike_values[0] > 0.12 * 777.4
        assert all((s.strike >= 777.4) == (s.right is Right.CALL) for s in specs)


class TestPerExpiryStrikes:
    def test_each_expiry_only_uses_strikes_it_lists(self) -> None:
        expirations = _spy_like_expirations(TODAY)
        near = TODAY + timedelta(days=7)
        far = TODAY + timedelta(days=120)
        listed = {
            near: tuple(float(k) for k in range(600, 900)),
            far: tuple(float(k) for k in range(600, 900, 5)),
        }
        specs = select_contracts(
            [near, far],
            listed,
            777.4,
            today=TODAY,
            n_expiries=2,
            band=0.08,
            budget=26,
            min_days=3,
            max_days=130,
        )
        assert {s.expiry for s in specs} == {near, far}
        for spec in specs:
            assert spec.strike in listed[spec.expiry]
        assert all(s.strike % 5 == 0 for s in specs if s.expiry == far)
        assert expirations

    def test_an_expiry_missing_from_the_mapping_contributes_nothing(self) -> None:
        near = TODAY + timedelta(days=7)
        far = TODAY + timedelta(days=60)
        specs = select_contracts(
            [near, far],
            {near: (90.0, 100.0, 110.0)},
            100.0,
            today=TODAY,
            n_expiries=2,
            band=0.2,
            budget=10,
        )
        assert {s.expiry for s in specs} == {near}


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
        events = out.of_type(ConnectionStateChanged)
        states = [s.state for s in events]
        assert states[0] is ConnectionState.CONNECTING
        assert states.count(ConnectionState.CONNECTED) == 1
        assert all(
            s is ConnectionState.CONNECTING
            for s in states[: states.index(ConnectionState.CONNECTED)]
        )
        steps = " | ".join(e.detail for e in events if e.state is ConnectionState.CONNECTING)
        for expected in (
            "connecting to TWS",
            "loading the SPY option chain",
            "waiting for the SPY price",
        ):
            assert expected in steps
        assert "listing strikes for" in steps and "listed strikes 1/" in steps
        assert "subscribing to" in steps
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

    async def test_an_unsubscribed_feed_fails_at_once_without_retrying(self) -> None:
        gateway = FakeGateway(spot_error=MarketDataUnavailableError("not subscribed"))
        provider, _ = _provider(gateway, max_reconnects=5)
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(
            lambda: any(
                s.state is ConnectionState.FAILED for s in out.of_type(ConnectionStateChanged)
            )
        )
        await provider.stop()
        states = [s.state for s in out.of_type(ConnectionStateChanged)]
        assert states[-1] is ConnectionState.FAILED
        assert set(states[:-1]) == {ConnectionState.CONNECTING}
        assert out.of_type(ConnectionStateChanged)[-1].detail == "not subscribed"
        assert out.of_type(ChainReceived) == []
        assert gateway.disconnects >= 1

    async def test_provider_asks_the_broker_for_each_expiry_strike_list_once(self) -> None:
        info = listed_chain()
        sparse = {e: tuple(float(k) for k in range(70, 131, 5)) for e in info.expirations}
        gateway = FakeGateway(info=info, strikes_by_expiry=sparse, spot_after_reads={3: 108.0})
        provider, _ = _provider(gateway, options=fake_options(band=0.1))
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(lambda: len(gateway.subscriptions) >= 2)
        await provider.stop()
        assert len(gateway.strike_requests) == len(set(gateway.strike_requests))
        for subscription in gateway.subscriptions:
            assert all(spec.strike % 5 == 0 for spec in subscription)

    async def test_setup_after_a_failure_is_reported_as_reconnecting(self) -> None:
        gateway = FakeGateway(connect_failures=1)
        provider, _ = _provider(gateway)
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(lambda: len(out.of_type(ChainReceived)) >= 1)
        await provider.stop()
        events = out.of_type(ConnectionStateChanged)
        retry = [e for e in events if e.state is ConnectionState.RECONNECTING]
        assert any("connecting to TWS" in e.detail for e in retry)
        assert any("subscribing to" in e.detail for e in retry)

    async def test_strike_lists_are_fetched_concurrently(self) -> None:
        gateway = FakeGateway(strike_delay=0.2)
        provider, _ = _provider(gateway, options=fake_options(expiries=5, max_lines=40))
        out = RecordingPublisher()
        loop = asyncio.get_running_loop()
        start = loop.time()
        await provider.start(out)
        await wait_until(
            lambda: any(
                e.state is ConnectionState.CONNECTED for e in out.of_type(ConnectionStateChanged)
            )
        )
        elapsed = loop.time() - start
        await provider.stop()
        assert gateway.max_parallel_strike_calls == 5
        assert elapsed < 0.2 * 3
        details = [e.detail for e in out.of_type(ConnectionStateChanged)]
        assert any(d.startswith("listing strikes for 5 expiries") for d in details)
        assert any(d.startswith("listed strikes 5/5") for d in details)

    async def test_concurrent_listing_is_capped(self) -> None:
        info = listed_chain()
        gateway = FakeGateway(info=info, strike_delay=0.05)
        provider, _ = _provider(gateway, options=fake_options(expiries=7, max_lines=70))
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(lambda: len(gateway.subscriptions) >= 1)
        await provider.stop()
        assert gateway.max_parallel_strike_calls <= MAX_PARALLEL_LISTINGS

    async def test_a_failing_listing_cancels_the_others_and_triggers_a_retry(self) -> None:
        class Flaky(FakeGateway):
            async def list_strikes(self, expiry: date) -> tuple[float, ...]:
                if not self.strike_requests:
                    self.strike_requests.append(expiry)
                    raise ConnectionLostError("link dropped while listing")
                return await super().list_strikes(expiry)

        gateway = Flaky(strike_delay=0.05)
        provider, _ = _provider(gateway, options=fake_options(expiries=4, max_lines=40))
        out = RecordingPublisher()
        await provider.start(out)
        await wait_until(lambda: len(gateway.subscriptions) >= 1)
        await provider.stop()
        states = [e.state for e in out.of_type(ConnectionStateChanged)]
        assert ConnectionState.RECONNECTING in states
        assert gateway.strike_calls_running == 0

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


class TestBrokerErrors:
    TEXT = "Requested market data is not subscribed. Delayed market data is not enabled."

    @pytest.mark.parametrize("code", [354, 10168])
    def test_not_subscribed_errors_say_how_to_get_delayed_data(self, code: int) -> None:
        message = explain_ib_error(code, self.TEXT, market_data_type=1)
        assert message is not None
        assert f"IBKR error {code}" in message
        assert "VOLSURFACE_IBKR__MARKET_DATA_TYPE=3" in message

    def test_when_delayed_data_is_already_requested_the_advice_is_to_subscribe(self) -> None:
        message = explain_ib_error(10168, self.TEXT, market_data_type=3)
        assert message is not None
        assert "MARKET_DATA_TYPE" not in message
        assert "subscribe" in message

    @pytest.mark.parametrize("code", [2104, 2106, 2158, 2186, 200, 10167])
    def test_other_codes_are_not_treated_as_fatal(self, code: int) -> None:
        assert explain_ib_error(code, "anything", market_data_type=1) is None

    async def test_the_real_gateway_turns_an_underlying_error_into_a_clear_failure(self) -> None:
        ib_async = pytest.importorskip("ib_async")
        from volsurface.marketdata.ibkr import IbAsyncGateway  # noqa: PLC0415

        gateway = IbAsyncGateway("SPY", IbkrOptions(market_data_type=1))
        option = ib_async.Option("SPY", "20271217", 500.0, "C", "SMART")
        gateway._on_error(7, 10168, self.TEXT, option)
        assert gateway._data_error is None
        gateway._on_error(5, 10168, self.TEXT, ib_async.Stock("SPY", "SMART", "USD"))
        with pytest.raises(MarketDataUnavailableError, match="MARKET_DATA_TYPE=3"):
            await gateway.wait_for_spot(1.0)


class _Details:
    def __init__(self, contract: object | None) -> None:
        self.contract = contract


class TestRealGatewayContracts:
    @staticmethod
    def _gateway(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
        ib_async = pytest.importorskip("ib_async")
        from volsurface.marketdata.ibkr import IbAsyncGateway  # noqa: PLC0415

        gateway = IbAsyncGateway("SPY", IbkrOptions())
        gateway._info = OptionChainInfo(
            (date(2026, 10, 16),), (), multiplier=100, trading_class="SPY", exchange="SMART"
        )
        requests: list[object] = []
        subscribed: list[object] = []
        cancelled: list[object] = []

        async def contract_details(template: object) -> list[_Details]:
            requests.append(template)
            rows = [_Details(None)]
            rows.extend(
                _Details(ib_async.Option("SPY", "20261016", strike, right, "SMART"))
                for strike in (770.0, 775.0, 780.0)
                for right in ("C", "P")
            )
            rows.append(_Details(ib_async.Option("SPY", "20261016", 0.0, "C", "SMART")))
            return rows

        def request_market_data(contract, ticks, snapshot, regulatory):  # type: ignore[no-untyped-def]
            subscribed.append((contract.strike, contract.right, ticks))
            return type("Ticker", (), {"contract": contract})()

        monkeypatch.setattr(gateway._ib, "reqContractDetailsAsync", contract_details)
        monkeypatch.setattr(gateway._ib, "reqMktData", request_market_data)
        monkeypatch.setattr(gateway._ib, "cancelMktData", cancelled.append)
        return gateway, requests, subscribed, cancelled

    async def test_strikes_come_from_contract_details_for_the_expiry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gateway, requests, _, _ = self._gateway(monkeypatch)
        strikes = await gateway.list_strikes(date(2026, 10, 16))
        assert strikes == (770.0, 775.0, 780.0)
        assert len(requests) == 1
        template = requests[0]
        assert template.lastTradeDateOrContractMonth == "20261016"  # type: ignore[attr-defined]
        assert template.tradingClass == "SPY"  # type: ignore[attr-defined]
        assert template.strike == 0.0  # type: ignore[attr-defined]

    async def test_subscribe_uses_listed_contracts_and_skips_unlisted_ones(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gateway, _, subscribed, cancelled = self._gateway(monkeypatch)
        expiry = date(2026, 10, 16)
        await gateway.list_strikes(expiry)
        specs = [
            OptionSpec(expiry, 770.0, Right.PUT),
            OptionSpec(expiry, 780.0, Right.CALL),
            OptionSpec(expiry, 829.0, Right.CALL),
        ]
        await gateway.subscribe(specs)
        assert sorted(s[:2] for s in subscribed) == [(770.0, "P"), (780.0, "C")]
        assert all(s[2] == "101" for s in subscribed)
        await gateway.subscribe(specs[:1])
        assert len(cancelled) == 1
        await gateway.subscribe(specs[:1])
        assert len(subscribed) == 2

    async def test_listing_needs_the_chain_to_be_loaded_first(self) -> None:
        pytest.importorskip("ib_async")
        from volsurface.domain.errors import ProviderError  # noqa: PLC0415
        from volsurface.marketdata.ibkr import IbAsyncGateway  # noqa: PLC0415

        with pytest.raises(ProviderError, match="load_chain"):
            await IbAsyncGateway("SPY", IbkrOptions()).list_strikes(date(2026, 10, 16))
