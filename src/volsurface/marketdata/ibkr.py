"""Interactive Brokers market data as an event-driven provider.

Structure (an anti-corruption layer around the broker API):

* :class:`BrokerGateway` is the small interface the provider needs. :class:`IbAsyncGateway` is its
  only implementation that touches ``ib_async``; everything else works on plain value objects and
  is tested against a fake gateway.
* :func:`select_contracts` decides what to subscribe to. IBKR allows 100 concurrent market-data
  lines by default (shared with TWS itself), so the number of option subscriptions is a budget,
  not a free choice. Only the out-of-the-money side of each strike is requested, which is what the
  surface fit uses and halves the line count.
* :class:`IbkrProvider` supervises the session: connect, subscribe, publish a ``ChainReceived``
  every ``poll_interval_s``, re-select strikes when spot drifts, and reconnect with exponential
  backoff, reporting every transition as a ``ConnectionStateChanged`` event.

Open interest comes from generic tick 101 (``callOpenInterest``/``putOpenInterest``); vendor
implied volatility is the TWS option model's value and is kept next to ours for validation.
The gateway was written against the ib_async documentation and has not been run against a live
TWS from this code base's test environment: use the ``ibkr`` source with a paper account to
validate it.
"""

import asyncio
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import numpy as np

from volsurface.domain.clock import SystemClock
from volsurface.domain.errors import ConfigurationError, ConnectionLostError, ProviderError
from volsurface.domain.events import ChainReceived, ConnectionState, ConnectionStateChanged
from volsurface.domain.ports import Clock, EventPublisher
from volsurface.domain.types import ChainSnapshot, ExerciseStyle, Right
from volsurface.marketdata.options import IbkrOptions

GENERIC_TICKS = "101"
SPOT_TIMEOUT_S = 15.0


@dataclass(frozen=True, slots=True, order=True)
class OptionSpec:
    """One option to subscribe to."""

    expiry: date
    strike: float
    right: Right


@dataclass(frozen=True, slots=True)
class OptionChainInfo:
    """Listed expirations and strikes for an underlying, as reported by the broker."""

    expirations: tuple[date, ...]
    strikes: tuple[float, ...]
    multiplier: int = 100
    trading_class: str = ""
    exchange: str = "SMART"


@dataclass(frozen=True, slots=True)
class BrokerQuote:
    spec: OptionSpec
    bid: float
    ask: float
    open_interest: float
    model_iv: float


@dataclass(frozen=True, slots=True)
class BrokerSnapshot:
    spot: float
    quotes: tuple[BrokerQuote, ...]


class BrokerGateway(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    @property
    def is_connected(self) -> bool: ...

    async def load_chain(self) -> OptionChainInfo: ...

    async def wait_for_spot(self, timeout_s: float) -> float: ...

    async def subscribe(self, specs: Sequence[OptionSpec]) -> None:
        """Make ``specs`` the active subscription set, cancelling anything not in it."""
        ...

    def read(self) -> BrokerSnapshot: ...


def select_contracts(
    expirations: Sequence[date],
    strikes: Sequence[float],
    spot: float,
    *,
    today: date,
    n_expiries: int,
    band: float,
    budget: int,
) -> list[OptionSpec]:
    """Choose at most ``budget`` out-of-the-money options near spot.

    The first ``n_expiries`` expirations on or after ``today`` are kept; the budget is split evenly
    between them; within an expiry the strikes closest to spot (inside ``spot * (1 +/- band)``) are
    preferred. Strikes above spot are calls and the rest puts.
    """
    if budget < 1:
        raise ConfigurationError("the market-data line budget leaves no room for options")
    chosen_expiries = sorted(e for e in set(expirations) if e >= today)[:n_expiries]
    if not chosen_expiries:
        return []
    low, high = spot * (1.0 - band), spot * (1.0 + band)
    near = sorted((k for k in set(strikes) if low <= k <= high), key=lambda k: abs(k - spot))
    quota = max(budget // len(chosen_expiries), 1)
    picked = sorted(near[:quota])
    specs = [
        OptionSpec(expiry, strike, Right.CALL if strike >= spot else Right.PUT)
        for expiry in chosen_expiries
        for strike in picked
    ]
    return specs[:budget]


def _clean_price(value: float) -> float:
    return value if math.isfinite(value) and value >= 0.0 else math.nan


def build_snapshot(
    symbol: str,
    broker: BrokerSnapshot,
    info: OptionChainInfo,
    options: IbkrOptions,
    asof: datetime,
) -> ChainSnapshot | None:
    """Translate a broker snapshot into a domain snapshot; ``None`` if nothing is usable yet."""
    if not (math.isfinite(broker.spot) and broker.spot > 0) or not broker.quotes:
        return None
    quotes = sorted(broker.quotes, key=lambda q: q.spec)
    bid = np.array([_clean_price(q.bid) for q in quotes])
    ask = np.array([_clean_price(q.ask) for q in quotes])
    if not np.any((bid > 0) & (ask > 0)):
        return None
    return ChainSnapshot(
        underlying=symbol,
        asof=asof,
        spot=broker.spot,
        rate=options.rate,
        dividend_yield=options.dividend_yield,
        expiry=np.array([q.spec.expiry for q in quotes], dtype="datetime64[D]"),
        strike=np.array([q.spec.strike for q in quotes]),
        is_call=np.array([q.spec.right.is_call for q in quotes]),
        bid=bid,
        ask=ask,
        open_interest=np.array([_clean_price(q.open_interest) for q in quotes]),
        vendor_iv=np.array([_clean_price(q.model_iv) for q in quotes]),
        exercise=ExerciseStyle.AMERICAN,
        multiplier=info.multiplier,
    )


type GatewayFactory = Callable[[str, IbkrOptions], BrokerGateway]


class IbkrProvider:
    name = "ibkr"

    def __init__(
        self,
        symbol: str,
        options: IbkrOptions | None = None,
        *,
        gateway_factory: GatewayFactory | None = None,
        clock: Clock | None = None,
        poll_interval_s: float = 1.0,
        warmup_s: float = 3.0,
        max_reconnects: int = 5,
        backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
    ) -> None:
        self._symbol = symbol
        self._options = options or IbkrOptions()
        self._factory = gateway_factory or IbAsyncGateway
        self._clock: Clock = clock or SystemClock()
        self._poll = poll_interval_s
        self._warmup = warmup_s
        self._max_reconnects = max_reconnects
        self._backoff = backoff_s
        self._max_backoff = max_backoff_s
        self._gateway: BrokerGateway | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self, publisher: EventPublisher) -> None:
        if self._task is not None:
            return
        self._task = asyncio.get_running_loop().create_task(
            self._supervise(publisher), name=f"ibkr-{self._symbol}"
        )

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        gateway, self._gateway = self._gateway, None
        if gateway is not None:
            await gateway.disconnect()

    async def _supervise(self, publisher: EventPublisher) -> None:
        failures = 0
        while True:
            self._emit(
                publisher,
                ConnectionState.CONNECTING if failures == 0 else ConnectionState.RECONNECTING,
            )
            try:
                await self._run_session(publisher)
            except asyncio.CancelledError:
                raise
            except (ConnectionLostError, ProviderError, OSError, TimeoutError) as exc:
                failures += 1
                if failures > self._max_reconnects:
                    self._emit(
                        publisher,
                        ConnectionState.FAILED,
                        f"gave up after {failures} attempts: {exc}",
                    )
                    return
                delay = min(self._backoff * 2 ** (failures - 1), self._max_backoff)
                self._emit(
                    publisher, ConnectionState.RECONNECTING, f"{exc}; retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            else:
                failures = 0

    async def _run_session(self, publisher: EventPublisher) -> None:
        gateway = self._factory(self._symbol, self._options)
        self._gateway = gateway
        try:
            await gateway.connect()
            info = await gateway.load_chain()
            spot = await gateway.wait_for_spot(SPOT_TIMEOUT_S)
            subscribed_spot = await self._subscribe(gateway, info, spot)
            self._emit(publisher, ConnectionState.CONNECTED)
            await asyncio.sleep(self._warmup)
            while True:
                if not gateway.is_connected:
                    raise ConnectionLostError("the broker connection dropped")
                broker = gateway.read()
                if (
                    math.isfinite(broker.spot)
                    and abs(broker.spot / subscribed_spot - 1.0) > self._options.band / 2.0
                ):
                    subscribed_spot = await self._subscribe(gateway, info, broker.spot)
                chain = build_snapshot(self._symbol, broker, info, self._options, self._clock.now())
                if chain is not None:
                    publisher.publish(ChainReceived(source=self.name, snapshot=chain))
                await asyncio.sleep(self._poll)
        finally:
            await gateway.disconnect()

    async def _subscribe(self, gateway: BrokerGateway, info: OptionChainInfo, spot: float) -> float:
        specs = select_contracts(
            info.expirations,
            info.strikes,
            spot,
            today=self._clock.now().date(),
            n_expiries=self._options.expiries,
            band=self._options.band,
            budget=self._options.max_lines - 1,
        )
        if not specs:
            raise ProviderError(f"no listed options near {spot:.2f} for {self._symbol}")
        await gateway.subscribe(specs)
        return spot

    def _emit(self, publisher: EventPublisher, state: ConnectionState, detail: str = "") -> None:
        publisher.publish(ConnectionStateChanged(source=self.name, state=state, detail=detail))


class IbAsyncGateway:
    """The ``ib_async`` implementation of :class:`BrokerGateway`."""

    def __init__(self, symbol: str, options: IbkrOptions) -> None:
        try:
            import ib_async  # noqa: PLC0415
        except ImportError as exc:
            raise ConfigurationError(
                "the ibkr source needs the optional dependency: uv sync --extra ibkr"
            ) from exc
        self._lib = ib_async
        self._symbol = symbol
        self._options = options
        self._ib = ib_async.IB()
        self._stock = ib_async.Stock(symbol, "SMART", "USD")
        self._underlying_ticker: object | None = None
        self._info: OptionChainInfo | None = None
        self._tickers: dict[OptionSpec, object] = {}

    @property
    def is_connected(self) -> bool:
        return bool(self._ib.isConnected())

    async def connect(self) -> None:
        o = self._options
        try:
            await self._ib.connectAsync(
                o.host, o.port, clientId=o.client_id, timeout=o.connect_timeout_s
            )
        except (OSError, TimeoutError) as exc:
            raise ConnectionLostError(
                f"cannot reach TWS/Gateway at {o.host}:{o.port}: {exc}"
            ) from exc
        self._ib.reqMarketDataType(o.market_data_type)

    async def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()

    async def load_chain(self) -> OptionChainInfo:
        qualified = await self._ib.qualifyContractsAsync(self._stock)
        if not qualified or not self._stock.conId:
            raise ProviderError(f"unknown underlying {self._symbol!r}")
        chains = await self._ib.reqSecDefOptParamsAsync(
            self._stock.symbol, "", self._stock.secType, self._stock.conId
        )
        smart = [c for c in chains if c.exchange == "SMART"] or list(chains)
        preferred = [c for c in smart if c.tradingClass == self._symbol] or smart
        if not preferred:
            raise ProviderError(f"{self._symbol} has no listed options")
        chain = max(preferred, key=lambda c: len(c.expirations))
        self._underlying_ticker = self._ib.reqMktData(self._stock, "", False, False)
        self._info = OptionChainInfo(
            expirations=tuple(sorted(_parse_expiry(e) for e in chain.expirations)),
            strikes=tuple(sorted(float(k) for k in chain.strikes)),
            multiplier=int(chain.multiplier or 100),
            trading_class=chain.tradingClass,
            exchange=chain.exchange,
        )
        return self._info

    async def wait_for_spot(self, timeout_s: float) -> float:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            spot = self._spot()
            if math.isfinite(spot) and spot > 0:
                return spot
            await asyncio.sleep(0.2)
        raise ProviderError(f"no price for {self._symbol} after {timeout_s:.0f}s")

    async def subscribe(self, specs: Sequence[OptionSpec]) -> None:
        if self._info is None:
            raise ProviderError("load_chain() must run before subscribe()")
        wanted = set(specs)
        for spec in list(self._tickers):
            if spec not in wanted:
                self._ib.cancelMktData(self._tickers.pop(spec).contract)  # type: ignore[attr-defined]
        fresh = [spec for spec in specs if spec not in self._tickers]
        contracts = {
            spec: self._lib.Option(
                self._symbol,
                spec.expiry.strftime("%Y%m%d"),
                spec.strike,
                spec.right.value,
                self._info.exchange,
                tradingClass=self._info.trading_class,
            )
            for spec in fresh
        }
        if contracts:
            await self._ib.qualifyContractsAsync(*contracts.values())
        for spec, contract in contracts.items():
            if contract.conId:
                self._tickers[spec] = self._ib.reqMktData(contract, GENERIC_TICKS, False, False)

    def read(self) -> BrokerSnapshot:
        quotes = []
        for spec, ticker in self._tickers.items():
            greeks = getattr(ticker, "modelGreeks", None)
            open_interest = (
                getattr(ticker, "callOpenInterest", math.nan)
                if spec.right is Right.CALL
                else getattr(ticker, "putOpenInterest", math.nan)
            )
            quotes.append(
                BrokerQuote(
                    spec=spec,
                    bid=float(getattr(ticker, "bid", math.nan)),
                    ask=float(getattr(ticker, "ask", math.nan)),
                    open_interest=float(open_interest),
                    model_iv=float(getattr(greeks, "impliedVol", math.nan) or math.nan),
                )
            )
        return BrokerSnapshot(spot=self._spot(), quotes=tuple(quotes))

    def _spot(self) -> float:
        ticker = self._underlying_ticker
        if ticker is None:
            return math.nan
        price = ticker.marketPrice()  # type: ignore[attr-defined]
        return float(price) if price is not None else math.nan


def _parse_expiry(text: str) -> date:
    return datetime.strptime(text, "%Y%m%d").date()
