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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol

import numpy as np

from volsurface.domain.clock import SystemClock
from volsurface.domain.errors import (
    ConfigurationError,
    ConnectionLostError,
    MarketDataUnavailableError,
    ProviderError,
)
from volsurface.domain.events import ChainReceived, ConnectionState, ConnectionStateChanged
from volsurface.domain.ports import Clock, EventPublisher
from volsurface.domain.types import ChainSnapshot, ExerciseStyle, Right
from volsurface.marketdata.options import IbkrOptions

GENERIC_TICKS = "101"
SPOT_TIMEOUT_S = 15.0
MAX_PARALLEL_LISTINGS = 6
NOT_SUBSCRIBED_CODES = frozenset({354, 10168})
_RIGHTS = {"C": Right.CALL, "P": Right.PUT}


def explain_ib_error(code: int, text: str, *, market_data_type: int) -> str | None:
    """An actionable message for broker errors that retrying cannot fix, else ``None``."""
    if code not in NOT_SUBSCRIBED_CODES:
        return None
    advice = (
        "set VOLSURFACE_IBKR__MARKET_DATA_TYPE=3 to use free delayed data"
        if market_data_type != 3
        else "subscribe to market data for this instrument in the IBKR account"
    )
    return f"IBKR error {code}: {text.strip()} Your account has no live subscription; {advice}."


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

    async def list_strikes(self, expiry: date) -> tuple[float, ...]:
        """The strikes actually listed for ``expiry``; they differ between expirations."""
        ...

    async def subscribe(self, specs: Sequence[OptionSpec]) -> None:
        """Make ``specs`` the active subscription set, cancelling anything not in it."""
        ...

    def read(self) -> BrokerSnapshot: ...


def spread_expiries(
    expirations: Iterable[date], *, today: date, count: int, min_days: int, max_days: int
) -> list[date]:
    """Pick ``count`` expirations spread geometrically over the maturities on offer.

    Listed expirations cluster at the short end (SPY has daily ones), so the nearest few would
    describe one week of the term structure. Targets are log-spaced between the first and last
    eligible expiry and each is matched to the closest unused listed date. If nothing falls in
    ``[min_days, max_days]`` the nearest dates on or after ``today`` are used instead.
    """
    future = sorted({e for e in expirations if e >= today})
    eligible = [e for e in future if min_days <= (e - today).days <= max_days] or future[:count]
    if len(eligible) <= count:
        return eligible
    log_days = np.log(np.maximum([(e - today).days for e in eligible], 1.0))
    targets = np.linspace(log_days[0], log_days[-1], count)
    free = np.ones(len(eligible), dtype=bool)
    chosen: list[int] = []
    for target in targets:
        distance = np.where(free, np.abs(log_days - target), np.inf)
        index = int(np.argmin(distance))
        free[index] = False
        chosen.append(index)
    return [eligible[i] for i in sorted(chosen)]


def spread_strikes(
    strikes: Iterable[float], spot: float, *, band: float, count: int
) -> list[float]:
    """Pick ``count`` listed strikes spread evenly in log-moneyness over ``spot * (1 +/- band)``.

    Taking the strikes nearest to spot would, for $1 strike spacing, cover well under one percent
    around the money and leave the smile's wings unconstrained.
    """
    low, high = spot * (1.0 - band), spot * (1.0 + band)
    inside = np.array(sorted(k for k in set(strikes) if low <= k <= high), dtype=np.float64)
    if inside.size <= count:
        return [float(k) for k in inside]
    targets = spot * np.exp(np.linspace(np.log(1.0 - band), np.log(1.0 + band), count))
    free = np.ones(inside.size, dtype=bool)
    chosen: list[int] = []
    for target in targets:
        distance = np.where(free, np.abs(inside - target), np.inf)
        index = int(np.argmin(distance))
        free[index] = False
        chosen.append(index)
    return [float(inside[i]) for i in sorted(chosen)]


def select_contracts(
    expirations: Sequence[date],
    strikes: Sequence[float] | Mapping[date, Sequence[float]],
    spot: float,
    *,
    today: date,
    n_expiries: int,
    band: float,
    budget: int,
    min_days: int = 0,
    max_days: int = 100_000,
) -> list[OptionSpec]:
    """Choose at most ``budget`` out-of-the-money options covering the surface.

    ``n_expiries`` expirations are spread over ``[min_days, max_days]`` (see
    :func:`spread_expiries`), the budget is split evenly between them, and each expiry gets
    strikes spread across ``spot * (1 +/- band)`` (see :func:`spread_strikes`). ``strikes`` is
    either one list for every expiry or a mapping with the strikes each expiry really lists.
    Strikes at or above spot are calls and the rest puts.
    """
    if budget < 1:
        raise ConfigurationError("the market-data line budget leaves no room for options")
    chosen_expiries = spread_expiries(
        expirations, today=today, count=n_expiries, min_days=min_days, max_days=max_days
    )
    if not chosen_expiries:
        return []
    quota = max(budget // len(chosen_expiries), 1)
    specs: list[OptionSpec] = []
    for expiry in chosen_expiries:
        listed = strikes.get(expiry, ()) if isinstance(strikes, Mapping) else strikes
        specs.extend(
            OptionSpec(expiry, strike, Right.CALL if strike >= spot else Right.PUT)
            for strike in spread_strikes(listed, spot, band=band, count=quota)
        )
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
        self._strikes: dict[date, tuple[float, ...]] = {}
        self._failures = 0

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
            self._failures = failures
            self._stage(publisher)
            try:
                await self._run_session(publisher)
            except asyncio.CancelledError:
                raise
            except MarketDataUnavailableError as exc:
                self._emit(publisher, ConnectionState.FAILED, str(exc))
                return
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
        self._strikes = {}
        o = self._options
        try:
            self._stage(publisher, f"connecting to TWS at {o.host}:{o.port}")
            await gateway.connect()
            self._stage(publisher, f"loading the {self._symbol} option chain")
            info = await gateway.load_chain()
            self._stage(publisher, f"waiting for the {self._symbol} price")
            spot = await gateway.wait_for_spot(SPOT_TIMEOUT_S)
            subscribed_spot = await self._subscribe(gateway, info, spot, publisher)
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
                    subscribed_spot = await self._subscribe(gateway, info, broker.spot, publisher)
                chain = build_snapshot(self._symbol, broker, info, self._options, self._clock.now())
                if chain is not None:
                    publisher.publish(ChainReceived(source=self.name, snapshot=chain))
                await asyncio.sleep(self._poll)
        finally:
            await gateway.disconnect()

    async def _subscribe(
        self,
        gateway: BrokerGateway,
        info: OptionChainInfo,
        spot: float,
        publisher: EventPublisher,
    ) -> float:
        o = self._options
        today = self._clock.now().date()
        expiries = spread_expiries(
            info.expirations,
            today=today,
            count=o.expiries,
            min_days=o.min_days,
            max_days=o.max_days,
        )
        await self._list_strikes(
            gateway, [e for e in expiries if e not in self._strikes], publisher
        )
        specs = select_contracts(
            expiries,
            self._strikes,
            spot,
            today=today,
            n_expiries=len(expiries),
            band=o.band,
            budget=o.max_lines - 1,
            min_days=o.min_days,
            max_days=o.max_days,
        )
        if not specs:
            raise ProviderError(f"no listed options near {spot:.2f} for {self._symbol}")
        self._stage(publisher, f"subscribing to {len(specs)} options")
        await gateway.subscribe(specs)
        return spot

    async def _list_strikes(
        self, gateway: BrokerGateway, missing: list[date], publisher: EventPublisher
    ) -> None:
        """Fetch the strike lists concurrently: each request is slow for a big monthly chain."""
        if not missing:
            return
        self._stage(publisher, f"listing strikes for {len(missing)} expiries")
        limit = asyncio.Semaphore(MAX_PARALLEL_LISTINGS)

        async def fetch(expiry: date) -> tuple[date, tuple[float, ...]]:
            async with limit:
                return expiry, await gateway.list_strikes(expiry)

        tasks = [asyncio.ensure_future(fetch(expiry)) for expiry in missing]
        try:
            for number, finished in enumerate(asyncio.as_completed(tasks), start=1):
                expiry, strikes = await finished
                self._strikes[expiry] = strikes
                self._stage(publisher, f"listed strikes {number}/{len(missing)} (expiry {expiry})")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    def _stage(self, publisher: EventPublisher, detail: str = "") -> None:
        """Report set-up progress; it counts as reconnecting once an attempt has failed."""
        state = ConnectionState.RECONNECTING if self._failures else ConnectionState.CONNECTING
        self._emit(publisher, state, detail)

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
        self._listed: dict[OptionSpec, Any] = {}
        self._data_error: str | None = None
        self._ib.errorEvent += self._on_error

    def _on_error(self, req_id: int, code: int, text: str, contract: object) -> None:
        if getattr(contract, "secType", "") != "STK":
            return
        message = explain_ib_error(code, text, market_data_type=self._options.market_data_type)
        if message is not None:
            self._data_error = message

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
            if self._data_error is not None:
                raise MarketDataUnavailableError(self._data_error)
            await asyncio.sleep(0.2)
        raise ProviderError(f"no price for {self._symbol} after {timeout_s:.0f}s")

    async def list_strikes(self, expiry: date) -> tuple[float, ...]:
        if self._info is None:
            raise ProviderError("load_chain() must run before list_strikes()")
        template = self._lib.Option(
            self._symbol,
            expiry.strftime("%Y%m%d"),
            exchange=self._info.exchange,
            currency="USD",
            tradingClass=self._info.trading_class,
        )
        details = await self._ib.reqContractDetailsAsync(template)
        strikes: set[float] = set()
        for item in details:
            contract: Any = item.contract
            if contract is None:
                continue
            right = _RIGHTS.get(str(contract.right).upper()[:1])
            if right is None or not contract.strike > 0:
                continue
            self._listed[OptionSpec(expiry, float(contract.strike), right)] = contract
            strikes.add(float(contract.strike))
        return tuple(sorted(strikes))

    async def subscribe(self, specs: Sequence[OptionSpec]) -> None:
        wanted = set(specs)
        for spec in list(self._tickers):
            if spec not in wanted:
                self._ib.cancelMktData(self._tickers.pop(spec).contract)  # type: ignore[attr-defined]
        for spec in specs:
            contract = self._listed.get(spec)
            if contract is not None and spec not in self._tickers:
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
