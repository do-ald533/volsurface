"""Test doubles and builders shared by several test modules."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

from volsurface.domain.errors import ConnectionLostError
from volsurface.domain.types import ChainSnapshot
from volsurface.marketdata.ibkr import (
    BrokerQuote,
    BrokerSnapshot,
    OptionChainInfo,
    OptionSpec,
)
from volsurface.marketdata.options import IbkrOptions
from volsurface.marketdata.synthetic import SyntheticChain, SyntheticConfig

T0 = datetime(2026, 10, 2, 14, 0, tzinfo=UTC)


def small_chain(
    *, tick: int = 0, spot: float = 100.0, vol_scale: float = 1.0, seed: int = 5
) -> ChainSnapshot:
    config = SyntheticConfig(seed=seed, expiry_days=(14, 30, 60), strike_step=2.0)
    return SyntheticChain(config).snapshot(T0 + timedelta(seconds=30 * tick), spot, vol_scale, tick)


def listed_chain(today: date | None = None) -> OptionChainInfo:
    today = today or T0.date()
    expirations = tuple(today + timedelta(days=d) for d in (7, 14, 21, 30, 60, 90, 120))
    strikes = tuple(float(k) for k in range(70, 131))
    return OptionChainInfo(expirations, strikes, multiplier=100, trading_class="SPY")


class FakeGateway:
    """In-memory ``BrokerGateway``: scripted connection failures and a drifting spot."""

    def __init__(
        self,
        *,
        info: OptionChainInfo | None = None,
        spot: float = 100.0,
        connect_failures: int = 0,
        drop_after_reads: int | None = None,
        spot_after_reads: dict[int, float] | None = None,
    ) -> None:
        self.info = info or listed_chain()
        self.spot = spot
        self.connect_failures = connect_failures
        self.drop_after_reads = drop_after_reads
        self.spot_after_reads = spot_after_reads or {}
        self.reads = 0
        self.connected = False
        self.disconnects = 0
        self.subscriptions: list[tuple[OptionSpec, ...]] = []
        self._active: tuple[OptionSpec, ...] = ()

    @property
    def is_connected(self) -> bool:
        if self.drop_after_reads is not None and self.reads >= self.drop_after_reads:
            return False
        return self.connected

    async def connect(self) -> None:
        if self.connect_failures > 0:
            self.connect_failures -= 1
            raise ConnectionLostError("scripted connection failure")
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False
        self.disconnects += 1

    async def load_chain(self) -> OptionChainInfo:
        return self.info

    async def wait_for_spot(self, timeout_s: float) -> float:
        return self.spot

    async def subscribe(self, specs: Sequence[OptionSpec]) -> None:
        self._active = tuple(specs)
        self.subscriptions.append(self._active)

    def read(self) -> BrokerSnapshot:
        self.reads += 1
        spot = self.spot_after_reads.get(self.reads, self.spot)
        quotes = []
        for spec in self._active:
            intrinsic = max((spot - spec.strike) * spec.right.sign, 0.0)
            mid = intrinsic + 1.0 + 0.01 * abs(spec.strike - spot)
            quotes.append(
                BrokerQuote(
                    spec=spec,
                    bid=round(mid - 0.05, 2),
                    ask=round(mid + 0.05, 2),
                    open_interest=500.0,
                    model_iv=0.2,
                )
            )
        return BrokerSnapshot(spot=spot, quotes=tuple(quotes))


def fake_options(**overrides: float) -> IbkrOptions:
    return replace(IbkrOptions(max_lines=30, expiries=3, band=0.1), **overrides)


async def wait_until(
    predicate: Callable[[], bool], timeout: float = 5.0, step: float = 0.005
) -> None:
    """Poll ``predicate`` on the event loop until it is true or ``timeout`` seconds pass."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("condition not reached in time")
        await asyncio.sleep(step)
