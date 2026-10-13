"""Synthetic market data: a chain with a known volatility surface, streamed as events.

The ground truth is an SSVI surface (:mod:`volsurface.surface.ssvi`). Option quotes are produced by
pricing every contract with the American CRR tree at the true volatility plus noise, then adding a
bid-ask spread and tick rounding, so inverting those quotes with the same tree should recover the
surface. Open interest is shaped (calls peaking above, puts below) around the *initial* spot and
then stays at those strikes while spot moves, as real open interest does, so GEX changes sign
when spot crosses the middle of the distribution.

Nothing here touches the network or the wall clock: simulated time advances by ``sim_step`` per
tick and every random draw comes from ``default_rng([seed, tick])``, so a run is reproducible and
independent of how fast it is consumed.
"""

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

from volsurface.domain.calendar import years_to_expiry
from volsurface.domain.errors import InvalidInputError
from volsurface.domain.events import ChainReceived, ConnectionState, ConnectionStateChanged
from volsurface.domain.ports import EventPublisher
from volsurface.domain.types import ChainSnapshot, ExerciseStyle, FloatArray
from volsurface.pricing.american import american_price_gamma
from volsurface.surface.ssvi import SsviSurface

DEFAULT_EXPIRY_DAYS = (7, 14, 30, 60, 90, 150)
DEFAULT_ATM_VOLS = (0.170, 0.175, 0.180, 0.190, 0.200, 0.205)
DEFAULT_START = datetime(2026, 10, 2, 14, 0, tzinfo=UTC)


def default_ssvi(expiry_days: Sequence[int] = DEFAULT_EXPIRY_DAYS) -> SsviSurface:
    """An equity-index-like surface: negative skew, mild upward term structure."""
    days = np.asarray(expiry_days, dtype=np.float64)
    vols = np.interp(days, DEFAULT_EXPIRY_DAYS, DEFAULT_ATM_VOLS)
    maturities = days / 365.0
    return SsviSurface(
        maturities=maturities,
        atm_total_variance=vols**2 * maturities,
        rho=-0.7,
        eta=0.9,
        gamma=0.5,
    )


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Everything that defines a synthetic market."""

    underlying: str = "SYN"
    spot0: float = 100.0
    rate: float = 0.04
    dividend_yield: float = 0.012
    start: datetime = DEFAULT_START
    sim_step: timedelta = timedelta(seconds=30)
    interval_s: float = 0.5
    expiry_days: tuple[int, ...] = DEFAULT_EXPIRY_DAYS
    moneyness_range: tuple[float, float] = (0.78, 1.22)
    strike_step: float = 1.0
    iv_noise: float = 0.002
    spread_fraction: float = 0.02
    min_spread: float = 0.02
    spot_vol_per_tick: float = 0.0008
    vol_mean_reversion: float = 0.92
    vol_shock: float = 0.01
    leverage: float = 3.0
    spot_path: tuple[float, ...] | None = None
    max_ticks: int | None = None
    tree_steps: int = 120
    seed: int = 7
    ssvi: SsviSurface | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.start.tzinfo is None:
            raise InvalidInputError("start must be timezone-aware")
        if self.spot0 <= 0 or self.strike_step <= 0 or self.interval_s < 0:
            raise InvalidInputError("spot0 and strike_step must be positive; interval_s >= 0")
        lo, hi = self.moneyness_range
        if not 0 < lo < 1 < hi:
            raise InvalidInputError("moneyness_range must straddle 1")
        if not self.expiry_days or any(d <= 0 for d in self.expiry_days):
            raise InvalidInputError("expiry_days must be positive")
        if self.max_ticks is not None and self.max_ticks < 1:
            raise InvalidInputError("max_ticks must be at least 1")

    def surface(self) -> SsviSurface:
        return self.ssvi if self.ssvi is not None else default_ssvi(self.expiry_days)


class SyntheticChain:
    """Pure generator: (time, spot, volatility level) in, ``ChainSnapshot`` out."""

    def __init__(self, config: SyntheticConfig) -> None:
        self._config = config
        self._surface = config.surface()

    @property
    def truth(self) -> SsviSurface:
        return self._surface

    def snapshot(
        self, asof: datetime, spot: float, vol_scale: float, tick: int = 0
    ) -> ChainSnapshot:
        cfg = self._config
        rng = np.random.default_rng([cfg.seed, tick])
        expiry_dates = [(asof + timedelta(days=d)).date() for d in cfg.expiry_days]
        lo, hi = cfg.moneyness_range
        strikes = np.arange(
            math.ceil(spot * lo / cfg.strike_step) * cfg.strike_step,
            spot * hi,
            cfg.strike_step,
        )
        n_exp, n_k = len(expiry_dates), strikes.size

        expiry = np.repeat(np.array(expiry_dates, dtype="datetime64[D]"), n_k * 2)
        strike = np.tile(np.repeat(strikes, 2), n_exp)
        is_call = np.tile(np.array([True, False]), n_exp * n_k)

        t = years_to_expiry(asof, expiry)
        forward = spot * np.exp((cfg.rate - cfg.dividend_yield) * t)
        k = np.log(strike / forward)
        true_iv = np.sqrt(self._surface.total_variance(k, t) / t) * vol_scale
        noisy_iv = np.maximum(true_iv + rng.normal(0.0, cfg.iv_noise, true_iv.shape), 0.03)

        mid, _ = american_price_gamma(
            spot,
            strike,
            t,
            cfg.rate,
            cfg.dividend_yield,
            noisy_iv,
            is_call,
            steps=cfg.tree_steps,
        )
        half = np.maximum(cfg.min_spread, cfg.spread_fraction * mid) / 2.0
        bid = np.floor(np.maximum(mid - half, 0.0) * 100.0) / 100.0
        ask = np.ceil((mid + half) * 100.0) / 100.0

        open_interest = self._open_interest(strike, cfg.spot0, t, is_call, rng)
        return ChainSnapshot(
            underlying=cfg.underlying,
            asof=asof,
            spot=spot,
            rate=cfg.rate,
            dividend_yield=cfg.dividend_yield,
            expiry=expiry,
            strike=strike,
            is_call=is_call,
            bid=bid,
            ask=ask,
            open_interest=open_interest,
            vendor_iv=np.full(strike.shape, np.nan),
            exercise=ExerciseStyle.AMERICAN,
        )

    @staticmethod
    def _open_interest(
        strike: FloatArray,
        anchor: float,
        t: FloatArray,
        is_call: np.ndarray,
        rng: np.random.Generator,
    ) -> FloatArray:
        x = np.log(strike / anchor)
        calls = 8000.0 * np.exp(-0.5 * ((x - 0.03) / 0.06) ** 2)
        puts = 9000.0 * np.exp(-0.5 * ((x + 0.04) / 0.07) ** 2)
        base = np.where(is_call, calls, puts) * np.exp(-t / 0.35)
        noise = rng.lognormal(0.0, 0.3, base.shape)
        return np.round(base * noise)


class SyntheticProvider:
    """Streams synthetic chains as :class:`~volsurface.domain.events.ChainReceived` events."""

    name = "synthetic"

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self._config = config or SyntheticConfig()
        self._chain = SyntheticChain(self._config)
        self._task: asyncio.Task[None] | None = None

    @property
    def config(self) -> SyntheticConfig:
        return self._config

    @property
    def truth(self) -> SsviSurface:
        return self._chain.truth

    async def start(self, publisher: EventPublisher) -> None:
        if self._task is not None:
            return
        self._task = asyncio.get_running_loop().create_task(
            self._run(publisher), name="synthetic-provider"
        )

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def wait_finished(self) -> None:
        """Block until a ``max_ticks`` run has produced every tick (for tests and the CLI)."""
        if self._task is not None:
            await asyncio.shield(self._task)

    async def _run(self, publisher: EventPublisher) -> None:
        cfg = self._config
        publisher.publish(ConnectionStateChanged(source=self.name, state=ConnectionState.CONNECTED))
        spot, vol_scale = cfg.spot0, 1.0
        tick = 0
        while cfg.max_ticks is None or tick < cfg.max_ticks:
            asof = cfg.start + cfg.sim_step * tick
            spot, vol_scale = self._advance(spot, vol_scale, tick)
            snapshot = await asyncio.to_thread(self._chain.snapshot, asof, spot, vol_scale, tick)
            publisher.publish(ChainReceived(source=self.name, snapshot=snapshot))
            tick += 1
            await asyncio.sleep(cfg.interval_s)
        publisher.publish(
            ConnectionStateChanged(
                source=self.name, state=ConnectionState.DISCONNECTED, detail="finished"
            )
        )

    def _advance(self, spot: float, vol_scale: float, tick: int) -> tuple[float, float]:
        cfg = self._config
        if tick == 0:
            return (cfg.spot_path[0] if cfg.spot_path else cfg.spot0), 1.0
        rng = np.random.default_rng([cfg.seed, 10_000_000 + tick])
        if cfg.spot_path is not None:
            new_spot = cfg.spot_path[min(tick, len(cfg.spot_path) - 1)]
            ret = math.log(new_spot / spot)
        else:
            ret = float(rng.normal(0.0, cfg.spot_vol_per_tick))
            new_spot = spot * math.exp(ret)
        shock = float(rng.normal(0.0, cfg.vol_shock))
        new_scale = 1.0 + cfg.vol_mean_reversion * (vol_scale - 1.0) + shock - cfg.leverage * ret
        return new_spot, float(np.clip(new_scale, 0.6, 2.0))
