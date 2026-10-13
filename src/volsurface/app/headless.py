"""Headless use cases behind the command-line interface: demo, record and benchmark.

They run the same container, session and pipeline as the GUI, just without a window, which is
what makes them useful as smoke tests and as scripts on a machine with no display.
"""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from volsurface.app.commands import SetRecording, StartSession, StopSession
from volsurface.app.config import Settings
from volsurface.app.container import open_container
from volsurface.app.events import SurfaceComputed
from volsurface.app.results import SessionUpdate
from volsurface.app.session import MarketSession
from volsurface.app.session_state import SessionState
from volsurface.app.warmup import warm_up_numerics
from volsurface.domain.errors import ProviderError
from volsurface.marketdata.factory import ProviderFactory, ProviderOptions, SourceSpec
from volsurface.marketdata.synthetic import SyntheticConfig
from volsurface.pricing.american import american_implied_vol, american_price_gamma
from volsurface.pricing.backends import NumbaCpuBackend
from volsurface.pricing.black_scholes import bs_price
from volsurface.pricing.implied_vol import implied_vol_bs
from volsurface.pricing.montecarlo import GbmModel, MonteCarloEngine, Payoff, PayoffKind
from volsurface.surface.svi import SviParams, total_variance
from volsurface.surface.svi_fit import fit_svi_slice

DEFAULT_TIMEOUT_S = 120.0


async def collect_updates(
    settings: Settings,
    spec: SourceSpec,
    count: int,
    *,
    options: ProviderOptions | None = None,
    factory: ProviderFactory | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[SessionUpdate]:
    """Run a session until ``count`` updates arrived (or the source ended) and return them."""
    updates: list[SessionUpdate] = []
    async with open_container(settings, options=options, factory=factory) as app:
        done = asyncio.Event()

        def keep(event: SurfaceComputed) -> None:
            updates.append(event.update)
            if len(updates) >= count:
                done.set()

        app.bus.subscribe(SurfaceComputed, keep, name="headless.collect")
        await app.dispatcher.dispatch(StartSession(spec))
        try:
            await asyncio.wait_for(_until(done, app.session), timeout_s)
        except TimeoutError as exc:
            raise ProviderError(f"no data from {spec} within {timeout_s:.0f}s") from exc
        await app.dispatcher.dispatch(StopSession())
        await app.bus.drain()
    return updates


async def _until(done: asyncio.Event, session: MarketSession) -> None:
    while not done.is_set():
        if session.state in {SessionState.STOPPED, SessionState.FAILED}:
            await asyncio.sleep(0.05)
            return
        await asyncio.sleep(0.02)


def fast_synthetic(interval_s: float = 0.05) -> ProviderOptions:
    return ProviderOptions(synthetic=SyntheticConfig(interval_s=interval_s))


def format_report(update: SessionUpdate) -> str:
    """A plain-text summary of one update: surface slices, GEX and timings."""
    chain = update.chain
    lines = [
        f"{chain.underlying}  {chain.asof:%Y-%m-%d %H:%M:%S %Z}  spot {chain.spot:.2f}  "
        f"{chain.size} options, {int(update.usable.sum())} usable",
    ]
    surface = update.surface
    if surface is None:
        lines.append("  no surface: " + "; ".join(update.warnings))
    else:
        lines.append(
            "  expiry        days   ATM vol%   RMSE vol pts  points  butterfly  Lee   rho    b"
        )
        for s in surface.slices:
            p = s.params
            lines.append(
                f"  {s.expiry}  {s.t * 365:5.0f}   {100 * s.atm_vol():7.2f}"
                f"   {100 * s.fit.rmse_vol:11.3f}"
                f"  {s.fit.n_points:6d}  {'ok' if s.fit.butterfly_free else 'VIOLATED':9s}"
                f"  {'ok' if s.fit.lee_ok else 'BAD':4s} {p.rho:+.2f}  {p.b:.3f}"
            )
        report = surface.arbitrage_report()
        lines.append(
            f"  static arbitrage: {'none found' if report.ok else 'PRESENT'}"
            f" ({report.butterfly_violations} butterfly, {len(report.calendar)} calendar)"
        )
        if surface.skipped:
            lines.append(
                "  skipped: " + ", ".join(f"{s.expiry} ({s.reason})" for s in surface.skipped)
            )
    if update.gex is not None:
        g = update.gex
        flip = f"{g.flip:.2f} ({g.distance_to_flip_pct:+.2f}%)" if np.isfinite(g.flip) else "none"
        lines.append(
            f"  GEX at spot {g.gex_at_spot:+.3f} bn$/1%   flip {flip}   {g.n_options} options"
        )
    lines.extend(f"  ALERT {alert.message}" for alert in update.alerts)
    parts = "  ".join(f"{name} {1000 * seconds:.1f} ms" for name, seconds in update.timings.items())
    lines.append(f"  timings: {parts}  total {1000 * update.timings.total_s:.1f} ms")
    lines.extend(f"  note: {warning}" for warning in update.warnings)
    return "\n".join(lines)


async def record(
    settings: Settings,
    spec: SourceSpec,
    out: Path,
    *,
    seconds: float,
    options: ProviderOptions | None = None,
) -> int:
    """Record ``spec`` into ``out`` for ``seconds`` and return the number of snapshots saved."""
    async with open_container(settings, options=options) as app:
        await app.dispatcher.dispatch(SetRecording(out))
        await app.dispatcher.dispatch(StartSession(spec))
        await asyncio.sleep(seconds)
        await app.dispatcher.dispatch(StopSession())
        await app.bus.drain()
        written = app.recorder.stop()
    return written


def run_benchmarks(report: Callable[[str], None] = print) -> list[tuple[str, float]]:
    """Quick wall-clock micro-benchmarks of the main kernels (milliseconds, best of several)."""
    warm_up_numerics(NumbaCpuBackend())
    rng = np.random.default_rng(7)
    n = 10_000
    strikes = 100.0 * rng.uniform(0.8, 1.2, n)
    t = rng.uniform(0.01, 2.0, n)
    call = rng.random(n) < 0.5
    price = bs_price(100.0, strikes, t, 0.03, 0.01, rng.uniform(0.1, 0.6, n), call)
    m = 1_500
    am_k, am_t = strikes[:m], t[:m] * 0.2
    am_sigma = rng.uniform(0.15, 0.5, m)
    am_price, _ = american_price_gamma(100.0, am_k, am_t, 0.03, 0.01, am_sigma, call[:m])
    k = np.linspace(-0.45, 0.45, 40)
    truth = SviParams(0.01, 0.15, -0.45, 0.02, 0.12)
    w = total_variance(k, truth) + rng.normal(0, 2e-4, k.size)
    seed = fit_svi_slice(k, w).params
    engine = MonteCarloEngine(GbmModel(100.0, 0.03, 0.01, 0.2))
    payoff = Payoff(PayoffKind.CALL, 100.0)

    cases: list[tuple[str, Callable[[], object]]] = [
        (
            "Black-Scholes IV, 10,000 options",
            lambda: implied_vol_bs(price, 100.0, strikes, t, 0.03, 0.01, call),
        ),
        (
            "CRR price+gamma, 1,500 options, 200 steps",
            lambda: american_price_gamma(100.0, am_k, am_t, 0.03, 0.01, am_sigma, call[:m]),
        ),
        (
            "CRR implied vol (Brent), 1,500 options",
            lambda: american_implied_vol(am_price, 100.0, am_k, am_t, 0.03, 0.01, call[:m]),
        ),
        ("SVI slice fit, cold start", lambda: fit_svi_slice(k, w)),
        ("SVI slice fit, warm start", lambda: fit_svi_slice(k, w, initial=seed)),
        ("Monte Carlo GBM, 1,000,000 paths", lambda: engine.price(payoff, 1.0, 1_000_000)),
    ]
    results: list[tuple[str, float]] = []
    for name, run in cases:
        best = float("inf")
        for _ in range(5):
            start = time.perf_counter()
            run()
            best = min(best, time.perf_counter() - start)
        results.append((name, 1000 * best))
        report(f"{name:46s} {1000 * best:10.2f} ms")
    return results
