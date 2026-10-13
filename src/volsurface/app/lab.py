"""The calculations behind the Pricing Lab tabs, free of any Qt.

Three small use cases, each taking a frozen input record and returning a frozen result:
vanilla pricing (Black-Scholes next to the American tree), Monte Carlo, and a step-by-step view of
Newton-Raphson implied volatility.
"""

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from volsurface.domain.errors import ConvergenceError, InvalidInputError
from volsurface.domain.types import BoolArray, FloatArray
from volsurface.pricing.backends import PricingBackend
from volsurface.pricing.black_scholes import bs_greeks, bs_price
from volsurface.pricing.implied_vol import NewtonStep, newton_iterations
from volsurface.pricing.montecarlo import (
    GbmModel,
    HestonModel,
    McResult,
    MonteCarloEngine,
    PathModel,
    Payoff,
    PayoffKind,
    TwoHestonModel,
)
from volsurface.pricing.montecarlo.engine import Z_95

DAYS_PER_YEAR = 365.0
ModelName = Literal["gbm", "heston", "rainbow"]
VANILLA_METRICS: Final = ("Value", "Delta", "Gamma", "Vega", "Theta", "Rho")
NEWTON_SNAP: Final = 0.012
MIN_USABLE_VEGA: Final = 1e-12


@dataclass(frozen=True, slots=True)
class HoverInfo:
    """What a chart shows while the mouse rests on it, built without Qt so it can be tested.

    ``x`` is where the crosshair snaps to, ``rows`` are label/value pairs for the tooltip and
    ``points`` are chart positions to mark with a dot.
    """

    x: float
    title: str
    rows: tuple[tuple[str, str], ...]
    points: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class VanillaInputs:
    spot: float = 100.0
    strike: float = 100.0
    days: float = 90.0
    rate: float = 0.04
    dividend_yield: float = 0.012
    vol: float = 0.20
    is_call: bool = False
    tree_steps: int = 200

    @property
    def years(self) -> float:
        return self.days / DAYS_PER_YEAR


@dataclass(frozen=True, slots=True)
class VanillaQuote:
    bs_price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    american_price: float
    american_gamma: float

    @property
    def early_exercise_premium(self) -> float:
        return self.american_price - self.bs_price


def price_vanilla(inputs: VanillaInputs, backend: PricingBackend) -> VanillaQuote:
    """European value and Greeks (Black-Scholes) and the American tree value and gamma."""
    s, k, t = inputs.spot, inputs.strike, inputs.years
    args = (s, k, t, inputs.rate, inputs.dividend_yield, inputs.vol, inputs.is_call)
    greeks = bs_greeks(*args)
    price, gamma = backend.american_price_gamma(*args, steps=inputs.tree_steps)
    return VanillaQuote(
        bs_price=float(bs_price(*args)),
        delta=float(greeks.delta),
        gamma=float(greeks.gamma),
        vega=float(greeks.vega),
        theta=float(greeks.theta),
        rho=float(greeks.rho),
        american_price=float(price),
        american_gamma=float(gamma),
    )


@dataclass(frozen=True, slots=True, eq=False)
class VanillaCurves:
    """Black-Scholes value and Greeks, and the American tree value, across a range of spots."""

    inputs: VanillaInputs
    spots: FloatArray
    intrinsic: FloatArray
    bs_value: FloatArray
    american_value: FloatArray
    greeks: dict[str, FloatArray]

    def series(self, metric: str) -> list[tuple[str, FloatArray]]:
        """The named curves to draw for ``metric`` (one of :data:`VANILLA_METRICS`)."""
        if metric == "Value":
            return [
                ("Black-Scholes", self.bs_value),
                ("American tree", self.american_value),
                ("Intrinsic", self.intrinsic),
            ]
        return [(metric, self.greeks[metric.lower()])]

    def hover(self, spot: float, metric: str = "Value") -> HoverInfo:
        """Everything known at the grid spot closest to ``spot``."""
        i = int(np.argmin(np.abs(self.spots - spot)))
        x = float(self.spots[i])
        bs, american, intrinsic = (
            float(self.bs_value[i]),
            float(self.american_value[i]),
            float(self.intrinsic[i]),
        )
        g = {name: float(values[i]) for name, values in self.greeks.items()}
        rows = (
            ("Black-Scholes", f"{bs:.4f}"),
            ("American tree", f"{american:.4f}"),
            ("Intrinsic", f"{intrinsic:.4f}"),
            ("Time value", f"{bs - intrinsic:+.4f}"),
            ("Delta", f"{g['delta']:+.4f}"),
            ("Gamma", f"{g['gamma']:.5f}"),
            ("Vega (per 1.00 vol)", f"{g['vega']:.4f}"),
            ("Theta (per year)", f"{g['theta']:+.4f}"),
            ("Rho", f"{g['rho']:+.4f}"),
        )
        change = x / self.inputs.spot - 1.0
        points = tuple((x, float(values[i])) for _, values in self.series(metric))
        return HoverInfo(x, f"Spot {x:.2f}  ({change:+.1%} vs current)", rows, points)


def vanilla_curves(
    inputs: VanillaInputs, backend: PricingBackend, *, points: int = 121
) -> VanillaCurves:
    """Value and Greeks for spots from 0.7x the lower to 1.3x the higher of spot and strike."""
    low = 0.7 * min(inputs.spot, inputs.strike)
    high = 1.3 * max(inputs.spot, inputs.strike)
    spots = np.linspace(low, high, points)
    args = (
        spots,
        inputs.strike,
        inputs.years,
        inputs.rate,
        inputs.dividend_yield,
        inputs.vol,
        inputs.is_call,
    )
    greeks = bs_greeks(*args)
    american, _ = backend.american_price_gamma(*args, steps=inputs.tree_steps)
    sign = 1.0 if inputs.is_call else -1.0
    return VanillaCurves(
        inputs=inputs,
        spots=spots,
        intrinsic=np.maximum(sign * (spots - inputs.strike), 0.0),
        bs_value=np.asarray(bs_price(*args), dtype=np.float64),
        american_value=np.asarray(american, dtype=np.float64),
        greeks={
            "delta": np.asarray(greeks.delta, dtype=np.float64),
            "gamma": np.asarray(greeks.gamma, dtype=np.float64),
            "vega": np.asarray(greeks.vega, dtype=np.float64),
            "theta": np.asarray(greeks.theta, dtype=np.float64),
            "rho": np.asarray(greeks.rho, dtype=np.float64),
        },
    )


@dataclass(frozen=True, slots=True)
class McInputs:
    model: ModelName = "gbm"
    payoff: PayoffKind = PayoffKind.CALL
    spot: float = 100.0
    strike: float = 100.0
    days: float = 365.0
    rate: float = 0.04
    dividend_yield: float = 0.012
    vol: float = 0.20
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    rho_assets: float = 0.0
    n_paths: int = 200_000
    steps: int = 100
    antithetic: bool = True
    seed: int = 42

    @property
    def years(self) -> float:
        return self.days / DAYS_PER_YEAR


@dataclass(frozen=True, slots=True)
class McOutcome:
    result: McResult
    model_name: str
    reference: float | None
    feller_ok: bool | None

    @property
    def deviation_in_se(self) -> float | None:
        """How many standard errors the estimate is from the closed-form reference."""
        if self.reference is None or not self.result.std_error > 0:
            return None
        return (self.result.price - self.reference) / self.result.std_error

    def shown_mask(self) -> BoolArray:
        """Checkpoints worth plotting: the first few hundred paths only add noise."""
        counts = self.result.counts
        shown = counts >= max(500.0, 0.02 * counts[-1])
        return shown if int(shown.sum()) >= 3 else np.ones_like(counts, dtype=np.bool_)

    def convergence(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Plotted checkpoints as ``(paths, estimate, lower, upper)`` with a 95% band."""
        result, shown = self.result, self.shown_mask()
        price, band = result.running_price[shown], Z_95 * result.running_std_error[shown]
        return result.counts[shown], price, price - band, price + band

    def hover(self, paths: float) -> HoverInfo:
        """The running estimate at the plotted checkpoint closest to ``paths``."""
        result = self.result
        candidates = np.flatnonzero(self.shown_mask())
        i = int(candidates[np.argmin(np.abs(result.counts[candidates] - paths))])
        n = float(result.counts[i])
        price, error = float(result.running_price[i]), float(result.running_std_error[i])
        rows = [
            ("Estimate", f"{price:.4f}"),
            ("Std error", f"{error:.4f}"),
            ("95% CI", f"{price - Z_95 * error:.4f} to {price + Z_95 * error:.4f}"),
        ]
        if self.reference is not None:
            rows.append(("Black-Scholes", f"{self.reference:.4f}"))
            if error > 0:
                rows.append(("Off by", f"{(price - self.reference) / error:+.2f} s.e."))
        return HoverInfo(n, f"{n:,.0f} paths", tuple(rows), ((n, price),))


def build_model(inputs: McInputs) -> PathModel:
    match inputs.model:
        case "gbm":
            return GbmModel(inputs.spot, inputs.rate, inputs.dividend_yield, inputs.vol)
        case "heston":
            return HestonModel(
                inputs.spot,
                inputs.v0,
                inputs.rate,
                inputs.dividend_yield,
                inputs.kappa,
                inputs.theta,
                inputs.xi,
                inputs.rho,
            )
        case "rainbow":
            return TwoHestonModel(
                inputs.spot,
                inputs.v0,
                inputs.rate,
                inputs.dividend_yield,
                inputs.kappa,
                inputs.theta,
                inputs.xi,
                inputs.rho,
                inputs.rho_assets,
            )
    raise InvalidInputError(f"unknown model {inputs.model!r}")


def run_monte_carlo(inputs: McInputs) -> McOutcome:
    """Price the payoff; for GBM vanilla payoffs also report the Black-Scholes reference."""
    model = build_model(inputs)
    engine = MonteCarloEngine(model, steps=inputs.steps, antithetic=inputs.antithetic)
    result = engine.price(
        Payoff(inputs.payoff, inputs.strike), inputs.years, inputs.n_paths, seed=inputs.seed
    )
    reference = None
    if inputs.model == "gbm" and inputs.payoff in {PayoffKind.CALL, PayoffKind.PUT}:
        reference = float(
            bs_price(
                inputs.spot,
                inputs.strike,
                inputs.years,
                inputs.rate,
                inputs.dividend_yield,
                inputs.vol,
                inputs.payoff is PayoffKind.CALL,
            )
        )
    feller = model.feller_satisfied if isinstance(model, HestonModel | TwoHestonModel) else None
    return McOutcome(result, model.name, reference, feller)


@dataclass(frozen=True, slots=True)
class NewtonInputs:
    spot: float = 100.0
    strike: float = 105.0
    days: float = 90.0
    rate: float = 0.04
    dividend_yield: float = 0.012
    market_vol: float = 0.30
    start_vol: float = 0.10
    is_call: bool = True
    tol: float = 1e-8

    @property
    def years(self) -> float:
        return self.days / DAYS_PER_YEAR


@dataclass(frozen=True, slots=True, eq=False)
class NewtonTrace:
    """Everything needed to draw Newton-Raphson on the price-versus-volatility curve."""

    vols: FloatArray
    prices: FloatArray
    market_price: float
    steps: tuple[NewtonStep, ...]
    converged: bool
    message: str
    inputs: NewtonInputs

    def hover(self, vol_pct: float, *, upto: int | None = None) -> HoverInfo:
        """The iteration under the cursor, or else what Newton would do from that volatility."""
        visible = self.steps if upto is None else self.steps[: upto + 1]
        span = 100.0 * float(self.vols[-1] - self.vols[0])
        near = [s for s in visible if abs(100.0 * s.sigma - vol_pct) <= NEWTON_SNAP * span]
        if near:
            step = min(near, key=lambda s: abs(100.0 * s.sigma - vol_pct))
            return self._describe(
                step.sigma, step.model_price, step.vega, f"Iteration {step.iteration}"
            )
        sigma = float(np.clip(vol_pct / 100.0, self.vols[0], self.vols[-1]))
        i = self.inputs
        args = (i.spot, i.strike, i.years, i.rate, i.dividend_yield, sigma, i.is_call)
        return self._describe(
            sigma,
            float(bs_price(*args)),
            float(bs_greeks(*args).vega),
            "Newton from this guess",
        )

    def _describe(self, sigma: float, price: float, vega: float, title: str) -> HoverInfo:
        error = price - self.market_price
        if abs(error) <= self.inputs.tol:
            nxt = "converged"
        elif np.isfinite(vega) and vega > MIN_USABLE_VEGA:
            nxt = f"{100 * (sigma - error / vega):.4f} %"
        else:
            nxt = "flat curve: no step"
        rows = (
            ("Volatility", f"{100 * sigma:.4f} %"),
            ("Model price", f"{price:.6f}"),
            ("Market price", f"{self.market_price:.6f}"),
            ("Model - market", f"{error:+.6f}"),
            ("Vega (slope)", f"{vega:.4f}"),
            ("Newton goes to", nxt),
        )
        return HoverInfo(100.0 * sigma, title, rows, ((100.0 * sigma, price),))


def trace_newton(inputs: NewtonInputs, *, curve_points: int = 200) -> NewtonTrace:
    """Run textbook Newton-Raphson on the price implied by ``market_vol`` and record each step."""
    t = inputs.years
    market_price = float(
        bs_price(
            inputs.spot,
            inputs.strike,
            t,
            inputs.rate,
            inputs.dividend_yield,
            inputs.market_vol,
            inputs.is_call,
        )
    )
    upper = max(0.8, 2.0 * inputs.market_vol, 1.5 * inputs.start_vol)
    vols = np.linspace(0.01, upper, curve_points)
    prices = bs_price(
        inputs.spot, inputs.strike, t, inputs.rate, inputs.dividend_yield, vols, inputs.is_call
    )
    try:
        steps = newton_iterations(
            market_price,
            inputs.spot,
            inputs.strike,
            t,
            inputs.rate,
            inputs.dividend_yield,
            inputs.is_call,
            sigma0=inputs.start_vol,
            tol=inputs.tol,
        )
    except ConvergenceError as exc:
        return NewtonTrace(vols, prices, market_price, (), False, str(exc), inputs)
    last = steps[-1]
    message = f"converged to {100 * last.sigma:.4f}% in {len(steps) - 1} iterations"
    return NewtonTrace(vols, prices, market_price, steps, True, message, inputs)
