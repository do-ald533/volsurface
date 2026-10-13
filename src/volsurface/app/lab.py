"""The calculations behind the Pricing Lab tabs, free of any Qt.

Three small use cases, each taking a frozen input record and returning a frozen result:
vanilla pricing (Black-Scholes next to the American tree), Monte Carlo, and a step-by-step view of
Newton-Raphson implied volatility.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from volsurface.domain.errors import ConvergenceError, InvalidInputError
from volsurface.domain.types import FloatArray
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

DAYS_PER_YEAR = 365.0
ModelName = Literal["gbm", "heston", "rainbow"]


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
        return NewtonTrace(vols, prices, market_price, (), False, str(exc))
    last = steps[-1]
    message = f"converged to {100 * last.sigma:.4f}% in {len(steps) - 1} iterations"
    return NewtonTrace(vols, prices, market_price, steps, True, message)
