"""Stochastic models for Monte Carlo pricing (the path-generation Strategy).

Each model turns a payoff, a horizon and a number of paths into one *independent* payoff sample
per path. With antithetic variates a sample is the average of a path and its mirror image, so the
samples stay independent and the standard error computed from them is valid.
"""

import enum
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import FloatArray
from volsurface.pricing._parallel import PARALLEL_LOCK
from volsurface.pricing.montecarlo import kernels


class PayoffKind(enum.IntEnum):
    CALL = kernels.CALL
    PUT = kernels.PUT
    BEST_OF_CALL = kernels.BEST_OF_CALL
    WORST_OF_CALL = kernels.WORST_OF_CALL
    ASIAN_CALL = kernels.ASIAN_CALL

    @property
    def needs_two_assets(self) -> bool:
        return self in {PayoffKind.BEST_OF_CALL, PayoffKind.WORST_OF_CALL}


@dataclass(frozen=True, slots=True)
class Payoff:
    kind: PayoffKind
    strike: float

    def __post_init__(self) -> None:
        if self.strike <= 0:
            raise InvalidInputError("strike must be positive")


class PathModel(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def rate(self) -> float: ...

    def supports(self, kind: PayoffKind) -> bool: ...

    def sample(
        self, payoff: Payoff, t: float, n_paths: int, *, seed: int, antithetic: bool, steps: int
    ) -> FloatArray:
        """Undiscounted payoff samples, one per path."""
        ...


def _check_run(payoff: Payoff, t: float, n_paths: int, steps: int, model: PathModel) -> None:
    if t <= 0 or n_paths < 1 or steps < 1:
        raise InvalidInputError("t must be positive and n_paths, steps at least 1")
    if not model.supports(payoff.kind):
        raise InvalidInputError(f"{model.name} cannot price a {payoff.kind.name} payoff")


@dataclass(frozen=True, slots=True)
class GbmModel:
    """Geometric Brownian motion: ``dS = (r - q) S dt + sigma S dW``."""

    s0: float
    r: float
    q: float
    sigma: float

    name = "GBM"

    def __post_init__(self) -> None:
        if self.s0 <= 0 or self.sigma < 0:
            raise InvalidInputError("s0 must be positive and sigma non-negative")

    @property
    def rate(self) -> float:
        return self.r

    def supports(self, kind: PayoffKind) -> bool:
        return not kind.needs_two_assets

    def sample(
        self, payoff: Payoff, t: float, n_paths: int, *, seed: int, antithetic: bool, steps: int
    ) -> FloatArray:
        _check_run(payoff, t, n_paths, steps, self)
        out = np.empty(n_paths)
        with PARALLEL_LOCK:
            kernels.simulate_gbm(
                n_paths,
                self.s0,
                self.r,
                self.q,
                self.sigma,
                t,
                int(payoff.kind),
                payoff.strike,
                steps,
                seed,
                antithetic,
                out,
            )
        return out


@dataclass(frozen=True, slots=True)
class HestonModel:
    """Heston stochastic volatility, simulated with full-truncation Euler on the log-price."""

    s0: float
    v0: float
    r: float
    q: float
    kappa: float
    theta: float
    xi: float
    rho: float

    name = "Heston"

    def __post_init__(self) -> None:
        if self.s0 <= 0 or self.v0 < 0 or self.theta < 0 or self.kappa < 0 or self.xi < 0:
            raise InvalidInputError("Heston parameters must be non-negative (s0 positive)")
        if not -1.0 <= self.rho <= 1.0:
            raise InvalidInputError("rho must be within [-1, 1]")

    @property
    def rate(self) -> float:
        return self.r

    @property
    def feller_satisfied(self) -> bool:
        """``2 kappa theta >= xi^2``: the variance process cannot hit zero in continuous time."""
        return 2.0 * self.kappa * self.theta >= self.xi * self.xi

    def supports(self, kind: PayoffKind) -> bool:
        return not kind.needs_two_assets

    def sample(
        self, payoff: Payoff, t: float, n_paths: int, *, seed: int, antithetic: bool, steps: int
    ) -> FloatArray:
        _check_run(payoff, t, n_paths, steps, self)
        out = np.empty(n_paths)
        with PARALLEL_LOCK:
            kernels.simulate_heston(
                n_paths,
                self.s0,
                self.v0,
                self.r,
                self.q,
                self.kappa,
                self.theta,
                self.xi,
                self.rho,
                t,
                steps,
                int(payoff.kind),
                payoff.strike,
                seed,
                antithetic,
                out,
            )
        return out


@dataclass(frozen=True, slots=True)
class TwoHestonModel:
    """Two assets with identical Heston dynamics whose price shocks are correlated."""

    s0: float
    v0: float
    r: float
    q: float
    kappa: float
    theta: float
    xi: float
    rho: float
    rho_assets: float = 0.0

    name = "Two-asset Heston"

    def __post_init__(self) -> None:
        HestonModel(self.s0, self.v0, self.r, self.q, self.kappa, self.theta, self.xi, self.rho)
        if not -1.0 <= self.rho_assets <= 1.0:
            raise InvalidInputError("rho_assets must be within [-1, 1]")

    @property
    def rate(self) -> float:
        return self.r

    @property
    def feller_satisfied(self) -> bool:
        return 2.0 * self.kappa * self.theta >= self.xi * self.xi

    def supports(self, kind: PayoffKind) -> bool:
        return True

    def sample(
        self, payoff: Payoff, t: float, n_paths: int, *, seed: int, antithetic: bool, steps: int
    ) -> FloatArray:
        _check_run(payoff, t, n_paths, steps, self)
        out = np.empty(n_paths)
        with PARALLEL_LOCK:
            kernels.simulate_two_heston(
                n_paths,
                self.s0,
                self.v0,
                self.r,
                self.q,
                self.kappa,
                self.theta,
                self.xi,
                self.rho,
                self.rho_assets,
                t,
                steps,
                int(payoff.kind),
                payoff.strike,
                seed,
                antithetic,
                out,
            )
        return out
