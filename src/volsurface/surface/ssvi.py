"""Surface SVI (SSVI) of Gatheral and Jacquier, used here as an analytic ground truth.

The whole surface is parametrised by the at-the-money total variance term structure ``theta_t``
and a function ``phi``::

    w(k, theta_t) = (theta_t / 2) * (1 + rho * phi(theta_t) * k
                                      + sqrt((phi(theta_t) * k + rho)^2 + 1 - rho^2))

Static arbitrage is excluded by the sufficient conditions of arXiv:1204.0646 (Theorems 4.1 and 4.2),
checked numerically by :meth:`SsviSurface.conditions`. The synthetic market-data provider samples
this surface, so a fitted SVI surface can be compared with a known truth.

``phi`` is the power-law family ``eta * theta^-gamma * (1 + theta)^(gamma - 1)``; whether a given
``(eta, gamma, rho)`` is arbitrage-free is decided by the checks, not assumed.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import FloatArray


@dataclass(frozen=True, slots=True)
class SsviConditions:
    """Outcome of the numerical check of the Gatheral-Jacquier sufficient conditions."""

    theta_non_decreasing: bool
    calendar_derivative_ok: bool
    butterfly_strict_ok: bool
    butterfly_tight_ok: bool

    @property
    def calendar_free(self) -> bool:
        return self.theta_non_decreasing and self.calendar_derivative_ok

    @property
    def butterfly_free(self) -> bool:
        return self.butterfly_strict_ok and self.butterfly_tight_ok

    @property
    def arbitrage_free(self) -> bool:
        return self.calendar_free and self.butterfly_free


@dataclass(frozen=True, slots=True)
class SsviSurface:
    """SSVI with a piecewise-linear ATM total variance term structure."""

    maturities: FloatArray
    atm_total_variance: FloatArray
    rho: float
    eta: float
    gamma: float

    def __post_init__(self) -> None:
        t = np.asarray(self.maturities, dtype=np.float64)
        theta = np.asarray(self.atm_total_variance, dtype=np.float64)
        if t.ndim != 1 or t.shape != theta.shape or t.size < 1:
            raise InvalidInputError("maturities and atm_total_variance must be equal-length 1-D")
        if (t <= 0).any() or (np.diff(t) <= 0).any():
            raise InvalidInputError("maturities must be positive and strictly increasing")
        if (theta <= 0).any():
            raise InvalidInputError("atm_total_variance must be positive")
        if not -1.0 < self.rho < 1.0:
            raise InvalidInputError("rho must be in (-1, 1)")
        if self.eta <= 0 or not 0.0 <= self.gamma <= 1.0:
            raise InvalidInputError("eta must be positive and gamma in [0, 1]")
        object.__setattr__(self, "maturities", t)
        object.__setattr__(self, "atm_total_variance", theta)

    def theta(self, t: ArrayLike) -> FloatArray:
        """ATM total variance at maturity ``t`` (linear in ``t``, zero at ``t = 0``)."""
        tt = np.asarray(t, dtype=np.float64)
        nodes_t = np.concatenate(([0.0], self.maturities))
        nodes_theta = np.concatenate(([0.0], self.atm_total_variance))
        inside = np.interp(tt, nodes_t, nodes_theta)
        beyond = self.atm_total_variance[-1] * tt / self.maturities[-1]
        return np.asarray(np.where(tt > self.maturities[-1], beyond, inside))

    def phi(self, theta: ArrayLike) -> FloatArray:
        th = np.asarray(theta, dtype=np.float64)
        return np.asarray(self.eta * th ** (-self.gamma) * (1.0 + th) ** (self.gamma - 1.0))

    def total_variance(self, k: ArrayLike, t: ArrayLike) -> FloatArray:
        """``w(k, t)`` for log-moneyness ``k`` and maturity ``t`` (broadcast together)."""
        kk = np.asarray(k, dtype=np.float64)
        th = self.theta(t)
        ph = self.phi(th)
        inner = np.sqrt((ph * kk + self.rho) ** 2 + 1.0 - self.rho**2)
        return np.asarray(0.5 * th * (1.0 + self.rho * ph * kk + inner))

    def implied_vol(self, k: ArrayLike, t: ArrayLike) -> FloatArray:
        tt = np.asarray(t, dtype=np.float64)
        return np.sqrt(self.total_variance(k, tt) / tt)

    def conditions(self) -> SsviConditions:
        """Check Theorems 4.1 and 4.2 of arXiv:1204.0646 over the surface's theta range."""
        theta = np.linspace(1e-4, float(self.atm_total_variance.max()) * 1.5, 4000)
        phi = self.phi(theta)
        abs_rho = abs(self.rho)

        slope = np.gradient(theta * phi, theta)
        upper = (
            np.full_like(theta, np.inf)
            if self.rho == 0.0
            else (1.0 + np.sqrt(1.0 - self.rho**2)) / self.rho**2 * phi
        )
        derivative_ok = bool(np.all(slope >= -1e-9) and np.all(slope <= upper + 1e-9))

        return SsviConditions(
            theta_non_decreasing=bool(np.all(np.diff(self.atm_total_variance) >= 0)),
            calendar_derivative_ok=derivative_ok,
            butterfly_strict_ok=bool(np.all(theta * phi * (1.0 + abs_rho) < 4.0)),
            butterfly_tight_ok=bool(np.all(theta * phi**2 * (1.0 + abs_rho) <= 4.0)),
        )
