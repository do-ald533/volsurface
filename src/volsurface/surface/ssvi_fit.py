"""Fit one slice in the SSVI family, which is butterfly-free by construction.

For a single maturity, SSVI is raw SVI with the five parameters tied to three::

    a = theta * (1 - rho^2) / 2,  b = theta * phi / 2
    m = -rho / phi,  sigma = sqrt(1 - rho^2) / phi

(``theta`` is the at-the-money total variance, ``rho`` the skew and ``phi`` the curvature). The
sufficient conditions of Gatheral and Jacquier, arXiv:1204.0646 Theorem 4.2,
``theta * phi * (1 + |rho|) < 4`` and ``theta * phi^2 * (1 + |rho|) <= 4``, exclude butterfly
arbitrage; here they become a hard bound on ``phi`` by writing ``phi = u * phi_max(theta, rho)``
with ``u`` in ``(0, 1]``. The family is less flexible than raw SVI but is stable when the data
cover only one wing, which is why it serves as the fallback when a raw fit cannot be repaired.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from volsurface.domain.errors import PricingError
from volsurface.domain.types import FloatArray
from volsurface.surface.svi import SviParams

_RHO_MAX = 0.98
_U_MIN = 0.02
_SAFETY = 0.999


@dataclass(frozen=True, slots=True)
class SsviSliceFit:
    params: SviParams
    theta: float
    rho: float
    phi: float
    nfev: int


def phi_max(theta: float, rho: float) -> float:
    """Largest ``phi`` satisfying both butterfly conditions for this ``theta`` and ``rho``."""
    scale = theta * (1.0 + abs(rho))
    return _SAFETY * min(4.0 / scale, math.sqrt(4.0 / scale))


def to_raw_svi(theta: float, rho: float, phi: float) -> SviParams:
    """The raw SVI slice equal to the SSVI slice ``(theta, rho, phi)``."""
    root = math.sqrt(1.0 - rho * rho)
    return SviParams(
        a=theta * (1.0 - rho * rho) / 2.0,
        b=theta * phi / 2.0,
        rho=rho,
        m=-rho / phi,
        sigma=root / phi,
    )


def ssvi_total_variance(k: FloatArray, theta: float, rho: float, phi: float) -> FloatArray:
    inner = np.sqrt((phi * k + rho) ** 2 + 1.0 - rho * rho)
    return np.asarray(0.5 * theta * (1.0 + rho * phi * k + inner))


def fit_ssvi_slice(
    k: FloatArray,
    w: FloatArray,
    sqrt_weight: FloatArray,
    *,
    lower_slice: Callable[[FloatArray], FloatArray] | None = None,
    penalty_scale: float = 50.0,
) -> SsviSliceFit:
    """Weighted least squares over ``(theta, rho, u)``; optional calendar penalty.

    ``k``, ``w`` and ``sqrt_weight`` are the cleaned arrays held by the caller's fitting problem.
    """
    lower = np.array([max(float(w.min()) * 0.25, 1e-8), -_RHO_MAX, _U_MIN])
    upper = np.array([max(float(w.max()) * 2.0, 1e-6), _RHO_MAX, 1.0])
    grid = np.linspace(float(k.min()) - 0.5, float(k.max()) + 0.5, 81)

    def unpack(x: FloatArray) -> tuple[float, float, float]:
        theta, rho, u = float(x[0]), float(x[1]), float(x[2])
        return theta, rho, u * phi_max(theta, rho)

    def residuals(x: FloatArray) -> FloatArray:
        theta, rho, phi = unpack(x)
        res = sqrt_weight * (ssvi_total_variance(k, theta, rho, phi) - w)
        if lower_slice is None:
            return res
        gap = np.maximum(lower_slice(grid) - ssvi_total_variance(grid, theta, rho, phi), 0.0)
        return np.concatenate([res, penalty_scale * gap / math.sqrt(gap.size)])

    atm = float(w[int(np.argmin(np.abs(k)))])
    best = None
    nfev = 0
    for rho0, u0 in ((-0.5, 0.3), (0.0, 0.6), (-0.8, 0.15)):
        x0 = np.clip(np.array([atm, rho0, u0]), lower + 1e-9, upper - 1e-9)
        try:
            result = least_squares(
                residuals,
                x0,
                bounds=(lower, upper),
                method="trf",
                loss="soft_l1",
                f_scale=max(float(w.std()), 1e-4),
                x_scale="jac",
                max_nfev=120,
            )
        except (ValueError, FloatingPointError):
            continue
        nfev += int(result.nfev)
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        raise PricingError("SSVI slice fit failed from every starting point")
    theta, rho, phi = unpack(best.x)
    return SsviSliceFit(to_raw_svi(theta, rho, phi), theta, rho, phi, nfev)
