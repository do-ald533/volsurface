"""Raw SVI slices (Gatheral and Jacquier, arXiv:1204.0646).

A slice is the total implied variance ``w(k) = sigma_imp(k)^2 * T`` as a function of log-moneyness
``k = ln(K / F)`` at one expiry::

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

This module is the model: parameters, evaluation, derivatives and the static no-arbitrage
checks (the butterfly condition ``g(k) >= 0`` of Lemma 2.2 and Roger Lee's wing bound
``b * (1 + |rho|) <= 2``). Calibration lives in :mod:`volsurface.surface.svi_fit`.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import FloatArray
from volsurface.surface import svi_kernels

LEE_SLOPE_BOUND = 2.0
BUTTERFLY_TOLERANCE = -1e-8
CALENDAR_TOLERANCE = 1e-9
CHECK_PAD = 0.5

type SliceFunction = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True, slots=True)
class SviParams:
    """Raw SVI parameters ``(a, b, rho, m, sigma)`` with the standard validity constraints."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.a, self.b, self.rho, self.m, self.sigma)):
            raise InvalidInputError("SVI parameters must be finite")
        if self.b < 0:
            raise InvalidInputError(f"b must be non-negative, got {self.b}")
        if not -1.0 < self.rho < 1.0:
            raise InvalidInputError(f"rho must be in (-1, 1), got {self.rho}")
        if self.sigma <= 0:
            raise InvalidInputError(f"sigma must be positive, got {self.sigma}")
        if self.min_variance < -1e-12:
            raise InvalidInputError(
                "a + b*sigma*sqrt(1 - rho^2) must be non-negative so that w(k) >= 0"
            )

    @property
    def min_variance(self) -> float:
        """Smallest total variance, attained at ``k = m - rho*sigma/sqrt(1-rho^2)``."""
        return self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho * self.rho)

    @property
    def left_slope(self) -> float:
        """Asymptotic slope of ``w`` as ``k -> -inf`` (``b * (1 - rho)``)."""
        return self.b * (1.0 - self.rho)

    @property
    def right_slope(self) -> float:
        """Asymptotic slope of ``w`` as ``k -> +inf`` (``b * (1 + rho)``)."""
        return self.b * (1.0 + self.rho)

    @classmethod
    def from_minimum_variance(
        cls, s: float, b: float, rho: float, m: float, sigma: float
    ) -> "SviParams":
        """Build from the fitting parametrisation ``(s, b, rho, m, sigma)``."""
        return cls(s - b * sigma * math.sqrt(1.0 - rho * rho), b, rho, m, sigma)

    def as_fit_vector(self) -> FloatArray:
        return np.array([self.min_variance, self.b, self.rho, self.m, self.sigma])


def total_variance(k: ArrayLike, p: SviParams) -> FloatArray:
    """``w(k)``."""
    x = np.asarray(k, dtype=np.float64) - p.m
    return np.asarray(p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma * p.sigma)))


def first_derivative(k: ArrayLike, p: SviParams) -> FloatArray:
    """``w'(k)``."""
    x = np.asarray(k, dtype=np.float64) - p.m
    return np.asarray(p.b * (p.rho + x / np.sqrt(x * x + p.sigma * p.sigma)))


def second_derivative(k: ArrayLike, p: SviParams) -> FloatArray:
    """``w''(k)``."""
    x = np.asarray(k, dtype=np.float64) - p.m
    return np.asarray(p.b * p.sigma * p.sigma / (x * x + p.sigma * p.sigma) ** 1.5)


def implied_vol(k: ArrayLike, t: float, p: SviParams) -> FloatArray:
    """Black-Scholes implied volatility ``sqrt(w(k) / t)``."""
    if t <= 0:
        raise InvalidInputError("t must be positive")
    return np.sqrt(np.maximum(total_variance(k, p), 0.0) / t)


def butterfly_density_factor(k: ArrayLike, p: SviParams) -> FloatArray:
    """``g(k)``; the slice is free of butterfly arbitrage iff ``g >= 0`` everywhere (Lemma 2.2).

    ``-inf`` where ``w(k) <= 0`` (the risk-neutral density is not defined there).
    """
    grid = np.ascontiguousarray(np.asarray(k, dtype=np.float64).reshape(-1))
    g = cast(
        "FloatArray",
        svi_kernels.butterfly_factor_grid(grid, p.a, p.b, p.rho, p.m, p.sigma),
    )
    return g.reshape(np.shape(k))


def is_lee_compliant(p: SviParams) -> bool:
    """Both wing slopes are at most 2 (Roger Lee's moment formula)."""
    return max(p.left_slope, p.right_slope) <= LEE_SLOPE_BOUND + 1e-9


def check_slice(
    p: SviParams, k_min: float, k_max: float, *, pad: float = CHECK_PAD
) -> tuple[float, bool]:
    """Minimum of ``g`` on ``[k_min - pad, k_max + pad]`` and whether it is non-negative.

    Beyond the padded range the tails are governed by Lee's wing bound, checked separately.
    """
    grid = np.linspace(k_min - pad, k_max + pad, 801)
    min_g = float(np.min(butterfly_density_factor(grid, p)))
    return min_g, min_g >= BUTTERFLY_TOLERANCE
