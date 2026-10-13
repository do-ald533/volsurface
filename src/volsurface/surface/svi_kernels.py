"""Numba kernels for SVI fitting.

``least_squares`` calls the residual and Jacobian hundreds of times per slice and each call works
on a few dozen points, so interpreter overhead dominates the NumPy version. Compiled loops cut it
to a few microseconds, and the analytic Jacobian removes the five extra residual evaluations per
iteration that finite differences would need.

The optimiser's parametrisation is ``x = (s, b, rho, m, sigma)`` with ``s`` the minimum total
variance, so ``a = s - b * sigma * sqrt(1 - rho^2)``.
"""

import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def residuals_and_jacobian(
    x: np.ndarray, k: np.ndarray, w: np.ndarray, sqrt_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted residuals ``sqrt_weight * (w_model - w)`` and their Jacobian in ``x``."""
    n = k.shape[0]
    s = x[0]
    b = x[1]
    rho = x[2]
    m = x[3]
    sigma = x[4]
    c = math.sqrt(max(1.0 - rho * rho, 1e-300))
    res = np.empty(n)
    jac = np.empty((n, 5))
    for i in range(n):
        xi = k[i] - m
        root = math.sqrt(xi * xi + sigma * sigma)
        model = s - b * sigma * c + b * rho * xi + b * root
        sw = sqrt_weight[i]
        res[i] = sw * (model - w[i])
        jac[i, 0] = sw
        jac[i, 1] = sw * (-sigma * c + rho * xi + root)
        jac[i, 2] = sw * b * (xi + sigma * rho / c)
        jac[i, 3] = sw * b * (-rho - xi / root)
        jac[i, 4] = sw * b * (-c + sigma / root)
    return res, jac


@njit(cache=True, nogil=True)
def total_variance_grid(
    k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float
) -> np.ndarray:
    out = np.empty(k.shape[0])
    for i in range(k.shape[0]):
        xi = k[i] - m
        out[i] = a + b * (rho * xi + math.sqrt(xi * xi + sigma * sigma))
    return out


@njit(cache=True, nogil=True)
def butterfly_factor_grid(
    k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float
) -> np.ndarray:
    """``g(k)`` of Gatheral and Jacquier (Lemma 2.2); ``-inf`` where ``w(k) <= 0``."""
    out = np.empty(k.shape[0])
    sig2 = sigma * sigma
    for i in range(k.shape[0]):
        ki = k[i]
        xi = ki - m
        root = math.sqrt(xi * xi + sig2)
        w = a + b * (rho * xi + root)
        if w <= 0.0:
            out[i] = -np.inf
            continue
        wp = b * (rho + xi / root)
        wpp = b * sig2 / (root * root * root)
        lead = 1.0 - ki * wp / (2.0 * w)
        out[i] = lead * lead - wp * wp / 4.0 * (1.0 / w + 0.25) + wpp / 2.0
    return out
