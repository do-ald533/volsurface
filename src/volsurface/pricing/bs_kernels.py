"""Numba kernels for Black-Scholes-Merton: scalar routines and parallel batch drivers.

The scalar routines mirror :func:`volsurface.pricing.black_scholes.price_raw` and the safeguarded
Newton solver in :func:`volsurface.pricing.implied_vol.implied_vol_bs_reference`, which remain the
readable specification the tests compare these kernels against. Unlike the NumPy solver, which
iterates the whole batch until the slowest option converges, each option here stops as soon as it
converges, which is where most of the speed-up comes from.
"""

import math

import numpy as np
from numba import njit, prange

SIGMA_MIN = 1e-6
SIGMA_MAX = 10.0
MIN_VEGA = 1e-14
MIN_TOTAL_VOL = 1e-12
INV_SQRT_2 = 0.7071067811865476
INV_SQRT_2PI = 0.3989422804014327


@njit(cache=True, nogil=True)
def ndtr(x: float) -> float:
    return 0.5 * math.erfc(-x * INV_SQRT_2)


@njit(cache=True, nogil=True)
def bs_price_scalar(
    S: float, K: float, T: float, r: float, q: float, sigma: float, sign: float
) -> float:
    disc_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)
    total_vol = sigma * math.sqrt(T)
    if total_vol < MIN_TOTAL_VOL:
        return max(sign * (disc_s - disc_k), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / total_vol
    d2 = d1 - total_vol
    return sign * (disc_s * ndtr(sign * d1) - disc_k * ndtr(sign * d2))


@njit(cache=True, nogil=True)
def bs_vega_scalar(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    total_vol = sigma * math.sqrt(T)
    if total_vol < MIN_TOTAL_VOL:
        return np.nan
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / total_vol
    return S * math.exp(-q * T) * INV_SQRT_2PI * math.exp(-0.5 * d1 * d1) * math.sqrt(T)


@njit(cache=True, nogil=True)
def bs_iv_scalar(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sign: float,
    tol: float,
    max_iter: int,
) -> float:
    if not (math.isfinite(price) and math.isfinite(S) and math.isfinite(K) and math.isfinite(T)):
        return np.nan
    if S <= 0.0 or K <= 0.0 or T <= 0.0:
        return np.nan
    disc_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)
    lower = max(sign * (disc_s - disc_k), 0.0)
    upper = disc_s if sign > 0.0 else disc_k
    if not (price > lower and price < upper):
        return np.nan
    lo = SIGMA_MIN
    hi = SIGMA_MAX
    if bs_price_scalar(S, K, T, r, q, lo, sign) > price:
        return np.nan
    if bs_price_scalar(S, K, T, r, q, hi, sign) < price:
        return np.nan
    forward = S * math.exp((r - q) * T)
    sigma = math.sqrt(2.0 * abs(math.log(forward / K)) / T)
    sigma = min(max(sigma, 0.15), 2.0)
    for _ in range(max_iter):
        diff = bs_price_scalar(S, K, T, r, q, sigma, sign) - price
        if abs(diff) <= tol or hi - lo <= 1e-14:
            return sigma
        if diff < 0.0:
            lo = sigma
        else:
            hi = sigma
        vega = bs_vega_scalar(S, K, T, r, q, sigma)
        if vega > MIN_VEGA:
            newton = sigma - diff / vega
            if newton > lo and newton < hi:
                sigma = newton
                continue
        sigma = 0.5 * (lo + hi)
    return np.nan


@njit(parallel=True, cache=True)
def bs_iv_batch(
    price: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sign: np.ndarray,
    tol: float,
    max_iter: int,
    out: np.ndarray,
) -> None:
    for i in prange(price.shape[0]):
        out[i] = bs_iv_scalar(price[i], S[i], K[i], T[i], r[i], q[i], sign[i], tol, max_iter)
