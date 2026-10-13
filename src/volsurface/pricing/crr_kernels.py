"""Numba kernels for the Cox-Ross-Rubinstein binomial tree.

The tree follows the construction used by the GEX notebook this project re-implements (and by
QuantLib's ``BinomialVanillaEngine("crr")``): log-spaced nodes ``S * exp((2j - n) * dx)`` with
``dx = sigma * sqrt(dt)`` and up-probability ``0.5 + 0.5 * (r - q - sigma^2 / 2) * dt / dx``.
Gamma is a finite difference of the three node values at step 2.

Everything here is plain scalar/array code in ``nopython`` mode and releases the GIL, so callers
may run batches from worker threads. Public, validated wrappers live in
:mod:`volsurface.pricing.american`.
"""

import math

import numpy as np
from numba import njit, prange

SIGMA_LO = 0.01
SIGMA_HI = 5.0


@njit(cache=True, nogil=True)
def crr_price_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    sign: float,
    steps: int,
    american: bool,
) -> tuple[float, float]:
    """Option value and gamma on an n-step CRR tree; ``(nan, nan)`` if the tree is invalid.

    ``sign`` is +1 for a call and -1 for a put. With ``T <= 0`` or ``sigma <= 0`` the option is
    worth its intrinsic value and gamma is ``nan``.
    """
    if T <= 0.0 or sigma <= 0.0:
        return max(sign * (S - K), 0.0), np.nan
    dt = T / steps
    dx = sigma * math.sqrt(dt)
    pu = 0.5 + 0.5 * (r - q - 0.5 * sigma * sigma) * dt / dx
    if pu < 0.0 or pu > 1.0:
        return np.nan, np.nan
    pd = 1.0 - pu
    disc = math.exp(-r * dt)
    up2 = math.exp(2.0 * dx)

    work = np.empty(steps + 1)
    s = S * math.exp(-steps * dx)
    for j in range(steps + 1):
        work[j] = max(sign * (s - K), 0.0)
        s *= up2

    v0 = 0.0
    v1 = 0.0
    v2 = 0.0
    for step in range(steps - 1, -1, -1):
        s = S * math.exp(-step * dx)
        for j in range(step + 1):
            held = disc * (pu * work[j + 1] + pd * work[j])
            if american:
                work[j] = max(held, sign * (s - K), 0.0)
            else:
                work[j] = held
            s *= up2
        if step == 2:
            v0 = work[0]
            v1 = work[1]
            v2 = work[2]

    s_down = S * math.exp(-2.0 * dx)
    s_up = S * math.exp(2.0 * dx)
    gamma = ((v2 - v1) / (s_up - S) - (v1 - v0) / (S - s_down)) / (0.5 * (s_up - s_down))
    return work[0], gamma


@njit(cache=True, nogil=True)
def crr_iv_brent(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sign: float,
    steps: int,
    american: bool,
    sigma_tol: float,
    max_iter: int,
) -> float:
    """Implied volatility on the tree by Brent's method (bracketed, with bisection fallback).

    The bracket is ``[SIGMA_LO, SIGMA_HI]``; ``nan`` if the price lies outside what the tree can
    produce for volatilities in that range, or if the budget runs out. It finds the same root as
    bisection (the sign change of ``tree(sigma) - price``) in roughly a third of the evaluations.
    """
    a = SIGMA_LO
    b = SIGMA_HI
    fa = crr_price_gamma(S, K, T, r, q, a, sign, steps, american)[0] - price
    fb = crr_price_gamma(S, K, T, r, q, b, sign, steps, american)[0] - price
    if not (np.isfinite(fa) and np.isfinite(fb)) or fa >= 0.0 or fb <= 0.0:
        return np.nan
    c = b
    fc = fb
    d = b - a
    e = d
    for _ in range(max_iter):
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c = a
            fc = fa
            d = b - a
            e = d
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        tol1 = 2.0 * 2.220446049250313e-16 * abs(b) + 0.5 * sigma_tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                qq = 1.0 - s
            else:
                qq = fa / fc
                rr = fb / fc
                p = s * (2.0 * xm * qq * (qq - rr) - (b - a) * (rr - 1.0))
                qq = (qq - 1.0) * (rr - 1.0) * (s - 1.0)
            if p > 0.0:
                qq = -qq
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * qq - abs(tol1 * qq), abs(e * qq)):
                e = d
                d = p / qq
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a = b
        fa = fb
        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm > 0.0 else -tol1
        fb = crr_price_gamma(S, K, T, r, q, b, sign, steps, american)[0] - price
        if not np.isfinite(fb):
            return np.nan
    return np.nan


@njit(cache=True, nogil=True)
def crr_iv_bisection(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sign: float,
    steps: int,
    american: bool,
    iterations: int,
) -> float:
    """The notebook's solver: bisection on ``[SIGMA_LO, SIGMA_HI]`` for a fixed iteration count."""
    lo = SIGMA_LO
    hi = SIGMA_HI
    p_lo = crr_price_gamma(S, K, T, r, q, lo, sign, steps, american)[0]
    p_hi = crr_price_gamma(S, K, T, r, q, hi, sign, steps, american)[0]
    if not (np.isfinite(p_lo) and np.isfinite(p_hi)) or price <= p_lo or price >= p_hi:
        return np.nan
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if crr_price_gamma(S, K, T, r, q, mid, sign, steps, american)[0] > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@njit(parallel=True, cache=True)
def crr_batch(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    sign: np.ndarray,
    steps: int,
    american: bool,
    price_out: np.ndarray,
    gamma_out: np.ndarray,
) -> None:
    for i in prange(S.shape[0]):
        price_out[i], gamma_out[i] = crr_price_gamma(
            S[i], K[i], T[i], r[i], q[i], sigma[i], sign[i], steps, american
        )


@njit(parallel=True, cache=True)
def iv_batch(
    price: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sign: np.ndarray,
    steps: int,
    american: bool,
    sigma_tol: float,
    max_iter: int,
    bisection: bool,
    out: np.ndarray,
) -> None:
    for i in prange(S.shape[0]):
        if bisection:
            out[i] = crr_iv_bisection(
                price[i], S[i], K[i], T[i], r[i], q[i], sign[i], steps, american, max_iter
            )
        else:
            out[i] = crr_iv_brent(
                price[i],
                S[i],
                K[i],
                T[i],
                r[i],
                q[i],
                sign[i],
                steps,
                american,
                sigma_tol,
                max_iter,
            )


@njit(parallel=True, cache=True)
def gamma_grid_batch(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    sign: np.ndarray,
    levels: np.ndarray,
    steps: int,
    american: bool,
    out: np.ndarray,
) -> None:
    """Gamma of every option at every spot level ``S[i] * levels[j]`` (nan where sigma is nan)."""
    n = S.shape[0]
    m = levels.shape[0]
    for i in prange(n):
        if np.isnan(sigma[i]):
            for j in range(m):
                out[i, j] = np.nan
            continue
        for j in range(m):
            out[i, j] = crr_price_gamma(
                S[i] * levels[j], K[i], T[i], r[i], q[i], sigma[i], sign[i], steps, american
            )[1]
