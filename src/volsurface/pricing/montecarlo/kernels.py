"""Numba kernels for Monte Carlo path simulation.

Random numbers come from a counter-based generator: every draw is a hash of ``(seed, path, draw
index)``. A path therefore sees the same numbers no matter which thread simulates it or how many
threads exist, which makes results bit-reproducible under ``prange``. The hash is SplitMix64
(Steele, Lea and Flood, 2014); normals are produced in pairs by the Box-Muller transform.

Payoff codes: 0 call, 1 put, 2 best-of call, 3 worst-of call, 4 arithmetic-average (Asian) call.
Models: 0 geometric Brownian motion (exact terminal sampling), 1 Heston with full-truncation Euler
on log-price, 2 two Heston assets with correlated price shocks.
"""

import math

import numpy as np
from numba import njit, prange

CALL = 0
PUT = 1
BEST_OF_CALL = 2
WORST_OF_CALL = 3
ASIAN_CALL = 4

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S11 = np.uint64(11)
_S32 = np.uint64(32)
_INV_2_53 = 1.0 / 9007199254740992.0
_TWO_PI = 2.0 * math.pi


@njit(cache=True, nogil=True)
def splitmix64(x: np.uint64) -> np.uint64:
    z = x + _GOLDEN
    z = (z ^ (z >> _S30)) * _MIX1
    z = (z ^ (z >> _S27)) * _MIX2
    return z ^ (z >> _S31)


@njit(cache=True, nogil=True)
def uniform(seed: np.uint64, path: np.uint64, draw: np.uint64) -> float:
    """A number in the open interval (0, 1) determined by ``(seed, path, draw)``."""
    key = splitmix64(seed ^ splitmix64((path << _S32) + draw))
    return ((key >> _S11) + 0.5) * _INV_2_53


@njit(cache=True, nogil=True)
def normal_pair(seed: np.uint64, path: np.uint64, draw: np.uint64) -> tuple[float, float]:
    """Two independent standard normals for the draw slot ``draw`` (uses uniforms 2d and 2d+1)."""
    u1 = uniform(seed, path, np.uint64(2) * draw)
    u2 = uniform(seed, path, np.uint64(2) * draw + np.uint64(1))
    radius = math.sqrt(-2.0 * math.log(u1))
    return radius * math.cos(_TWO_PI * u2), radius * math.sin(_TWO_PI * u2)


@njit(cache=True, nogil=True)
def payoff_value(code: int, strike: float, s1: float, s2: float, average: float) -> float:
    if code == PUT:
        return max(strike - s1, 0.0)
    if code == BEST_OF_CALL:
        return max(max(s1, s2) - strike, 0.0)
    if code == WORST_OF_CALL:
        return max(min(s1, s2) - strike, 0.0)
    if code == ASIAN_CALL:
        return max(average - strike, 0.0)
    return max(s1 - strike, 0.0)


@njit(cache=True, nogil=True)
def gbm_sample(
    s0: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    code: int,
    strike: float,
    seed: np.uint64,
    path: np.uint64,
    sign: float,
    steps: int,
) -> float:
    """One GBM path. Terminal-only payoffs use exact sampling; Asian payoffs step through time."""
    drift = (r - q - 0.5 * sigma * sigma) * t
    if code != ASIAN_CALL:
        z, _ = normal_pair(seed, path, np.uint64(0))
        s_t = s0 * math.exp(drift + sigma * math.sqrt(t) * sign * z)
        return payoff_value(code, strike, s_t, s_t, s_t)
    dt = t / steps
    step_drift = (r - q - 0.5 * sigma * sigma) * dt
    step_vol = sigma * math.sqrt(dt)
    s = s0
    total = 0.0
    for j in range(steps):
        z, _ = normal_pair(seed, path, np.uint64(j))
        s *= math.exp(step_drift + step_vol * sign * z)
        total += s
    return payoff_value(code, strike, s, s, total / steps)


@njit(cache=True, nogil=True)
def heston_sample(
    s0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    steps: int,
    code: int,
    strike: float,
    seed: np.uint64,
    path: np.uint64,
    sign: float,
) -> float:
    """One Heston path (full-truncation Euler on log-price); payoff on that asset only."""
    dt = t / steps
    sqrt_dt = math.sqrt(dt)
    rho_c = math.sqrt(max(1.0 - rho * rho, 0.0))
    s = s0
    v = v0
    total = 0.0
    for j in range(steps):
        n1, n2 = normal_pair(seed, path, np.uint64(j))
        dw1 = sqrt_dt * sign * n1
        dw2 = sqrt_dt * sign * (rho * n1 + rho_c * n2)
        v_pos = max(v, 0.0)
        root = math.sqrt(v_pos)
        s *= math.exp((r - q - 0.5 * v_pos) * dt + root * dw1)
        v = max(v_pos + kappa * (theta - v_pos) * dt + xi * root * dw2, 0.0)
        total += s
    return payoff_value(code, strike, s, s, total / steps)


@njit(cache=True, nogil=True)
def two_heston_sample(
    s0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    rho_assets: float,
    t: float,
    steps: int,
    code: int,
    strike: float,
    seed: np.uint64,
    path: np.uint64,
    sign: float,
) -> float:
    """Two Heston assets, same parameters, price shocks correlated by ``rho_assets``."""
    dt = t / steps
    sqrt_dt = math.sqrt(dt)
    rho_c = math.sqrt(max(1.0 - rho * rho, 0.0))
    ra_c = math.sqrt(max(1.0 - rho_assets * rho_assets, 0.0))
    sa = s0
    sb = s0
    va = v0
    vb = v0
    total = 0.0
    for j in range(steps):
        a1, a2 = normal_pair(seed, path, np.uint64(2 * j))
        b1, b2 = normal_pair(seed, path, np.uint64(2 * j + 1))
        za = a1
        zb = rho_assets * a1 + ra_c * b1
        dwa = sqrt_dt * sign * za
        dwb = sqrt_dt * sign * zb
        dva = sqrt_dt * sign * (rho * za + rho_c * a2)
        dvb = sqrt_dt * sign * (rho * zb + rho_c * b2)
        va_pos = max(va, 0.0)
        vb_pos = max(vb, 0.0)
        ra = math.sqrt(va_pos)
        rb = math.sqrt(vb_pos)
        sa *= math.exp((r - q - 0.5 * va_pos) * dt + ra * dwa)
        sb *= math.exp((r - q - 0.5 * vb_pos) * dt + rb * dwb)
        va = max(va_pos + kappa * (theta - va_pos) * dt + xi * ra * dva, 0.0)
        vb = max(vb_pos + kappa * (theta - vb_pos) * dt + xi * rb * dvb, 0.0)
        total += sa
    return payoff_value(code, strike, sa, sb, total / steps)


@njit(parallel=True, cache=True)
def simulate_gbm(
    n: int,
    s0: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    code: int,
    strike: float,
    steps: int,
    seed: int,
    antithetic: bool,
    out: np.ndarray,
) -> None:
    """Fill ``out[i]`` with the (undiscounted) payoff sample of path ``i``."""
    base = np.uint64(seed)
    for i in prange(n):
        path = np.uint64(i)
        value = gbm_sample(s0, r, q, sigma, t, code, strike, base, path, 1.0, steps)
        if antithetic:
            value = 0.5 * (
                value + gbm_sample(s0, r, q, sigma, t, code, strike, base, path, -1.0, steps)
            )
        out[i] = value


@njit(parallel=True, cache=True)
def simulate_heston(
    n: int,
    s0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    steps: int,
    code: int,
    strike: float,
    seed: int,
    antithetic: bool,
    out: np.ndarray,
) -> None:
    base = np.uint64(seed)
    for i in prange(n):
        path = np.uint64(i)
        value = heston_sample(
            s0, v0, r, q, kappa, theta, xi, rho, t, steps, code, strike, base, path, 1.0
        )
        if antithetic:
            other = heston_sample(
                s0, v0, r, q, kappa, theta, xi, rho, t, steps, code, strike, base, path, -1.0
            )
            value = 0.5 * (value + other)
        out[i] = value


@njit(parallel=True, cache=True)
def simulate_two_heston(
    n: int,
    s0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    rho_assets: float,
    t: float,
    steps: int,
    code: int,
    strike: float,
    seed: int,
    antithetic: bool,
    out: np.ndarray,
) -> None:
    base = np.uint64(seed)
    for i in prange(n):
        path = np.uint64(i)
        value = two_heston_sample(
            s0, v0, r, q, kappa, theta, xi, rho, rho_assets, t, steps, code, strike, base, path, 1.0
        )
        if antithetic:
            other = two_heston_sample(
                s0,
                v0,
                r,
                q,
                kappa,
                theta,
                xi,
                rho,
                rho_assets,
                t,
                steps,
                code,
                strike,
                base,
                path,
                -1.0,
            )
            value = 0.5 * (value + other)
        out[i] = value
