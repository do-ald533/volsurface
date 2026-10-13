"""Black-Scholes-Merton prices and Greeks for European options with a continuous dividend yield.

All functions are vectorised: scalars and arrays broadcast against each other. Conventions:

* ``T`` is time to expiry in years, ``r`` the continuously compounded risk-free rate and ``q`` the
  continuous dividend yield; ``sigma`` is the annualised volatility (0.20 means 20 %).
* ``vega`` is the derivative with respect to ``sigma`` itself (per 1.00 of volatility, not per
  volatility point), ``rho`` the derivative with respect to ``r``, and ``theta`` the derivative
  with respect to calendar time (so it is ``-dV/dT``, per year).
* When ``sigma * sqrt(T)`` is numerically zero the option is worth its discounted forward
  intrinsic value and the Greeks are ``nan`` (their limits are not finite numbers).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import FloatArray

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)
_MIN_TOTAL_VOL = 1e-12


@dataclass(frozen=True, slots=True)
class Greeks:
    """First- and second-order sensitivities (see the module docstring for conventions)."""

    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray
    rho: FloatArray


def norm_pdf(x: FloatArray) -> FloatArray:
    """Standard normal density."""
    return np.asarray(_INV_SQRT_2PI * np.exp(-0.5 * x * x))


def sign_of(is_call: ArrayLike) -> FloatArray:
    """+1.0 for calls and -1.0 for puts."""
    return np.where(np.asarray(is_call, dtype=np.bool_), 1.0, -1.0)


def _validated(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    arrays = np.broadcast_arrays(*(np.asarray(x, dtype=np.float64) for x in (S, K, T, r, q, sigma)))
    s, k, t, rate, div, vol = arrays
    for name, arr in zip(("S", "K", "T", "r", "q", "sigma"), arrays, strict=True):
        if not np.isfinite(arr).all():
            raise InvalidInputError(f"{name} must be finite")
    if (s <= 0).any() or (k <= 0).any():
        raise InvalidInputError("S and K must be positive")
    if (t < 0).any() or (vol < 0).any():
        raise InvalidInputError("T and sigma must be non-negative")
    return s, k, t, rate, div, vol


def d1_d2(
    S: FloatArray, K: FloatArray, T: FloatArray, r: FloatArray, q: FloatArray, sigma: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """The Black-Scholes d1 and d2 terms (no validation; inf/nan where sigma*sqrt(T) is 0)."""
    total_vol = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / total_vol
    return d1, d1 - total_vol


def price_raw(
    S: FloatArray,
    K: FloatArray,
    T: FloatArray,
    r: FloatArray,
    q: FloatArray,
    sigma: FloatArray,
    sign: FloatArray,
) -> FloatArray:
    """Price without input validation, for inner loops (the implied-vol solver)."""
    disc_s = S * np.exp(-q * T)
    disc_k = K * np.exp(-r * T)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    with np.errstate(invalid="ignore"):
        price = sign * (disc_s * ndtr(sign * d1) - disc_k * ndtr(sign * d2))
    intrinsic = np.maximum(sign * (disc_s - disc_k), 0.0)
    return np.where(sigma * np.sqrt(T) < _MIN_TOTAL_VOL, intrinsic, price)


def vega_raw(
    S: FloatArray, K: FloatArray, T: FloatArray, r: FloatArray, q: FloatArray, sigma: FloatArray
) -> FloatArray:
    """Vega without input validation; ``nan`` in the degenerate limit."""
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm_pdf(d1) * np.sqrt(T)


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
) -> FloatArray:
    """European option value."""
    s, k, t, rate, div, vol = _validated(S, K, T, r, q, sigma)
    return price_raw(s, k, t, rate, div, vol, np.broadcast_to(sign_of(is_call), s.shape))


def bs_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
) -> Greeks:
    """Delta, gamma, vega, theta and rho."""
    s, k, t, rate, div, vol = _validated(S, K, T, r, q, sigma)
    sign = np.broadcast_to(sign_of(is_call), s.shape)
    sqrt_t = np.sqrt(t)
    d1, d2 = d1_d2(s, k, t, rate, div, vol)
    disc_q = np.exp(-div * t)
    disc_r = np.exp(-rate * t)
    pdf = norm_pdf(d1)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = sign * disc_q * ndtr(sign * d1)
        gamma = disc_q * pdf / (s * vol * sqrt_t)
        vega = s * disc_q * pdf * sqrt_t
        theta = (
            -s * disc_q * pdf * vol / (2.0 * sqrt_t)
            - sign * rate * k * disc_r * ndtr(sign * d2)
            + sign * div * s * disc_q * ndtr(sign * d1)
        )
        rho = sign * k * t * disc_r * ndtr(sign * d2)
    degenerate = vol * sqrt_t < _MIN_TOTAL_VOL
    nan = np.full(s.shape, np.nan)
    return Greeks(
        delta=np.where(degenerate, nan, delta),
        gamma=np.where(degenerate, nan, gamma),
        vega=np.where(degenerate, nan, vega),
        theta=np.where(degenerate, nan, theta),
        rho=np.where(degenerate, nan, rho),
    )
