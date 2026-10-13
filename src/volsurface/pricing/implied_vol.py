"""Implied volatility under Black-Scholes-Merton.

Three solvers with different jobs:

* :func:`implied_vol_bs` is what the pipeline uses: safeguarded Newton-Raphson compiled with
  Numba and run in parallel, one option per loop iteration, each stopping as soon as it converges.
  Every iteration keeps a bracket ``[lo, hi]`` around the root (price is increasing in volatility)
  and falls back to bisection whenever the Newton step leaves the bracket or vega is tiny, so it
  converges for every price that is attainable and returns ``nan`` for the rest.
* :func:`implied_vol_bs_reference` is the same algorithm written with whole-array NumPy
  operations. It is the readable specification the compiled solver is tested against.
* :func:`newton_iterations` is the textbook algorithm, one option at a time, returning every
  step. It powers the "Newton explorer" in the Pricing Lab and fails loudly where plain Newton
  fails (vanishing vega), which is the reason the safeguarded version exists.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from volsurface.domain.errors import ConvergenceError, InvalidInputError
from volsurface.domain.types import FloatArray
from volsurface.pricing import bs_kernels
from volsurface.pricing._parallel import PARALLEL_LOCK
from volsurface.pricing.black_scholes import price_raw, sign_of, vega_raw

SIGMA_MIN = 1e-6
SIGMA_MAX = 10.0
_MIN_VEGA = 1e-14


def no_arbitrage_bounds(
    S: FloatArray, K: FloatArray, T: FloatArray, r: FloatArray, q: FloatArray, sign: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Lower and upper bound on a European option price: forward intrinsic and the asset/strike."""
    disc_s = S * np.exp(-q * T)
    disc_k = K * np.exp(-r * T)
    lower = np.maximum(sign * (disc_s - disc_k), 0.0)
    upper = np.where(sign > 0, disc_s, disc_k)
    return lower, upper


def implied_vol_bs_reference(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    is_call: ArrayLike,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> FloatArray:
    """NumPy specification of :func:`implied_vol_bs`; ``nan`` where no volatility exists.

    ``nan`` is returned for a non-finite price, ``T <= 0``, a price on or outside the
    no-arbitrage bounds, or a price that needs volatility outside ``[1e-6, 10]``.
    """
    p, s, k, t, rate, div = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (price, S, K, T, r, q))
    )
    call = np.broadcast_to(np.asarray(is_call, dtype=np.bool_), p.shape)
    if tol <= 0 or max_iter < 1:
        raise InvalidInputError("tol must be positive and max_iter at least 1")
    sign = sign_of(call)
    result = np.full(p.shape, np.nan)

    with np.errstate(invalid="ignore"):
        finite = np.isfinite(p) & np.isfinite(s) & np.isfinite(k) & np.isfinite(t)
        finite &= (s > 0) & (k > 0) & (t > 0)
        lower, upper = no_arbitrage_bounds(s, k, t, rate, div, sign)
        valid = finite & (p > lower) & (p < upper)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return result

    def take(a: FloatArray) -> FloatArray:
        return a.reshape(-1)[idx]

    target, s, k, t, rate, div, sg = (take(a) for a in (p, s, k, t, rate, div, sign))

    lo = np.full(idx.size, SIGMA_MIN)
    hi = np.full(idx.size, SIGMA_MAX)
    reachable = (price_raw(s, k, t, rate, div, lo, sg) <= target) & (
        price_raw(s, k, t, rate, div, hi, sg) >= target
    )
    forward = s * np.exp((rate - div) * t)
    sigma = np.clip(np.sqrt(2.0 * np.abs(np.log(forward / k)) / t), 0.15, 2.0)
    live = reachable.copy()
    out = np.full(idx.size, np.nan)

    for _ in range(max_iter):
        diff = price_raw(s, k, t, rate, div, sigma, sg) - target
        done = live & ((np.abs(diff) <= tol) | (hi - lo <= 1e-14))
        out[done] = sigma[done]
        live &= ~done
        if not live.any():
            break
        lo = np.where(live & (diff < 0), sigma, lo)
        hi = np.where(live & (diff > 0), sigma, hi)
        vega = vega_raw(s, k, t, rate, div, sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = np.where(vega > _MIN_VEGA, sigma - diff / vega, np.nan)
        use_newton = np.isfinite(newton) & (newton > lo) & (newton < hi)
        sigma = np.where(live, np.where(use_newton, newton, 0.5 * (lo + hi)), sigma)

    result.reshape(-1)[idx] = out
    return result


def implied_vol_bs(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    is_call: ArrayLike,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> FloatArray:
    """Implied volatility of European option prices; ``nan`` where none exists.

    ``nan`` is returned for a non-finite price, ``T <= 0``, a price on or outside the
    no-arbitrage bounds, or a price that needs volatility outside ``[1e-6, 10]``. Inputs broadcast
    against each other and the output has the broadcast shape.
    """
    if tol <= 0 or max_iter < 1:
        raise InvalidInputError("tol must be positive and max_iter at least 1")
    arrays = np.broadcast_arrays(
        *(np.asarray(x, dtype=np.float64) for x in (price, S, K, T, r, q)),
        np.asarray(is_call, dtype=np.bool_),
    )
    shape = arrays[0].shape
    p, s, k, t, rate, div = (
        np.array(a, dtype=np.float64, order="C", copy=True).reshape(-1) for a in arrays[:6]
    )
    sign = np.where(arrays[6].reshape(-1), 1.0, -1.0)
    out = np.empty_like(p)
    with PARALLEL_LOCK:
        bs_kernels.bs_iv_batch(p, s, k, t, rate, div, sign, tol, max_iter, out)
    return out.reshape(shape)


@dataclass(frozen=True, slots=True)
class NewtonStep:
    """One Newton-Raphson iteration, for plotting."""

    iteration: int
    sigma: float
    model_price: float
    vega: float
    error: float


def newton_iterations(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    is_call: bool,
    *,
    sigma0: float = 0.3,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> tuple[NewtonStep, ...]:
    """Plain Newton-Raphson on ``f(sigma) = BS(sigma) - price``, recording every iterate.

    Raises :class:`ConvergenceError` if vega vanishes, the iterate leaves ``(0, SIGMA_MAX]`` or
    the budget runs out; use :func:`implied_vol_bs` when robustness matters.
    """
    sign = sign_of(is_call)
    s_, k_, t_, r_, q_ = (np.asarray(x, dtype=np.float64) for x in (S, K, T, r, q))
    sigma = float(sigma0)
    steps: list[NewtonStep] = []
    for i in range(max_iter + 1):
        vol = np.asarray(sigma, dtype=np.float64)
        model = float(price_raw(s_, k_, t_, r_, q_, vol, sign))
        vega = float(vega_raw(s_, k_, t_, r_, q_, vol))
        error = model - price
        steps.append(NewtonStep(i, sigma, model, vega, error))
        if abs(error) <= tol:
            return tuple(steps)
        if not np.isfinite(vega) or vega < _MIN_VEGA:
            raise ConvergenceError(f"vega vanished at sigma={sigma:.6g}; Newton cannot continue")
        sigma -= error / vega
        if not (0.0 < sigma <= SIGMA_MAX):
            raise ConvergenceError(f"Newton step left the valid range (sigma={sigma:.6g})")
    raise ConvergenceError(f"no convergence within {max_iter} iterations")
