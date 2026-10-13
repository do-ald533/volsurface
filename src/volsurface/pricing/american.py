"""American (and European) option values, gamma and implied volatility on a CRR tree.

Validated, vectorised front end over :mod:`volsurface.pricing.crr_kernels`. Inputs broadcast
against each other; ``nan`` inputs give ``nan`` outputs (so a half-filled chain can be passed
through), while values that are definitely wrong (non-positive ``S`` or ``K``, negative ``sigma``,
fewer than three steps) raise :class:`~volsurface.domain.errors.InvalidInputError`.

Batches are serialised with the process-wide lock in :mod:`volsurface.pricing._parallel`; each
batch still uses every core.
"""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import FloatArray
from volsurface.pricing import crr_kernels as k
from volsurface.pricing._parallel import PARALLEL_LOCK

DEFAULT_STEPS = 200
IVMethod = Literal["brent", "bisection"]


def _check_steps(steps: int) -> None:
    if steps < 3:
        raise InvalidInputError(f"steps must be at least 3 (gamma reads tree step 2), got {steps}")


def _flat(*values: ArrayLike) -> tuple[tuple[int, ...], list[FloatArray]]:
    """Broadcast to a common shape; return that shape and writable contiguous 1-D copies."""
    arrays = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in values))
    shape = arrays[0].shape
    return shape, [np.array(a, dtype=np.float64, order="C", copy=True).reshape(-1) for a in arrays]


def _check_market(S: FloatArray, K: FloatArray, sigma: FloatArray | None = None) -> None:
    with np.errstate(invalid="ignore"):
        if (S <= 0).any() or (K <= 0).any():
            raise InvalidInputError("S and K must be positive")
        if sigma is not None and (sigma < 0).any():
            raise InvalidInputError("sigma must be non-negative")


def american_price_gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
    *,
    steps: int = DEFAULT_STEPS,
    american: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """Option value and gamma for each option. ``american=False`` gives the European value."""
    _check_steps(steps)
    shape, (s, strike, t, rate, div, vol, call) = _flat(S, K, T, r, q, sigma, is_call)
    _check_market(s, strike, vol)
    sign = np.where(call > 0, 1.0, -1.0)
    price = np.empty_like(s)
    gamma = np.empty_like(s)
    with PARALLEL_LOCK:
        k.crr_batch(s, strike, t, rate, div, vol, sign, steps, american, price, gamma)
    return price.reshape(shape), gamma.reshape(shape)


def american_implied_vol(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    is_call: ArrayLike,
    *,
    steps: int = DEFAULT_STEPS,
    american: bool = True,
    method: IVMethod = "brent",
    sigma_tol: float = 1e-10,
    max_iter: int | None = None,
) -> FloatArray:
    """Volatility that makes the tree value equal ``price``; ``nan`` if there is none in [1%, 500%].

    ``method="bisection"`` reproduces the source notebook (fixed 40 halvings); the default
    ``"brent"`` finds the same root with far fewer tree evaluations. ``sigma_tol`` is the absolute
    tolerance on the returned volatility (Brent only).
    """
    _check_steps(steps)
    if sigma_tol <= 0:
        raise InvalidInputError("sigma_tol must be positive")
    shape, (p, s, strike, t, rate, div, call) = _flat(price, S, K, T, r, q, is_call)
    _check_market(s, strike)
    sign = np.where(call > 0, 1.0, -1.0)
    iterations = max_iter if max_iter is not None else (40 if method == "bisection" else 100)
    out = np.full_like(s, np.nan)
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(p) & (t > 0)
    idx = np.flatnonzero(usable)
    if idx.size:
        px, spot, kk, tt, rr, qq, sg = (
            np.ascontiguousarray(a[idx]) for a in (p, s, strike, t, rate, div, sign)
        )
        result = np.empty(idx.size)
        with PARALLEL_LOCK:
            k.iv_batch(
                px,
                spot,
                kk,
                tt,
                rr,
                qq,
                sg,
                steps,
                american,
                sigma_tol,
                iterations,
                method == "bisection",
                result,
            )
        out[idx] = result
    return out.reshape(shape)


def american_gamma_grid(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike,
    levels: ArrayLike,
    *,
    steps: int = DEFAULT_STEPS,
    american: bool = True,
) -> FloatArray:
    """Gamma of every option at spot ``S * level`` for each level; shape ``(options, levels)``.

    Option inputs must be one-dimensional (a flat chain). ``nan`` rows stay ``nan``.
    """
    _check_steps(steps)
    shape, (s, strike, t, rate, div, vol, call) = _flat(S, K, T, r, q, sigma, is_call)
    if len(shape) != 1:
        raise InvalidInputError("american_gamma_grid expects one-dimensional option inputs")
    _check_market(s, strike, vol)
    grid = np.array(levels, dtype=np.float64, order="C", copy=True).reshape(-1)
    if grid.size == 0 or (grid <= 0).any() or not np.isfinite(grid).all():
        raise InvalidInputError("levels must be a non-empty array of positive numbers")
    sign = np.where(call > 0, 1.0, -1.0)
    out = np.empty((s.size, grid.size))
    with PARALLEL_LOCK:
        k.gamma_grid_batch(s, strike, t, rate, div, vol, sign, grid, steps, american, out)
    return out
