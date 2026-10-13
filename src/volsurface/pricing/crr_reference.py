"""A plain-NumPy CRR tree: slow, readable, dtype-selectable.

It exists for two reasons: it is the specification the Numba kernel is tested against, and it lets
``docs/math/crr.md`` measure what single precision (as used by the original GPU notebook) costs.
"""

import math

import numpy as np

from volsurface.domain.errors import InvalidInputError


def crr_reference(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    *,
    steps: int = 200,
    american: bool = True,
    dtype: type[np.floating] = np.float64,
) -> tuple[float, float]:
    """Price and gamma of one option; every intermediate array uses ``dtype``."""
    if steps < 3:
        raise InvalidInputError("steps must be at least 3 (gamma reads tree step 2)")
    f = dtype
    sign = f(1.0 if is_call else -1.0)
    dt = f(T) / f(steps)
    dx = f(sigma) * np.sqrt(dt)
    pu = f(0.5) + f(0.5) * (f(r) - f(q) - f(0.5) * f(sigma) ** 2) * dt / dx
    pd = f(1.0) - pu
    disc = np.exp(-f(r) * dt)
    spot, strike = f(S), f(K)

    def node_prices(step: int) -> np.ndarray:
        j = np.arange(step + 1, dtype=f)
        return np.asarray(spot * np.exp((f(2.0) * j - f(step)) * dx), dtype=f)

    zero = f(0.0)
    values = np.maximum(sign * (node_prices(steps) - strike), zero)
    v0 = v1 = v2 = zero
    for step in range(steps - 1, -1, -1):
        values = disc * (pu * values[1 : step + 2] + pd * values[: step + 1])
        if american:
            values = np.maximum(values, np.maximum(sign * (node_prices(step) - strike), zero))
        if step == 2:
            v0, v1, v2 = values[0], values[1], values[2]
    s_down = spot * np.exp(-f(2.0) * dx)
    s_up = spot * np.exp(f(2.0) * dx)
    gamma = ((v2 - v1) / (s_up - spot) - (v1 - v0) / (spot - s_down)) / (f(0.5) * (s_up - s_down))
    price = float(values[0])
    return (price, float(gamma)) if math.isfinite(price) else (math.nan, math.nan)
