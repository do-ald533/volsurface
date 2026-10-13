"""The Monte Carlo engine: a fixed pipeline with a pluggable path model (Template Method).

``price`` always does the same four things: validate, ask the model for payoff samples, discount
them, and aggregate into a price with a standard error and a convergence history. Which stochastic
process generated the paths is the model's business (Strategy); the statistics live here once.
"""

import math
from dataclasses import dataclass

import numpy as np

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import FloatArray
from volsurface.pricing.montecarlo.models import PathModel, Payoff

Z_95 = 1.959963984540054


@dataclass(frozen=True, slots=True, eq=False)
class McResult:
    """A Monte Carlo price estimate with its uncertainty and how it converged."""

    price: float
    std_error: float
    n_paths: int
    counts: FloatArray
    running_price: FloatArray
    running_std_error: FloatArray

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - Z_95 * self.std_error, self.price + Z_95 * self.std_error


class MonteCarloEngine:
    def __init__(self, model: PathModel, *, steps: int = 100, antithetic: bool = True) -> None:
        if steps < 1:
            raise InvalidInputError("steps must be at least 1")
        self._model = model
        self._steps = steps
        self._antithetic = antithetic

    @property
    def model(self) -> PathModel:
        return self._model

    def price(
        self, payoff: Payoff, t: float, n_paths: int, *, seed: int = 0, checkpoints: int = 64
    ) -> McResult:
        samples = self._model.sample(
            payoff, t, n_paths, seed=seed, antithetic=self._antithetic, steps=self._steps
        )
        discounted = math.exp(-self._model.rate * t) * samples
        return _aggregate(discounted, checkpoints)


def _aggregate(samples: FloatArray, checkpoints: int) -> McResult:
    n = samples.size
    counts = np.unique(np.linspace(min(2, n), n, num=max(checkpoints, 2)).astype(np.int64))
    running_sum = np.cumsum(samples)
    running_sq = np.cumsum(samples * samples)
    idx = counts - 1
    mean = running_sum[idx] / counts
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = (running_sq[idx] - running_sum[idx] ** 2 / counts) / (counts - 1)
        std_error = np.sqrt(np.maximum(variance, 0.0) / counts)
    return McResult(
        price=float(mean[-1]),
        std_error=float(std_error[-1]) if n > 1 else math.nan,
        n_paths=n,
        counts=counts.astype(np.float64),
        running_price=mean,
        running_std_error=std_error,
    )
