"""CRR tree benchmarks (deselected by default). Run: pytest -m perf tests/perf --benchmark-only."""

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from volsurface.pricing.american import (
    american_gamma_grid,
    american_implied_vol,
    american_price_gamma,
)

pytestmark = pytest.mark.perf

N = 1_500


@pytest.fixture(scope="module")
def chain() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    K = 100.0 * rng.uniform(0.85, 1.15, N)
    T = rng.uniform(0.02, 0.4, N)
    sigma = rng.uniform(0.15, 0.5, N)
    is_call = rng.random(N) < 0.5
    price, _ = american_price_gamma(100.0, K, T, 0.03, 0.01, sigma, is_call)
    return {"K": K, "T": T, "sigma": sigma, "is_call": is_call, "price": price}


def test_price_gamma_1500_options_200_steps(
    benchmark: BenchmarkFixture, chain: dict[str, np.ndarray]
) -> None:
    benchmark(
        american_price_gamma,
        100.0,
        chain["K"],
        chain["T"],
        0.03,
        0.01,
        chain["sigma"],
        chain["is_call"],
    )


def test_implied_vol_brent_1500_options(
    benchmark: BenchmarkFixture, chain: dict[str, np.ndarray]
) -> None:
    benchmark(
        american_implied_vol,
        chain["price"],
        100.0,
        chain["K"],
        chain["T"],
        0.03,
        0.01,
        chain["is_call"],
    )


def test_implied_vol_bisection_1500_options(
    benchmark: BenchmarkFixture, chain: dict[str, np.ndarray]
) -> None:
    benchmark(
        american_implied_vol,
        chain["price"],
        100.0,
        chain["K"],
        chain["T"],
        0.03,
        0.01,
        chain["is_call"],
        method="bisection",
    )


def test_gamma_grid_1500_options_81_levels(
    benchmark: BenchmarkFixture, chain: dict[str, np.ndarray]
) -> None:
    levels = np.linspace(0.9, 1.1, 81)
    benchmark.pedantic(
        american_gamma_grid,
        args=(100.0, chain["K"], chain["T"], 0.03, 0.01, chain["sigma"], chain["is_call"], levels),
        rounds=3,
        iterations=1,
    )
