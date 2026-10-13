"""Benchmarks (deselected by default). Run: pytest -m perf tests/perf --benchmark-only."""

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from volsurface.pricing.black_scholes import bs_price
from volsurface.pricing.implied_vol import implied_vol_bs, implied_vol_bs_reference

pytestmark = pytest.mark.perf

N = 10_000


@pytest.fixture(scope="module")
def chain() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    S = 100.0
    K = S * rng.uniform(0.8, 1.2, N)
    T = rng.uniform(0.01, 2.0, N)
    sigma = rng.uniform(0.1, 0.6, N)
    is_call = rng.random(N) < 0.5
    price = bs_price(S, K, T, 0.03, 0.01, sigma, is_call)
    return {"price": price, "K": K, "T": T, "is_call": is_call}


def test_implied_vol_10k_options_numba(
    benchmark: BenchmarkFixture, chain: dict[str, np.ndarray]
) -> None:
    result = benchmark(
        implied_vol_bs, chain["price"], 100.0, chain["K"], chain["T"], 0.03, 0.01, chain["is_call"]
    )
    assert np.isfinite(result).mean() > 0.99


def test_implied_vol_10k_options_numpy_reference(
    benchmark: BenchmarkFixture, chain: dict[str, np.ndarray]
) -> None:
    result = benchmark(
        implied_vol_bs_reference,
        chain["price"],
        100.0,
        chain["K"],
        chain["T"],
        0.03,
        0.01,
        chain["is_call"],
    )
    assert np.isfinite(result).mean() > 0.99


def test_bs_price_10k_options(benchmark: BenchmarkFixture, chain: dict[str, np.ndarray]) -> None:
    benchmark(bs_price, 100.0, chain["K"], chain["T"], 0.03, 0.01, 0.3, chain["is_call"])
