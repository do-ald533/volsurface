"""SVI fit benchmarks (deselected by default). Run: pytest -m perf tests/perf --benchmark-only."""

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from volsurface.surface.svi import SviParams, total_variance
from volsurface.surface.svi_fit import fit_svi_slice

pytestmark = pytest.mark.perf

TRUE = SviParams(a=0.01, b=0.15, rho=-0.45, m=0.02, sigma=0.12)


@pytest.fixture(scope="module")
def slice_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    k = np.linspace(-0.45, 0.45, 40)
    return k, total_variance(k, TRUE) + rng.normal(0, 2e-4, k.size)


def test_fit_one_slice_cold_start(
    benchmark: BenchmarkFixture, slice_data: tuple[np.ndarray, np.ndarray]
) -> None:
    k, w = slice_data
    fit = benchmark(fit_svi_slice, k, w)
    assert fit.arbitrage_free


def test_fit_one_slice_warm_start(
    benchmark: BenchmarkFixture, slice_data: tuple[np.ndarray, np.ndarray]
) -> None:
    k, w = slice_data
    seed = fit_svi_slice(k, w).params
    fit = benchmark(fit_svi_slice, k, w, initial=seed)
    assert fit.arbitrage_free
