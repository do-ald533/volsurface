"""Monte Carlo benchmarks (deselected by default). Run: pytest -m perf tests/perf --benchmark-only."""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from volsurface.pricing.montecarlo import (
    GbmModel,
    HestonModel,
    MonteCarloEngine,
    Payoff,
    PayoffKind,
)

pytestmark = pytest.mark.perf


def test_gbm_call_1m_paths(benchmark: BenchmarkFixture) -> None:
    engine = MonteCarloEngine(GbmModel(100.0, 0.03, 0.01, 0.2))
    result = benchmark(engine.price, Payoff(PayoffKind.CALL, 100.0), 1.0, 1_000_000)
    assert result.std_error < 0.02


def test_heston_call_200k_paths_100_steps(benchmark: BenchmarkFixture) -> None:
    engine = MonteCarloEngine(HestonModel(100.0, 0.04, 0.03, 0.01, 1.5, 0.04, 0.3, -0.7), steps=100)
    result = benchmark(engine.price, Payoff(PayoffKind.CALL, 100.0), 1.0, 200_000)
    assert result.std_error < 0.1
