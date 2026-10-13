from importlib.metadata import EntryPoint

import numpy as np
import pytest

from volsurface.domain.errors import ConfigurationError
from volsurface.pricing import backends
from volsurface.pricing.backends import (
    BackendRegistry,
    NumbaCpuBackend,
    PricingBackend,
    TimedBackend,
    default_registry,
)


def test_numba_backend_satisfies_the_protocol_and_warms_up() -> None:
    backend = NumbaCpuBackend()
    assert isinstance(backend, PricingBackend)
    backend.warmup()
    price, gamma = backend.american_price_gamma(100.0, 100.0, 0.5, 0.03, 0.0, 0.2, True, steps=50)
    assert float(price) > 0
    assert float(gamma) > 0


def test_default_registry_creates_the_cpu_backend() -> None:
    registry = default_registry()
    assert "numba-cpu" in registry.names
    assert registry.create().name == "numba-cpu"


def test_registry_rejects_duplicates_and_unknown_names() -> None:
    registry = BackendRegistry()
    registry.register("a", NumbaCpuBackend)
    with pytest.raises(ConfigurationError, match="already registered"):
        registry.register("a", NumbaCpuBackend)
    registry.register("a", NumbaCpuBackend, replace=True)
    with pytest.raises(ConfigurationError, match="unknown pricing backend 'b'"):
        registry.create("b")


def test_entry_point_plugins_are_loaded_and_broken_ones_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    good = EntryPoint("gpu-fake", "volsurface.pricing.backends:NumbaCpuBackend", "g")
    broken = EntryPoint("broken", "volsurface.does_not_exist:Nope", "g")
    monkeypatch.setattr(backends, "entry_points", lambda group: (good, broken))
    registry = BackendRegistry()
    with caplog.at_level("WARNING"):
        loaded = registry.load_entry_points()
    assert loaded == ["gpu-fake"]
    assert registry.names == ("gpu-fake",)
    assert "broken" in caplog.text


def test_timed_backend_reports_each_call_and_forwards_results() -> None:
    calls: list[tuple[str, float, int]] = []
    inner = NumbaCpuBackend()
    timed = TimedBackend(inner, lambda op, seconds, n: calls.append((op, seconds, n)))
    assert isinstance(timed, PricingBackend)
    assert timed.name == inner.name

    strikes = np.array([95.0, 100.0, 105.0])
    price, _ = timed.american_price_gamma(100.0, strikes, 0.5, 0.03, 0.0, 0.2, True, steps=50)
    expected, _ = inner.american_price_gamma(100.0, strikes, 0.5, 0.03, 0.0, 0.2, True, steps=50)
    np.testing.assert_array_equal(price, expected)
    timed.american_implied_vol(price, 100.0, strikes, 0.5, 0.03, 0.0, True, steps=50)
    timed.american_gamma_grid(100.0, strikes, 0.5, 0.03, 0.0, 0.2, True, [1.0], steps=50)
    timed.warmup()

    assert [c[0] for c in calls] == [
        "american_price_gamma",
        "american_implied_vol",
        "american_gamma_grid",
    ]
    assert all(c[1] >= 0 and c[2] == 3 for c in calls)


def test_timed_backend_reports_even_when_the_call_fails() -> None:
    calls: list[str] = []
    timed = TimedBackend(NumbaCpuBackend(), lambda op, _s, _n: calls.append(op))
    with pytest.raises(Exception, match="positive"):
        timed.american_price_gamma(-1.0, 100.0, 0.5, 0.03, 0.0, 0.2, True, steps=50)
    assert calls == ["american_price_gamma"]
