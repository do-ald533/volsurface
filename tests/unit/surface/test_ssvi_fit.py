import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volsurface.surface.ssvi_fit import fit_ssvi_slice, phi_max, ssvi_total_variance, to_raw_svi
from volsurface.surface.svi import butterfly_density_factor, total_variance
from volsurface.surface.svi_fit import fit_svi_slice


@given(
    theta=st.floats(min_value=1e-4, max_value=3.0),
    rho=st.floats(min_value=-0.98, max_value=0.98),
    u=st.floats(min_value=0.02, max_value=1.0),
)
@settings(max_examples=200, deadline=None)
def test_the_constrained_family_is_always_butterfly_free(
    theta: float, rho: float, u: float
) -> None:
    phi = u * phi_max(theta, rho)
    params = to_raw_svi(theta, rho, phi)
    k = np.linspace(-3.0, 3.0, 1201)
    assert np.min(butterfly_density_factor(k, params)) >= -1e-9


@given(
    theta=st.floats(min_value=1e-4, max_value=2.0),
    rho=st.floats(min_value=-0.95, max_value=0.95),
    u=st.floats(min_value=0.05, max_value=1.0),
)
def test_raw_svi_mapping_reproduces_the_ssvi_formula(theta: float, rho: float, u: float) -> None:
    phi = u * phi_max(theta, rho)
    k = np.linspace(-1.0, 1.0, 21)
    np.testing.assert_allclose(
        total_variance(k, to_raw_svi(theta, rho, phi)),
        ssvi_total_variance(k, theta, rho, phi),
        rtol=1e-9,
        atol=1e-12,
    )


def test_phi_max_satisfies_both_theorem_conditions() -> None:
    for theta in (0.001, 0.05, 0.5, 2.0, 10.0):
        for rho in (-0.9, 0.0, 0.7):
            phi = phi_max(theta, rho)
            assert theta * phi * (1 + abs(rho)) < 4
            assert theta * phi**2 * (1 + abs(rho)) <= 4 + 1e-12


def test_recovers_an_ssvi_slice_from_noisy_data() -> None:
    theta, rho = 0.012, -0.6
    phi = 0.5 * phi_max(theta, rho)
    rng = np.random.default_rng(0)
    k = np.linspace(-0.3, 0.2, 25)
    w = ssvi_total_variance(k, theta, rho, phi) + rng.normal(0, 1e-5, k.size)
    fit = fit_ssvi_slice(k, w, np.ones_like(k))
    fitted = total_variance(k, fit.params)
    assert np.max(np.abs(fitted - w)) < 1e-4
    assert fit.theta == pytest.approx(theta, rel=0.1)
    assert fit.rho == pytest.approx(rho, abs=0.15)
    assert math.isfinite(fit.phi)


def test_calendar_penalty_pushes_the_slice_above_a_floor() -> None:
    theta, rho = 0.01, -0.5
    phi = 0.4 * phi_max(theta, rho)
    k = np.linspace(-0.25, 0.2, 21)
    w = ssvi_total_variance(k, theta, rho, phi)

    def floor(grid: np.ndarray) -> np.ndarray:
        return ssvi_total_variance(grid, theta * 1.15, rho, phi)

    free = fit_ssvi_slice(k, w, np.ones_like(k))
    pushed = fit_ssvi_slice(k, w, np.ones_like(k), lower_slice=floor)
    grid = np.linspace(-0.25, 0.2, 61)
    gap_free = np.max(np.maximum(floor(grid) - total_variance(grid, free.params), 0))
    gap_pushed = np.max(np.maximum(floor(grid) - total_variance(grid, pushed.params), 0))
    assert gap_pushed < gap_free


def test_one_sided_data_is_fitted_without_arbitrage_by_the_fallback() -> None:
    steep = to_raw_svi(0.0018, -0.9, 0.9 * phi_max(0.0018, -0.9))
    k = np.linspace(-0.13, 0.03, 9)
    rng = np.random.default_rng(3)
    w = total_variance(k, steep) + rng.normal(0, 4e-5, k.size)
    fit = fit_svi_slice(k, np.maximum(w, 1e-6))
    assert fit.arbitrage_free
    assert fit.rmse_variance < 2e-4
