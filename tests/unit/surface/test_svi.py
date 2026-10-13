import numpy as np
import pytest
from scipy.optimize import approx_fprime

from volsurface.domain.errors import InvalidInputError
from volsurface.surface import svi, svi_kernels
from volsurface.surface.svi import (
    SviParams,
    butterfly_density_factor,
    check_slice,
    is_lee_compliant,
    total_variance,
)
from volsurface.surface.svi_fit import fit_svi_slice

TRUE = SviParams(a=0.01, b=0.15, rho=-0.45, m=0.02, sigma=0.12)
T = 0.25


def _numpy_g(k: np.ndarray, p: SviParams) -> np.ndarray:
    w = total_variance(k, p)
    wp = svi.first_derivative(k, p)
    wpp = svi.second_derivative(k, p)
    return (1 - k * wp / (2 * w)) ** 2 - wp**2 / 4 * (1 / w + 0.25) + wpp / 2


class TestParams:
    def test_validation(self) -> None:
        for bad in (
            {"b": -0.1},
            {"rho": 1.0},
            {"rho": -1.2},
            {"sigma": 0.0},
            {"a": -1.0},
            {"a": float("nan")},
        ):
            kwargs = {"a": 0.01, "b": 0.1, "rho": 0.0, "m": 0.0, "sigma": 0.1} | bad
            with pytest.raises(InvalidInputError):
                SviParams(**kwargs)

    def test_minimum_variance_is_the_curve_minimum(self) -> None:
        grid = np.linspace(-2, 2, 200001)
        assert total_variance(grid, TRUE).min() == pytest.approx(TRUE.min_variance, abs=1e-9)

    def test_fit_parametrisation_round_trip(self) -> None:
        again = SviParams.from_minimum_variance(*TRUE.as_fit_vector())
        assert again.a == pytest.approx(TRUE.a)
        assert again.b == TRUE.b

    def test_wing_slopes(self) -> None:
        k = np.array([50.0, 51.0])
        slope = np.diff(total_variance(k, TRUE))[0]
        assert slope == pytest.approx(TRUE.right_slope, rel=1e-3)

    def test_implied_vol(self) -> None:
        k = np.array([0.0])
        assert svi.implied_vol(k, T, TRUE)[0] == pytest.approx(
            np.sqrt(total_variance(k, TRUE)[0] / T)
        )
        with pytest.raises(InvalidInputError):
            svi.implied_vol(k, 0.0, TRUE)


class TestKernels:
    def test_butterfly_kernel_matches_the_numpy_formula(self) -> None:
        k = np.linspace(-1.5, 1.5, 61)
        np.testing.assert_allclose(butterfly_density_factor(k, TRUE), _numpy_g(k, TRUE), rtol=1e-10)

    def test_analytic_jacobian_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(0)
        k = np.sort(rng.uniform(-0.5, 0.5, 15))
        w = total_variance(k, TRUE) + rng.normal(0, 1e-4, k.size)
        sqrt_weight = rng.uniform(0.5, 1.5, k.size)
        x = np.array([0.012, 0.14, -0.4, 0.03, 0.15])

        def res(v: np.ndarray) -> np.ndarray:
            return svi_kernels.residuals_and_jacobian(v, k, w, sqrt_weight)[0]

        _, jac = svi_kernels.residuals_and_jacobian(x, k, w, sqrt_weight)
        for i in range(k.size):
            numeric = approx_fprime(x, lambda v, i=i: res(v)[i], 1e-7)
            np.testing.assert_allclose(jac[i], numeric, rtol=1e-4, atol=1e-7)

    def test_total_variance_kernel_matches_numpy(self) -> None:
        k = np.linspace(-1, 1, 33)
        got = svi_kernels.total_variance_grid(k, TRUE.a, TRUE.b, TRUE.rho, TRUE.m, TRUE.sigma)
        np.testing.assert_allclose(got, total_variance(k, TRUE), rtol=1e-12)


class TestButterflyAndWings:
    def test_well_behaved_slice_has_positive_density_factor(self) -> None:
        min_g, ok = check_slice(TRUE, -0.5, 0.5)
        assert ok
        assert min_g > 0
        assert is_lee_compliant(TRUE)

    def test_gatheral_jacquier_example_3_1_has_negative_density(self) -> None:
        paper = SviParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
        min_g, ok = check_slice(paper, -1.0, 1.0)
        assert min_g < 0
        assert not ok

    def test_steep_wings_violate_lees_bound(self) -> None:
        steep = SviParams(a=0.0, b=1.5, rho=0.5, m=0.0, sigma=0.2)
        assert not is_lee_compliant(steep)


class TestFit:
    @pytest.mark.parametrize("noise", [0.0, 2e-4])
    def test_recovers_a_known_slice(self, noise: float) -> None:
        rng = np.random.default_rng(1)
        k = np.linspace(-0.45, 0.45, 31)
        w = total_variance(k, TRUE) + rng.normal(0, noise, k.size)
        fit = fit_svi_slice(k, w, t=T)
        fitted = total_variance(k, fit.params)
        np.testing.assert_allclose(fitted, total_variance(k, TRUE), atol=max(5 * noise, 1e-8))
        assert fit.rmse_variance < max(3 * noise, 1e-7)
        assert fit.arbitrage_free
        assert fit.n_points == 31
        assert np.isfinite(fit.rmse_vol)

    def test_warm_start_skips_the_multistart_search(self) -> None:
        k = np.linspace(-0.4, 0.4, 25)
        w = total_variance(k, TRUE)
        cold = fit_svi_slice(k, w)
        warm = fit_svi_slice(k, w, initial=TRUE)
        assert warm.nfev < cold.nfev
        assert warm.rmse_variance < 1e-6

    def test_weights_pull_the_fit_toward_heavily_weighted_points(self) -> None:
        k = np.linspace(-0.4, 0.4, 21)
        w = total_variance(k, TRUE).copy()
        w[10] += 0.01
        flat = fit_svi_slice(k, w)
        heavy = np.ones(21)
        heavy[10] = 50.0
        pulled = fit_svi_slice(k, w, weights=heavy)
        err_flat = abs(total_variance(k[10], flat.params) - w[10])
        err_pulled = abs(total_variance(k[10], pulled.params) - w[10])
        assert err_pulled < err_flat

    def test_calendar_penalty_keeps_the_slice_above_the_previous_one(self) -> None:
        k = np.linspace(-0.4, 0.4, 25)
        w_short = total_variance(k, TRUE)
        short = fit_svi_slice(k, w_short).params

        crossing = w_short * 0.9
        crossing[:6] = w_short[:6] * 1.4

        def lower(grid: np.ndarray) -> np.ndarray:
            return total_variance(grid, short)

        free = fit_svi_slice(k, crossing)
        constrained = fit_svi_slice(k, crossing, lower_slice=lower)
        grid = np.linspace(-0.4, 0.4, 81)
        gap_free = np.maximum(lower(grid) - total_variance(grid, free.params), 0).max()
        gap_constrained = np.maximum(
            lower(grid) - total_variance(grid, constrained.params), 0
        ).max()
        assert gap_constrained < gap_free

    def test_ignores_non_positive_and_non_finite_points(self) -> None:
        k = np.linspace(-0.4, 0.4, 25)
        w = total_variance(k, TRUE)
        w[3] = np.nan
        w[7] = -1.0
        fit = fit_svi_slice(k, w)
        assert fit.n_points == 23

    def test_rejects_bad_inputs(self) -> None:
        k = np.linspace(-0.4, 0.4, 4)
        with pytest.raises(InvalidInputError, match="at least"):
            fit_svi_slice(k, total_variance(k, TRUE))
        with pytest.raises(InvalidInputError):
            fit_svi_slice(np.zeros(6), np.ones(5))
        with pytest.raises(InvalidInputError):
            fit_svi_slice(np.linspace(-1, 1, 6), np.ones(6), weights=-np.ones(6))
