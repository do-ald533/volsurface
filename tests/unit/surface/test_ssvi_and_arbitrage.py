import numpy as np
import pytest

from volsurface.domain.errors import InvalidInputError
from volsurface.marketdata.synthetic import default_ssvi
from volsurface.surface.arbitrage import build_report, check_calendar
from volsurface.surface.ssvi import SsviSurface
from volsurface.surface.svi import SviParams, check_slice, total_variance


def _svi_from_ssvi(surface: SsviSurface, t: float) -> SviParams:
    theta = float(surface.theta(t))
    phi = float(surface.phi(theta))
    rho = surface.rho
    return SviParams(
        a=theta * (1 - rho**2) / 2,
        b=theta * phi / 2,
        rho=rho,
        m=-rho / phi,
        sigma=np.sqrt(1 - rho**2) / phi,
    )


class TestSsvi:
    def test_default_surface_satisfies_the_gatheral_jacquier_conditions(self) -> None:
        conditions = default_ssvi().conditions()
        assert conditions.arbitrage_free
        assert conditions.calendar_free
        assert conditions.butterfly_free

    def test_atm_total_variance_equals_theta(self) -> None:
        surface = default_ssvi()
        t = surface.maturities
        np.testing.assert_allclose(surface.total_variance(0.0, t), surface.theta(t), rtol=1e-12)

    def test_theta_is_linear_between_nodes_and_proportional_beyond(self) -> None:
        surface = SsviSurface(np.array([0.1, 0.5]), np.array([0.004, 0.025]), -0.5, 0.8, 0.5)
        assert float(surface.theta(0.3)) == pytest.approx(0.0145)
        assert float(surface.theta(1.0)) == pytest.approx(0.05)
        assert float(surface.theta(0.05)) == pytest.approx(0.002)

    @pytest.mark.parametrize("t", [0.02, 0.1, 0.25, 0.4])
    def test_each_ssvi_slice_is_an_exact_raw_svi_slice(self, t: float) -> None:
        surface = default_ssvi()
        params = _svi_from_ssvi(surface, t)
        k = np.linspace(-0.6, 0.6, 41)
        np.testing.assert_allclose(
            total_variance(k, params), surface.total_variance(k, t), rtol=1e-10
        )
        _, butterfly_ok = check_slice(params, -0.6, 0.6)
        assert butterfly_ok

    def test_total_variance_is_non_decreasing_in_maturity(self) -> None:
        surface = default_ssvi()
        t = np.linspace(0.02, 0.4, 60)[:, None]
        k = np.linspace(-0.5, 0.5, 41)[None, :]
        w = surface.total_variance(k, t)
        assert np.all(np.diff(w, axis=0) >= -1e-12)

    def test_extreme_parameters_violate_the_conditions(self) -> None:
        aggressive = SsviSurface(np.array([0.5, 2.0]), np.array([0.2, 0.9]), -0.95, 6.0, 0.5)
        assert not aggressive.conditions().butterfly_free
        decreasing = SsviSurface(np.array([0.5, 1.0]), np.array([0.2, 0.1]), -0.5, 0.5, 0.5)
        assert not decreasing.conditions().theta_non_decreasing

    def test_validation(self) -> None:
        good = {
            "maturities": np.array([0.1, 0.2]),
            "atm_total_variance": np.array([0.01, 0.02]),
            "rho": -0.5,
            "eta": 1.0,
            "gamma": 0.5,
        }
        for bad in (
            {"maturities": np.array([0.2, 0.1])},
            {"atm_total_variance": np.array([0.01, -0.02])},
            {"atm_total_variance": np.array([0.01])},
            {"rho": 1.0},
            {"eta": 0.0},
            {"gamma": 1.5},
        ):
            with pytest.raises(InvalidInputError):
                SsviSurface(**(good | bad))

    def test_implied_vol(self) -> None:
        surface = default_ssvi()
        assert float(surface.implied_vol(0.0, 30 / 365)) == pytest.approx(0.18, rel=1e-9)


class TestArbitrageReport:
    def test_ordered_well_behaved_slices_are_clean(self) -> None:
        surface = default_ssvi()
        slices = [(t, _svi_from_ssvi(surface, t), -0.4, 0.4) for t in (0.05, 0.12, 0.25)]
        report = build_report(slices)
        assert report.ok
        assert report.butterfly_violations == 0
        assert report.calendar == ()

    def test_crossing_slices_are_flagged_with_the_worst_gap(self) -> None:
        short = SviParams(0.02, 0.15, -0.4, 0.0, 0.15)
        long_ = SviParams(0.005, 0.15, -0.4, 0.0, 0.15)
        violations = check_calendar([(0.2, long_), (0.1, short)], -0.4, 0.4)
        assert len(violations) == 1
        assert violations[0].t_short == 0.1
        assert violations[0].t_long == 0.2
        assert violations[0].max_gap == pytest.approx(0.015, abs=1e-6)

    def test_butterfly_example_from_the_paper_is_reported(self) -> None:
        paper = SviParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
        report = build_report([(0.5, paper, -1.0, 1.0)])
        assert not report.ok
        assert report.butterfly_violations == 1
        assert report.slices[0].min_g < 0

    def test_empty_input(self) -> None:
        report = build_report([])
        assert report.ok
        assert report.slices == ()
