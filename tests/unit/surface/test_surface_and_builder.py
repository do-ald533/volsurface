from datetime import UTC, date, datetime

import numpy as np
import pytest

from volsurface.domain.calendar import years_to_expiry
from volsurface.domain.errors import ConfigurationError, InvalidInputError, PricingError
from volsurface.domain.types import ChainSnapshot
from volsurface.marketdata.synthetic import SyntheticChain, SyntheticConfig
from volsurface.pricing.american import american_implied_vol
from volsurface.pricing.quote_filter import surface_quote_spec
from volsurface.surface.builder import SurfaceBuilder
from volsurface.surface.surface import SurfaceSlice, VolSurface
from volsurface.surface.svi import SviParams, total_variance
from volsurface.surface.svi_fit import SviFit, fit_svi_slice

ASOF = datetime(2026, 10, 2, 14, 0, tzinfo=UTC)


def _fit(params: SviParams) -> SviFit:
    k = np.linspace(-0.4, 0.4, 21)
    return fit_svi_slice(k, total_variance(k, params))


def _slice(day: int, t: float, params: SviParams) -> SurfaceSlice:
    return SurfaceSlice(
        expiry=date(2026, 10, day),
        t=t,
        forward=100.0,
        fit=_fit(params),
        k_min=-0.4,
        k_max=0.4,
    )


def _surface(*slices: SurfaceSlice) -> VolSurface:
    return VolSurface("SYN", ASOF, 100.0, 0.03, 0.01, tuple(slices))


NEAR = SviParams(0.004, 0.06, -0.4, 0.0, 0.15)
FAR = SviParams(0.016, 0.10, -0.4, 0.0, 0.15)


class TestVolSurface:
    def test_returns_slice_values_at_the_nodes(self) -> None:
        s1, s2 = _slice(9, 0.1, NEAR), _slice(16, 0.2, FAR)
        surface = _surface(s1, s2)
        k = np.linspace(-0.3, 0.3, 7)
        np.testing.assert_allclose(surface.total_variance(k, 0.1), s1.total_variance(k), rtol=1e-9)
        np.testing.assert_allclose(surface.total_variance(k, 0.2), s2.total_variance(k), rtol=1e-9)

    def test_interpolates_linearly_in_total_variance(self) -> None:
        s1, s2 = _slice(9, 0.1, NEAR), _slice(16, 0.2, FAR)
        surface = _surface(s1, s2)
        mid = surface.total_variance(0.0, 0.15)
        expected = 0.5 * (s1.total_variance(0.0) + s2.total_variance(0.0))
        assert float(mid) == pytest.approx(float(expected))

    def test_blend_of_ordered_slices_is_monotone_in_maturity(self) -> None:
        surface = _surface(_slice(9, 0.1, NEAR), _slice(16, 0.2, FAR))
        t = np.linspace(0.1, 0.2, 30)[:, None]
        k = np.linspace(-0.3, 0.3, 13)[None, :]
        assert np.all(np.diff(surface.total_variance(k, t), axis=0) >= -1e-12)

    def test_extrapolates_with_flat_vol_outside_the_fitted_range(self) -> None:
        s1, s2 = _slice(9, 0.1, NEAR), _slice(16, 0.2, FAR)
        surface = _surface(s1, s2)
        assert float(surface.total_variance(0.0, 0.05)) == pytest.approx(
            float(s1.total_variance(0.0)) * 0.5
        )
        assert float(surface.total_variance(0.0, 0.4)) == pytest.approx(
            float(s2.total_variance(0.0)) * 2.0
        )

    def test_single_slice_surface(self) -> None:
        surface = _surface(_slice(9, 0.1, NEAR))
        assert float(surface.total_variance(0.0, 0.2)) == pytest.approx(
            2 * float(surface.slices[0].total_variance(0.0))
        )

    def test_iv_grid_orientation_and_implied_vol_by_strike(self) -> None:
        surface = _surface(_slice(9, 0.1, NEAR), _slice(16, 0.2, FAR))
        k = np.linspace(-0.2, 0.2, 5)
        t = np.array([0.1, 0.15, 0.2])
        grid = surface.iv_grid(k, t)
        assert grid.shape == (3, 5)
        strike = surface.forward(0.15) * np.exp(k[2])
        assert float(surface.implied_vol(strike, 0.15)) == pytest.approx(grid[1, 2], rel=1e-12)

    def test_slices_must_increase_in_maturity(self) -> None:
        with pytest.raises(InvalidInputError):
            _surface(_slice(16, 0.2, FAR), _slice(9, 0.1, NEAR))

    def test_empty_surface_cannot_be_evaluated(self) -> None:
        with pytest.raises(InvalidInputError):
            _surface().total_variance(0.0, 0.1)

    def test_arbitrage_report_covers_every_slice(self) -> None:
        report = _surface(_slice(9, 0.1, NEAR), _slice(16, 0.2, FAR)).arbitrage_report()
        assert len(report.slices) == 2
        assert report.ok


@pytest.fixture(scope="module")
def synthetic_market() -> tuple[ChainSnapshot, np.ndarray, SyntheticChain]:
    config = SyntheticConfig(seed=11, iv_noise=0.002, strike_step=1.0)
    generator = SyntheticChain(config)
    chain = generator.snapshot(ASOF, 100.0, 1.0)
    iv = _chain_ivs(chain)
    return chain, iv, generator


def _chain_ivs(chain: ChainSnapshot) -> np.ndarray:
    t = years_to_expiry(chain.asof, chain.expiry)
    usable = surface_quote_spec(max_days=200).mask(chain)
    iv = american_implied_vol(
        chain.mid,
        chain.spot,
        chain.strike,
        t,
        chain.rate,
        chain.dividend_yield,
        chain.is_call,
        steps=120,
    )
    return np.where(usable, iv, np.nan)


class TestBuilderOnSyntheticData:
    def test_recovers_the_ground_truth_surface(
        self, synthetic_market: tuple[ChainSnapshot, np.ndarray, SyntheticChain]
    ) -> None:
        chain, iv, generator = synthetic_market
        surface = SurfaceBuilder().with_chain(chain).with_implied_vols(iv).build()
        assert len(surface.slices) == len(generator.truth.maturities)
        for s in surface.slices:
            lo, hi = max(-0.12, s.k_min), min(0.12, s.k_max)
            assert hi - lo > 0.04, s.expiry
            k = np.linspace(lo, hi, 9)
            fitted = np.sqrt(s.total_variance(k) / s.t)
            truth = generator.truth.implied_vol(k, s.t)
            assert np.max(np.abs(fitted - truth)) < 0.01, s.expiry
            assert s.fit.arbitrage_free

    def test_fitted_surface_is_free_of_static_arbitrage(
        self, synthetic_market: tuple[ChainSnapshot, np.ndarray, SyntheticChain]
    ) -> None:
        chain, iv, _ = synthetic_market
        report = SurfaceBuilder().with_chain(chain).with_implied_vols(iv).build().arbitrage_report()
        assert report.ok
        assert report.calendar == ()

    def test_warm_start_reuses_the_previous_parameters(
        self, synthetic_market: tuple[ChainSnapshot, np.ndarray, SyntheticChain]
    ) -> None:
        chain, iv, _ = synthetic_market
        cold = SurfaceBuilder().with_chain(chain).with_implied_vols(iv).build()
        warm = (
            SurfaceBuilder().with_chain(chain).with_implied_vols(iv).with_warm_start(cold).build()
        )
        assert sum(s.fit.nfev for s in warm.slices) < sum(s.fit.nfev for s in cold.slices)

    def test_uniform_weighting_and_unrestricted_options(
        self, synthetic_market: tuple[ChainSnapshot, np.ndarray, SyntheticChain]
    ) -> None:
        chain, iv, _ = synthetic_market
        surface = (
            SurfaceBuilder()
            .with_chain(chain)
            .with_implied_vols(iv)
            .with_weighting("uniform")
            .with_otm_only(False)
            .with_calendar_awareness(False)
            .with_arbitrage_penalties(False)
            .with_min_points(8)
            .build()
        )
        assert surface.slices

    def test_expiries_with_too_few_quotes_are_skipped_with_a_reason(
        self, synthetic_market: tuple[ChainSnapshot, np.ndarray, SyntheticChain]
    ) -> None:
        chain, iv, _ = synthetic_market
        thinned = iv.copy()
        first = chain.expiry == np.min(chain.expiry)
        thinned[first] = np.nan
        surface = SurfaceBuilder().with_chain(chain).with_implied_vols(thinned).build()
        assert len(surface.skipped) == 1
        assert "usable quotes" in surface.skipped[0].reason
        assert len(surface.slices) == len(np.unique(chain.expiry)) - 1


class TestBuilderValidation:
    def test_requires_chain_and_vols(self) -> None:
        with pytest.raises(ConfigurationError, match="with_chain"):
            SurfaceBuilder().build()
        chain = SyntheticChain(SyntheticConfig(expiry_days=(30,))).snapshot(ASOF, 100.0, 1.0)
        with pytest.raises(ConfigurationError, match="with_implied_vols"):
            SurfaceBuilder().with_chain(chain).build()

    def test_vol_array_must_match_the_chain(self) -> None:
        chain = SyntheticChain(SyntheticConfig(expiry_days=(30,))).snapshot(ASOF, 100.0, 1.0)
        with pytest.raises(InvalidInputError, match="shape"):
            SurfaceBuilder().with_chain(chain).with_implied_vols(np.ones(3)).build()

    def test_all_nan_vols_cannot_form_a_surface(self) -> None:
        chain = SyntheticChain(SyntheticConfig(expiry_days=(30,))).snapshot(ASOF, 100.0, 1.0)
        with pytest.raises(PricingError, match="no expiry"):
            SurfaceBuilder().with_chain(chain).with_implied_vols(
                np.full(chain.size, np.nan)
            ).build()

    def test_min_points_has_a_floor(self) -> None:
        with pytest.raises(InvalidInputError):
            SurfaceBuilder().with_min_points(4)
