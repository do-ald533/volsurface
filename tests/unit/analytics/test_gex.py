from datetime import UTC, date, datetime

import numpy as np
import pytest

from volsurface.analytics.gex import (
    atm_implied_vol,
    compute_gex,
    default_grid,
    flip_level,
)
from volsurface.domain.calendar import years_to_expiry
from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import ChainSnapshot, OptionContract, OptionQuote, Right
from volsurface.marketdata.synthetic import SyntheticChain, SyntheticConfig
from volsurface.pricing.american import american_implied_vol
from volsurface.pricing.backends import NumbaCpuBackend
from volsurface.pricing.quote_filter import AcceptAll, gex_quote_spec

ASOF = datetime(2026, 10, 2, 14, 0, tzinfo=UTC)
EXPIRY = date(2026, 11, 20)
BACKEND = NumbaCpuBackend()


def _chain(*quotes: OptionQuote, spot: float = 100.0) -> ChainSnapshot:
    return ChainSnapshot.from_quotes(
        underlying="SYN",
        asof=ASOF,
        spot=spot,
        rate=0.03,
        dividend_yield=0.0,
        quotes=quotes,
    )


def _quote(strike: float, right: Right, oi: int = 1000) -> OptionQuote:
    contract = OptionContract("SYN", EXPIRY, strike, right)
    return OptionQuote(contract, 2.0, 2.2, open_interest=oi)


class TestFlipLevel:
    def test_interpolates_linearly_between_grid_points(self) -> None:
        levels = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        gex = np.array([-2.0, -1.0, 1.0, 2.0, 3.0])
        assert flip_level(levels, gex, 100.0) == pytest.approx(97.5)

    def test_picks_the_crossing_nearest_spot(self) -> None:
        levels = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        gex = np.array([1.0, -1.0, -1.0, -1.0, 1.0])
        assert flip_level(levels, gex, 108.0) > 105.0
        assert flip_level(levels, gex, 91.0) < 95.0

    def test_nan_when_the_profile_never_changes_sign(self) -> None:
        levels = np.linspace(90, 110, 5)
        assert np.isnan(flip_level(levels, np.ones(5), 100.0))

    def test_rejects_bad_shapes(self) -> None:
        with pytest.raises(InvalidInputError):
            flip_level(np.array([1.0, 2.0]), np.array([1.0]), 1.5)
        with pytest.raises(InvalidInputError):
            flip_level(np.array([1.0]), np.array([1.0]), 1.0)


class TestGrid:
    def test_default_grid_includes_spot_and_is_symmetric(self) -> None:
        grid = default_grid()
        assert grid.size == 81
        assert grid[40] == pytest.approx(1.0)
        assert grid[0] == pytest.approx(0.9)
        assert grid[-1] == pytest.approx(1.1)

    def test_invalid_grids(self) -> None:
        with pytest.raises(InvalidInputError):
            default_grid(points=80)
        with pytest.raises(InvalidInputError):
            default_grid(low=1.1, high=1.2)


class TestComputeGex:
    def _iv(self, chain: ChainSnapshot) -> np.ndarray:
        t = years_to_expiry(chain.asof, chain.expiry)
        return american_implied_vol(
            chain.mid, chain.spot, chain.strike, t, chain.rate, chain.dividend_yield, chain.is_call
        )

    def test_a_lone_call_is_positive_and_a_lone_put_negative(self) -> None:
        for right, sign in ((Right.CALL, 1), (Right.PUT, -1)):
            chain = _chain(_quote(100.0, right))
            profile = compute_gex(chain, np.array([0.25]), BACKEND, spec=AcceptAll(), steps=60)
            assert np.all(sign * profile.gex >= 0)
            assert profile.gex_at_spot * sign > 0

    def test_matches_the_formula_for_one_option(self) -> None:
        chain = _chain(_quote(100.0, Right.CALL, oi=2500))
        sigma = np.array([0.3])
        profile = compute_gex(chain, sigma, BACKEND, spec=AcceptAll(), steps=80)
        t = years_to_expiry(chain.asof, chain.expiry)
        _, gamma = BACKEND.american_price_gamma(100.0, 100.0, t, 0.03, 0.0, sigma, True, steps=80)
        expected = 2500 * 100 * 0.01 / 1e9 * float(gamma[0]) * 100.0**2
        assert profile.gex_at_spot == pytest.approx(expected, rel=1e-10)

    def test_ignores_rows_without_open_interest_or_volatility(self) -> None:
        chain = _chain(
            _quote(100.0, Right.CALL),
            OptionQuote(OptionContract("SYN", EXPIRY, 105.0, Right.CALL), 1.0, 1.2),
            _quote(95.0, Right.PUT),
        )
        iv = np.array([0.25, 0.25, np.nan])
        profile = compute_gex(chain, iv, BACKEND, spec=gex_quote_spec(), steps=60)
        assert profile.n_options == 1

    def test_empty_selection_gives_a_flat_zero_profile(self) -> None:
        chain = _chain(_quote(100.0, Right.CALL))
        profile = compute_gex(chain, np.array([np.nan]), BACKEND, spec=AcceptAll())
        assert profile.n_options == 0
        assert np.all(profile.gex == 0)
        assert np.isnan(profile.flip)
        assert profile.size == 0.0

    def test_synthetic_market_has_a_sign_change_with_calls_above_and_puts_below(self) -> None:
        chain = SyntheticChain(SyntheticConfig(seed=4, iv_noise=0.0)).snapshot(ASOF, 100.0, 1.0)
        iv = self._iv(chain)
        profile = compute_gex(chain, iv, BACKEND, spec=gex_quote_spec(max_days=200), steps=60)
        assert profile.n_options > 100
        assert profile.gex[0] < 0 < profile.gex[-1]
        assert 90.0 < profile.flip < 110.0
        assert np.isfinite(profile.distance_to_flip_pct)

    def test_rejects_misaligned_volatilities(self) -> None:
        chain = _chain(_quote(100.0, Right.CALL))
        with pytest.raises(InvalidInputError):
            compute_gex(chain, np.array([0.2, 0.2]), BACKEND, spec=AcceptAll())


class TestAtmImpliedVol:
    def test_uses_nearest_expiry_at_least_a_day_out_and_nearest_strike(self) -> None:
        near = date(2026, 10, 2)
        nxt = date(2026, 10, 16)
        quotes = [
            OptionQuote(OptionContract("SYN", near, 100.0, Right.CALL), 1.0, 1.1),
            OptionQuote(OptionContract("SYN", nxt, 98.0, Right.CALL), 1.0, 1.1),
            OptionQuote(OptionContract("SYN", nxt, 101.0, Right.CALL), 1.0, 1.1),
            OptionQuote(OptionContract("SYN", date(2026, 11, 20), 100.0, Right.CALL), 1.0, 1.1),
        ]
        chain = _chain(*quotes)
        iv = np.array([0.10, 0.20, 0.30, 0.40])
        assert atm_implied_vol(chain, iv) == pytest.approx(30.0)

    def test_nan_without_usable_volatilities(self) -> None:
        chain = _chain(_quote(100.0, Right.CALL))
        assert np.isnan(atm_implied_vol(chain, np.array([np.nan])))
