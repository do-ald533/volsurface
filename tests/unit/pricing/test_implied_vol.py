import numpy as np
import pytest

from volsurface.domain.errors import ConvergenceError, InvalidInputError
from volsurface.pricing.black_scholes import bs_price
from volsurface.pricing.implied_vol import implied_vol_bs, newton_iterations

S, K, T, R, Q = 100.0, 100.0, 0.5, 0.03, 0.01


def test_recovers_the_volatility_that_priced_the_option() -> None:
    for sigma in (0.05, 0.2, 0.8, 2.5):
        for is_call in (True, False):
            px = bs_price(S, 110.0, T, R, Q, sigma, is_call)
            iv = implied_vol_bs(px, S, 110.0, T, R, Q, is_call)
            assert float(iv) == pytest.approx(sigma, rel=1e-6)


def test_vectorised_over_a_whole_chain() -> None:
    strikes = np.linspace(70.0, 140.0, 29)
    is_call = strikes >= S
    true_iv = 0.2 + 0.0004 * (strikes - S) ** 2 / 10
    px = bs_price(S, strikes, T, R, Q, true_iv, is_call)
    iv = implied_vol_bs(px, S, strikes, T, R, Q, is_call)
    np.testing.assert_allclose(iv, true_iv, rtol=1e-6)


def test_returns_nan_outside_the_no_arbitrage_bounds() -> None:
    forward_intrinsic = S * np.exp(-Q * T) - 90.0 * np.exp(-R * T)
    prices = np.array(
        [
            forward_intrinsic - 0.01,
            forward_intrinsic,
            S * np.exp(-Q * T) + 0.01,
            np.nan,
            -1.0,
        ]
    )
    iv = implied_vol_bs(prices, S, 90.0, T, R, Q, True)
    assert np.isnan(iv).all()


def test_nan_for_non_positive_expiry_and_unreachable_vol() -> None:
    assert np.isnan(implied_vol_bs(1.0, S, K, 0.0, R, Q, True))
    huge = float(bs_price(S, K, T, R, Q, 50.0, True))
    assert np.isnan(implied_vol_bs(huge, S, K, T, R, Q, True))


def test_deep_otm_where_plain_newton_would_stall() -> None:
    sigma = 0.15
    px = bs_price(S, 160.0, 0.05, R, Q, sigma, True)
    assert float(px) < 1e-6
    iv = implied_vol_bs(px, S, 160.0, 0.05, R, Q, True)
    assert float(bs_price(S, 160.0, 0.05, R, Q, iv, True)) == pytest.approx(float(px), abs=1e-9)


def test_rejects_bad_solver_settings() -> None:
    with pytest.raises(InvalidInputError):
        implied_vol_bs(1.0, S, K, T, R, Q, True, tol=0.0)
    with pytest.raises(InvalidInputError):
        implied_vol_bs(1.0, S, K, T, R, Q, True, max_iter=0)


class TestNewtonTrace:
    def test_converges_and_records_every_iterate(self) -> None:
        target = float(bs_price(S, K, T, R, Q, 0.35, True))
        steps = newton_iterations(target, S, K, T, R, Q, True, sigma0=0.1)
        assert steps[0].sigma == pytest.approx(0.1)
        assert steps[-1].sigma == pytest.approx(0.35, rel=1e-8)
        assert abs(steps[-1].error) <= 1e-10
        assert [s.iteration for s in steps] == list(range(len(steps)))
        errors = [abs(s.error) for s in steps]
        assert errors == sorted(errors, reverse=True)

    def test_agrees_with_the_safeguarded_solver(self) -> None:
        target = float(bs_price(S, 95.0, T, R, Q, 0.28, False))
        steps = newton_iterations(target, S, 95.0, T, R, Q, False)
        assert steps[-1].sigma == pytest.approx(
            float(implied_vol_bs(target, S, 95.0, T, R, Q, False))
        )

    def test_fails_loudly_where_vega_vanishes(self) -> None:
        target = float(bs_price(S, 200.0, 0.02, R, Q, 0.2, True)) + 1e-9
        with pytest.raises(ConvergenceError):
            newton_iterations(max(target, 1e-9), S, 200.0, 0.02, R, Q, True, sigma0=0.05)

    def test_reports_exhausted_budget(self) -> None:
        target = float(bs_price(S, K, T, R, Q, 0.35, True))
        with pytest.raises(ConvergenceError, match="no convergence"):
            newton_iterations(target, S, K, T, R, Q, True, sigma0=0.1, tol=1e-30, max_iter=2)
