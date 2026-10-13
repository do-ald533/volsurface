from collections.abc import Callable

import numpy as np
import pytest

from volsurface.domain.errors import InvalidInputError
from volsurface.pricing.american import (
    american_gamma_grid,
    american_implied_vol,
    american_price_gamma,
)
from volsurface.pricing.black_scholes import bs_price
from volsurface.pricing.crr_reference import crr_reference

S, R = 100.0, 0.04


def _random_options(n: int, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "K": S * rng.uniform(0.8, 1.2, n),
        "T": rng.uniform(0.02, 1.5, n),
        "q": rng.uniform(0.0, 0.05, n),
        "sigma": rng.uniform(0.1, 0.7, n),
        "is_call": rng.random(n) < 0.5,
    }


class TestPriceAndGamma:
    @pytest.mark.parametrize("american", [True, False])
    def test_numba_kernel_matches_the_numpy_reference(self, american: bool) -> None:
        o = _random_options(25)
        price, gamma = american_price_gamma(
            S, o["K"], o["T"], R, o["q"], o["sigma"], o["is_call"], steps=120, american=american
        )
        for i in range(25):
            ref_p, ref_g = crr_reference(
                S,
                o["K"][i],
                o["T"][i],
                R,
                o["q"][i],
                o["sigma"][i],
                bool(o["is_call"][i]),
                steps=120,
                american=american,
            )
            assert price[i] == pytest.approx(ref_p, rel=1e-11, abs=1e-12)
            assert gamma[i] == pytest.approx(ref_g, rel=1e-8, abs=1e-12)

    def test_european_tree_converges_to_black_scholes(self) -> None:
        exact = float(bs_price(S, 105.0, 0.5, R, 0.01, 0.3, True))
        errors = [
            abs(
                float(
                    american_price_gamma(
                        S, 105.0, 0.5, R, 0.01, 0.3, True, steps=n, american=False
                    )[0]
                )
                - exact
            )
            for n in (25, 100, 400, 1600)
        ]
        assert errors[-1] < 2e-3
        assert errors[-1] < errors[0]

    def test_american_put_is_worth_at_least_the_european_put_and_intrinsic(self) -> None:
        o = _random_options(40)
        put = np.zeros(40, dtype=bool)
        amer, _ = american_price_gamma(S, o["K"], o["T"], R, o["q"], o["sigma"], put)
        euro, _ = american_price_gamma(
            S, o["K"], o["T"], R, o["q"], o["sigma"], put, american=False
        )
        assert np.all(amer >= euro - 1e-12)
        assert np.all(amer >= np.maximum(o["K"] - S, 0.0) - 1e-12)
        assert np.any(amer > euro + 1e-6)

    def test_american_call_without_dividends_equals_the_european_call(self) -> None:
        o = _random_options(30)
        call = np.ones(30, dtype=bool)
        amer, _ = american_price_gamma(S, o["K"], o["T"], R, 0.0, o["sigma"], call)
        euro, _ = american_price_gamma(S, o["K"], o["T"], R, 0.0, o["sigma"], call, american=False)
        np.testing.assert_allclose(amer, euro, rtol=1e-12, atol=1e-12)

    def test_expired_or_zero_vol_options_are_worth_intrinsic_with_nan_gamma(self) -> None:
        price, gamma = american_price_gamma(S, [90.0, 110.0], [0.0, 0.5], R, 0.0, [0.2, 0.0], False)
        np.testing.assert_allclose(price, [0.0, 10.0])
        assert np.isnan(gamma).all()

    def test_nan_inputs_propagate(self) -> None:
        price, gamma = american_price_gamma(S, 100.0, 0.5, R, 0.0, np.array([0.2, np.nan]), True)
        assert np.isfinite(price[0])
        assert np.isnan(price[1])
        assert np.isnan(gamma[1])

    def test_keeps_the_input_shape(self) -> None:
        price, gamma = american_price_gamma(S, np.full((3, 4), 100.0), 0.5, R, 0.0, 0.2, True)
        assert price.shape == gamma.shape == (3, 4)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"S": 0.0}, "positive"),
            ({"K": -1.0}, "positive"),
            ({"sigma": -0.2}, "non-negative"),
            ({"steps": 2}, "at least 3"),
        ],
    )
    def test_invalid_arguments(self, kwargs: dict[str, float], message: str) -> None:
        args = {"S": 100.0, "K": 100.0, "sigma": 0.2, "steps": 50} | kwargs
        with pytest.raises(InvalidInputError, match=message):
            american_price_gamma(
                args["S"], args["K"], 0.5, R, 0.0, args["sigma"], True, steps=int(args["steps"])
            )

    def test_tree_outside_the_probability_range_gives_nan(self) -> None:
        price, _ = american_price_gamma(S, 100.0, 5.0, 0.5, 0.0, 0.01, True, steps=3)
        assert np.isnan(price)


@pytest.mark.oracle
@pytest.mark.parametrize("american", [True, False])
@pytest.mark.parametrize(
    ("K", "days", "r", "q", "sigma", "is_call"),
    [
        (100.0, 365, 0.05, 0.00, 0.20, False),
        (110.0, 90, 0.03, 0.02, 0.35, False),
        (90.0, 180, 0.03, 0.04, 0.30, True),
        (260.0, 30, 0.04, 0.01, 0.15, True),
        (95.0, 7, 0.04, 0.00, 0.45, False),
    ],
)
def test_tree_matches_quantlib_crr_engine(
    ql_american: Callable[..., tuple[float, float]],
    K: float,
    days: int,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    american: bool,
) -> None:
    spot = 250.0 if K > 200 else 100.0
    ql_price, ql_gamma = ql_american(spot, K, days, r, q, sigma, is_call, 200, american)
    price, gamma = american_price_gamma(
        spot, K, days / 365, r, q, sigma, is_call, steps=200, american=american
    )
    assert float(price) == pytest.approx(ql_price, rel=1e-9, abs=1e-10)
    assert float(gamma) == pytest.approx(ql_gamma, rel=1e-6, abs=1e-9)


class TestImpliedVol:
    def test_round_trips_a_whole_chain(self) -> None:
        o = _random_options(300, seed=5)
        price, _ = american_price_gamma(S, o["K"], o["T"], R, o["q"], o["sigma"], o["is_call"])
        iv = american_implied_vol(price, S, o["K"], o["T"], R, o["q"], o["is_call"])
        finite = np.isfinite(iv)
        assert finite.mean() > 0.95
        repriced, _ = american_price_gamma(
            S, o["K"][finite], o["T"][finite], R, o["q"][finite], iv[finite], o["is_call"][finite]
        )
        np.testing.assert_allclose(repriced, price[finite], rtol=1e-6, atol=1e-8)

    def test_brent_and_the_notebook_bisection_agree(self) -> None:
        o = _random_options(120, seed=9)
        price, _ = american_price_gamma(S, o["K"], o["T"], R, o["q"], o["sigma"], o["is_call"])
        brent = american_implied_vol(price, S, o["K"], o["T"], R, o["q"], o["is_call"])
        bisect = american_implied_vol(
            price, S, o["K"], o["T"], R, o["q"], o["is_call"], method="bisection"
        )
        both = np.isfinite(brent) & np.isfinite(bisect)
        assert both.mean() > 0.95
        np.testing.assert_allclose(brent[both], bisect[both], atol=1e-7)

    def test_nan_when_no_volatility_can_produce_the_price(self) -> None:
        iv = american_implied_vol([0.0, 1e6, np.nan, 5.0], S, 100.0, 0.5, R, 0.0, True)
        assert np.isnan(iv).tolist() == [True, True, True, False]

    def test_nan_for_expired_options(self) -> None:
        assert np.isnan(american_implied_vol(5.0, S, 100.0, 0.0, R, 0.0, True))

    def test_rejects_bad_settings(self) -> None:
        with pytest.raises(InvalidInputError):
            american_implied_vol(5.0, S, 100.0, 0.5, R, 0.0, True, sigma_tol=0.0)
        with pytest.raises(InvalidInputError):
            american_implied_vol(5.0, S, 100.0, 0.5, R, 0.0, True, steps=1)


class TestGammaGrid:
    def test_matches_pointwise_gamma(self) -> None:
        o = _random_options(12, seed=2)
        levels = np.array([0.9, 1.0, 1.1])
        grid = american_gamma_grid(
            S, o["K"], o["T"], R, o["q"], o["sigma"], o["is_call"], levels, steps=80
        )
        assert grid.shape == (12, 3)
        for j, level in enumerate(levels):
            _, gamma = american_price_gamma(
                S * level, o["K"], o["T"], R, o["q"], o["sigma"], o["is_call"], steps=80
            )
            np.testing.assert_allclose(grid[:, j], gamma, rtol=1e-12, atol=1e-14)

    def test_nan_volatility_rows_stay_nan(self) -> None:
        grid = american_gamma_grid(S, [100.0, 100.0], 0.5, R, 0.0, [0.2, np.nan], True, [1.0])
        assert np.isfinite(grid[0, 0])
        assert np.isnan(grid[1, 0])

    def test_rejects_bad_levels_and_shapes(self) -> None:
        with pytest.raises(InvalidInputError):
            american_gamma_grid(S, [100.0], 0.5, R, 0.0, 0.2, True, [])
        with pytest.raises(InvalidInputError):
            american_gamma_grid(S, [100.0], 0.5, R, 0.0, 0.2, True, [1.0, -1.0])
        with pytest.raises(InvalidInputError):
            american_gamma_grid(S, np.full((2, 2), 100.0), 0.5, R, 0.0, 0.2, True, [1.0])


class TestPrecision:
    """Single precision (as used by the original GPU notebook) costs a little accuracy."""

    def test_float32_price_error_is_small_and_gamma_error_is_larger(self) -> None:
        args = (S, 98.0, 0.4, R, 0.01, 0.3, False)
        p64, g64 = crr_reference(*args, steps=200, dtype=np.float64)
        p32, g32 = crr_reference(*args, steps=200, dtype=np.float32)
        assert abs(p32 - p64) / p64 < 1e-4
        assert abs(g32 - g64) / abs(g64) < 1e-2
