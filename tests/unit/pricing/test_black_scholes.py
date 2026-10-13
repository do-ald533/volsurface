from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from volsurface.domain.errors import InvalidInputError
from volsurface.pricing.black_scholes import bs_greeks, bs_price


def test_matches_the_seed_function_example() -> None:
    call = bs_price(30, 40, 240 / 365, 0.01, 0.0, 0.03, True)
    put = bs_price(30, 40, 240 / 365, 0.01, 0.0, 0.03, False)
    forward_gap = 40 * np.exp(-0.01 * 240 / 365) - 30
    assert float(call) == pytest.approx(0.0, abs=1e-12)
    assert float(put) == pytest.approx(forward_gap, rel=1e-6)


def test_broadcasts_scalars_against_arrays() -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    prices = bs_price(100.0, strikes, 1.0, 0.03, 0.01, 0.2, True)
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)


def test_put_call_parity_on_a_grid() -> None:
    S, K, T, r, q, sigma = 100.0, np.linspace(70, 140, 15), 0.75, 0.04, 0.02, 0.27
    call = bs_price(S, K, T, r, q, sigma, True)
    put = bs_price(S, K, T, r, q, sigma, False)
    np.testing.assert_allclose(
        call - put, S * np.exp(-q * T) - K * np.exp(-r * T), rtol=1e-12, atol=1e-12
    )


@pytest.mark.oracle
@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize(
    ("S", "K", "days", "r", "q", "sigma"),
    [
        (100.0, 100.0, 365, 0.05, 0.00, 0.20),
        (100.0, 120.0, 90, 0.03, 0.02, 0.35),
        (250.0, 200.0, 30, 0.04, 0.01, 0.15),
        (50.0, 55.0, 700, 0.01, 0.00, 0.60),
        (100.0, 100.0, 7, -0.005, 0.03, 0.12),
    ],
)
def test_price_and_greeks_match_quantlib(
    ql_european: Callable[..., Any],
    S: float,
    K: float,
    days: int,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> None:
    T = days / 365
    oracle = ql_european(S, K, days, r, q, sigma, is_call)
    price = float(bs_price(S, K, T, r, q, sigma, is_call))
    greeks = bs_greeks(S, K, T, r, q, sigma, is_call)
    assert price == pytest.approx(oracle.price, rel=1e-9, abs=1e-10)
    assert float(greeks.delta) == pytest.approx(oracle.delta, rel=1e-8, abs=1e-10)
    assert float(greeks.gamma) == pytest.approx(oracle.gamma, rel=1e-8, abs=1e-10)
    assert float(greeks.vega) == pytest.approx(oracle.vega, rel=1e-8, abs=1e-10)
    assert float(greeks.theta) == pytest.approx(oracle.theta, rel=1e-7, abs=1e-8)
    assert float(greeks.rho) == pytest.approx(oracle.rho, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_central_finite_differences(is_call: bool) -> None:
    S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.03, 0.01, 0.25
    g = bs_greeks(S, K, T, r, q, sigma, is_call)

    def v(**kw: float) -> float:
        args = {"S": S, "K": K, "T": T, "r": r, "q": q, "sigma": sigma} | kw
        return float(
            bs_price(args["S"], args["K"], args["T"], args["r"], args["q"], args["sigma"], is_call)
        )

    h = 1e-4
    assert float(g.delta) == pytest.approx((v(S=S + h) - v(S=S - h)) / (2 * h), rel=1e-6)
    assert float(g.gamma) == pytest.approx((v(S=S + h) - 2 * v() + v(S=S - h)) / h**2, rel=1e-4)
    assert float(g.vega) == pytest.approx(
        (v(sigma=sigma + h) - v(sigma=sigma - h)) / (2 * h), rel=1e-6
    )
    assert float(g.rho) == pytest.approx((v(r=r + h) - v(r=r - h)) / (2 * h), rel=1e-6)
    assert float(g.theta) == pytest.approx(-(v(T=T + h) - v(T=T - h)) / (2 * h), rel=1e-6)


@pytest.mark.parametrize(("T", "sigma"), [(0.0, 0.2), (0.5, 0.0), (0.0, 0.0)])
def test_degenerate_inputs_give_discounted_forward_intrinsic(T: float, sigma: float) -> None:
    S, K, r, q = 105.0, 100.0, 0.03, 0.01
    call = float(bs_price(S, K, T, r, q, sigma, True))
    put = float(bs_price(S, K, T, r, q, sigma, False))
    forward_gap = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert call == pytest.approx(max(forward_gap, 0.0))
    assert put == pytest.approx(max(-forward_gap, 0.0))
    assert np.isnan(bs_greeks(S, K, T, r, q, sigma, True).gamma)


@pytest.mark.parametrize(
    "bad",
    [
        {"S": 0.0},
        {"S": -1.0},
        {"K": 0.0},
        {"T": -0.1},
        {"sigma": -0.2},
        {"S": float("nan")},
        {"r": float("inf")},
    ],
)
def test_invalid_inputs_raise(bad: dict[str, float]) -> None:
    args = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.01, "q": 0.0, "sigma": 0.2} | bad
    with pytest.raises(InvalidInputError):
        bs_price(args["S"], args["K"], args["T"], args["r"], args["q"], args["sigma"], True)
    with pytest.raises(InvalidInputError):
        bs_greeks(args["S"], args["K"], args["T"], args["r"], args["q"], args["sigma"], True)
