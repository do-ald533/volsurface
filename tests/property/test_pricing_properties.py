"""Property tests for Black-Scholes and implied volatility over wide parameter ranges."""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from volsurface.pricing.black_scholes import bs_greeks, bs_price
from volsurface.pricing.implied_vol import implied_vol_bs

spots = st.floats(min_value=1.0, max_value=1000.0)
moneyness = st.floats(min_value=0.5, max_value=2.0)
expiries = st.floats(min_value=0.01, max_value=5.0)
rates = st.floats(min_value=-0.02, max_value=0.15)
yields = st.floats(min_value=0.0, max_value=0.08)
vols = st.floats(min_value=0.03, max_value=2.5)
rights = st.booleans()


@given(spots, moneyness, expiries, rates, yields, vols)
def test_put_call_parity(S: float, m: float, T: float, r: float, q: float, sigma: float) -> None:
    K = S * m
    call = float(bs_price(S, K, T, r, q, sigma, True))
    put = float(bs_price(S, K, T, r, q, sigma, False))
    expected = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert abs((call - put) - expected) <= 1e-9 * max(S, K)


@given(spots, moneyness, expiries, rates, yields, vols, rights)
def test_price_respects_no_arbitrage_bounds(
    S: float, m: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> None:
    K = S * m
    price = float(bs_price(S, K, T, r, q, sigma, is_call))
    disc_s, disc_k = S * np.exp(-q * T), K * np.exp(-r * T)
    lower = max((disc_s - disc_k) if is_call else (disc_k - disc_s), 0.0)
    upper = disc_s if is_call else disc_k
    tol = 1e-9 * max(S, K)
    assert lower - tol <= price <= upper + tol


@given(spots, moneyness, expiries, rates, yields, vols, rights)
def test_vega_and_gamma_are_non_negative(
    S: float, m: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> None:
    g = bs_greeks(S, S * m, T, r, q, sigma, is_call)
    assert float(g.vega) >= 0.0
    assert float(g.gamma) >= 0.0


@settings(max_examples=300, deadline=None)
@given(spots, moneyness, expiries, rates, yields, vols, rights)
def test_implied_vol_inverts_the_price_whenever_it_is_identifiable(
    S: float, m: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> None:
    K = S * m
    price = float(bs_price(S, K, T, r, q, sigma, is_call))
    disc_s, disc_k = S * np.exp(-q * T), K * np.exp(-r * T)
    lower = max((disc_s - disc_k) if is_call else (disc_k - disc_s), 0.0)
    upper = disc_s if is_call else disc_k
    assume(price - lower > 1e-8 * max(S, K) and upper - price > 1e-8 * max(S, K))
    iv = float(implied_vol_bs(price, S, K, T, r, q, is_call))
    assert np.isfinite(iv)
    repriced = float(bs_price(S, K, T, r, q, iv, is_call))
    assert abs(repriced - price) <= 1e-8 * max(1.0, price) + 1e-9
