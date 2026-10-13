import threading

import numpy as np
import pytest

from volsurface.domain.errors import InvalidInputError
from volsurface.pricing.black_scholes import bs_price
from volsurface.pricing.implied_vol import implied_vol_bs, implied_vol_bs_reference


def _chain(n: int, seed: int = 21) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    S = 100.0
    K = S * rng.uniform(0.6, 1.5, n)
    T = rng.uniform(0.003, 3.0, n)
    sigma = rng.uniform(0.05, 1.5, n)
    r = rng.uniform(-0.01, 0.06, n)
    q = rng.uniform(0.0, 0.04, n)
    is_call = rng.random(n) < 0.5
    price = bs_price(S, K, T, r, q, sigma, is_call)
    return {"price": price, "K": K, "T": T, "sigma": sigma, "r": r, "q": q, "is_call": is_call}


def test_compiled_solver_matches_the_numpy_specification() -> None:
    c = _chain(5000)
    fast = implied_vol_bs(c["price"], 100.0, c["K"], c["T"], c["r"], c["q"], c["is_call"])
    ref = implied_vol_bs_reference(c["price"], 100.0, c["K"], c["T"], c["r"], c["q"], c["is_call"])
    np.testing.assert_array_equal(np.isnan(fast), np.isnan(ref))
    both = np.isfinite(fast) & np.isfinite(ref)
    assert both.mean() > 0.9
    np.testing.assert_allclose(fast[both], ref[both], rtol=1e-4, atol=1e-8)
    repriced = bs_price(
        100.0,
        c["K"][both],
        c["T"][both],
        c["r"][both],
        c["q"][both],
        fast[both],
        c["is_call"][both],
    )
    np.testing.assert_allclose(repriced, c["price"][both], rtol=0, atol=2e-10)


def test_compiled_solver_handles_invalid_rows_like_the_specification() -> None:
    price = np.array([np.nan, -1.0, 0.0, 1e9, 5.0, 5.0])
    T = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.5])
    S = np.array([100.0, 100.0, 100.0, 100.0, 100.0, -3.0])
    fast = implied_vol_bs(price, S, 100.0, T, 0.03, 0.0, True)
    ref = implied_vol_bs_reference(price, S, 100.0, T, 0.03, 0.0, True)
    assert np.isnan(fast).all()
    assert np.isnan(ref).all()


def test_broadcasting_and_output_shape() -> None:
    strikes = np.array([[90.0, 100.0, 110.0], [95.0, 105.0, 115.0]])
    price = bs_price(100.0, strikes, 0.5, 0.03, 0.01, 0.25, True)
    iv = implied_vol_bs(price, 100.0, strikes, 0.5, 0.03, 0.01, True)
    assert iv.shape == (2, 3)
    np.testing.assert_allclose(iv, 0.25, rtol=1e-7)


def test_empty_input() -> None:
    assert implied_vol_bs(np.array([]), 100.0, np.array([]), 0.5, 0.03, 0.0, True).shape == (0,)


def test_rejects_bad_settings() -> None:
    with pytest.raises(InvalidInputError):
        implied_vol_bs(1.0, 100.0, 100.0, 0.5, 0.03, 0.0, True, tol=-1.0)


def test_concurrent_calls_from_several_threads_are_safe() -> None:
    c = _chain(2000, seed=5)
    expected = implied_vol_bs(c["price"], 100.0, c["K"], c["T"], c["r"], c["q"], c["is_call"])
    results: list[np.ndarray] = []

    def work() -> None:
        results.append(
            implied_vol_bs(c["price"], 100.0, c["K"], c["T"], c["r"], c["q"], c["is_call"])
        )

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 6
    for r in results:
        np.testing.assert_array_equal(r, expected)
