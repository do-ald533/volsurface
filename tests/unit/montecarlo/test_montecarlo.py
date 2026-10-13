import math

import numba
import numpy as np
import pytest

from volsurface.domain.errors import InvalidInputError
from volsurface.pricing.black_scholes import bs_price
from volsurface.pricing.montecarlo import (
    GbmModel,
    HestonModel,
    MonteCarloEngine,
    Payoff,
    PayoffKind,
    TwoHestonModel,
    kernels,
)

S0, K, R, Q, SIGMA, T = 100.0, 105.0, 0.03, 0.01, 0.25, 1.0


def _within(result, expected: float, sigmas: float = 3.5) -> None:  # type: ignore[no-untyped-def]
    assert abs(result.price - expected) <= sigmas * result.std_error, (
        f"{result.price:.5f} vs {expected:.5f} (se {result.std_error:.5f})"
    )


class TestRandomNumbers:
    def test_uniforms_are_in_the_open_unit_interval_with_the_right_moments(self) -> None:
        seed = np.uint64(11)
        u = np.array([kernels.uniform(seed, np.uint64(p), np.uint64(0)) for p in range(40000)])
        assert u.min() > 0.0 and u.max() < 1.0
        assert u.mean() == pytest.approx(0.5, abs=0.01)
        assert u.var() == pytest.approx(1 / 12, abs=0.005)

    def test_normals_have_standard_moments_and_pairs_are_uncorrelated(self) -> None:
        seed = np.uint64(5)
        pairs = np.array(
            [kernels.normal_pair(seed, np.uint64(p), np.uint64(0)) for p in range(40000)]
        )
        z1, z2 = pairs[:, 0], pairs[:, 1]
        for z in (z1, z2):
            assert z.mean() == pytest.approx(0.0, abs=0.02)
            assert z.std() == pytest.approx(1.0, abs=0.02)
            assert np.mean(z**3) == pytest.approx(0.0, abs=0.05)
            assert np.mean(z**4) == pytest.approx(3.0, abs=0.12)
        assert np.corrcoef(z1, z2)[0, 1] == pytest.approx(0.0, abs=0.02)

    def test_streams_of_neighbouring_paths_and_draws_are_uncorrelated(self) -> None:
        seed = np.uint64(5)
        a = np.array(
            [kernels.normal_pair(seed, np.uint64(p), np.uint64(0))[0] for p in range(20000)]
        )
        b = np.array(
            [kernels.normal_pair(seed, np.uint64(p + 1), np.uint64(0))[0] for p in range(20000)]
        )
        c = np.array(
            [kernels.normal_pair(seed, np.uint64(p), np.uint64(1))[0] for p in range(20000)]
        )
        assert abs(np.corrcoef(a, b)[0, 1]) < 0.03
        assert abs(np.corrcoef(a, c)[0, 1]) < 0.03

    def test_draws_depend_on_seed_path_and_index_only(self) -> None:
        one = kernels.uniform(np.uint64(1), np.uint64(2), np.uint64(3))
        assert one == kernels.uniform(np.uint64(1), np.uint64(2), np.uint64(3))
        assert one != kernels.uniform(np.uint64(2), np.uint64(2), np.uint64(3))
        assert one != kernels.uniform(np.uint64(1), np.uint64(3), np.uint64(3))
        assert one != kernels.uniform(np.uint64(1), np.uint64(2), np.uint64(4))


class TestGbm:
    @pytest.mark.parametrize("kind", [PayoffKind.CALL, PayoffKind.PUT])
    def test_matches_black_scholes_within_the_standard_error(self, kind: PayoffKind) -> None:
        engine = MonteCarloEngine(GbmModel(S0, R, Q, SIGMA))
        result = engine.price(Payoff(kind, K), T, 400_000, seed=3)
        expected = float(bs_price(S0, K, T, R, Q, SIGMA, kind is PayoffKind.CALL))
        _within(result, expected)
        assert result.std_error < 0.05

    def test_antithetic_variates_reduce_the_standard_error(self) -> None:
        model = GbmModel(S0, R, Q, SIGMA)
        plain = MonteCarloEngine(model, antithetic=False).price(
            Payoff(PayoffKind.CALL, K), T, 100_000
        )
        mirrored = MonteCarloEngine(model, antithetic=True).price(
            Payoff(PayoffKind.CALL, K), T, 100_000
        )
        assert mirrored.std_error < 0.85 * plain.std_error

    def test_standard_error_shrinks_like_one_over_root_n(self) -> None:
        engine = MonteCarloEngine(GbmModel(S0, R, Q, SIGMA))
        small = engine.price(Payoff(PayoffKind.CALL, K), T, 10_000, seed=1)
        large = engine.price(Payoff(PayoffKind.CALL, K), T, 160_000, seed=1)
        assert small.std_error / large.std_error == pytest.approx(4.0, rel=0.15)

    def test_asian_call_is_cheaper_than_the_european_call(self) -> None:
        engine = MonteCarloEngine(GbmModel(S0, R, Q, SIGMA), steps=50)
        asian = engine.price(Payoff(PayoffKind.ASIAN_CALL, K), T, 100_000, seed=2)
        european = engine.price(Payoff(PayoffKind.CALL, K), T, 100_000, seed=2)
        assert asian.price < european.price - 3 * (asian.std_error + european.std_error)

    def test_two_asset_payoffs_are_rejected(self) -> None:
        engine = MonteCarloEngine(GbmModel(S0, R, Q, SIGMA))
        with pytest.raises(InvalidInputError, match="cannot price"):
            engine.price(Payoff(PayoffKind.BEST_OF_CALL, K), T, 100)


class TestReproducibility:
    def test_same_seed_same_result_and_different_seed_different_result(self) -> None:
        engine = MonteCarloEngine(HestonModel(S0, 0.04, R, Q, 1.5, 0.04, 0.3, -0.7), steps=40)
        payoff = Payoff(PayoffKind.CALL, K)
        a = engine.price(payoff, T, 20_000, seed=9)
        b = engine.price(payoff, T, 20_000, seed=9)
        c = engine.price(payoff, T, 20_000, seed=10)
        assert a.price == b.price
        np.testing.assert_array_equal(a.running_price, b.running_price)
        assert a.price != c.price

    def test_results_do_not_depend_on_the_number_of_threads(self) -> None:
        engine = MonteCarloEngine(HestonModel(S0, 0.04, R, Q, 1.5, 0.04, 0.3, -0.7), steps=40)
        payoff = Payoff(PayoffKind.PUT, K)
        many = engine.price(payoff, T, 20_000, seed=4).price
        previous = numba.get_num_threads()
        numba.set_num_threads(1)
        try:
            single = engine.price(payoff, T, 20_000, seed=4).price
        finally:
            numba.set_num_threads(previous)
        assert many == single


class TestHeston:
    def test_zero_vol_of_vol_with_flat_variance_reduces_to_black_scholes(self) -> None:
        variance = SIGMA**2
        model = HestonModel(S0, variance, R, Q, kappa=2.0, theta=variance, xi=0.0, rho=-0.5)
        result = MonteCarloEngine(model, steps=100).price(
            Payoff(PayoffKind.CALL, K), T, 300_000, seed=7
        )
        expected = float(bs_price(S0, K, T, R, Q, SIGMA, True))
        assert abs(result.price - expected) <= 3.5 * result.std_error + 0.01

    def test_negative_correlation_makes_downside_puts_dearer_than_with_zero_correlation(
        self,
    ) -> None:
        common = {"s0": S0, "v0": 0.04, "r": R, "q": Q, "kappa": 1.5, "theta": 0.04, "xi": 0.6}
        payoff = Payoff(PayoffKind.PUT, 85.0)
        skewed = MonteCarloEngine(HestonModel(rho=-0.8, **common)).price(payoff, T, 200_000, seed=1)
        flat = MonteCarloEngine(HestonModel(rho=0.0, **common)).price(payoff, T, 200_000, seed=1)
        assert skewed.price > flat.price + 3 * (skewed.std_error + flat.std_error)

    def test_feller_condition_flag(self) -> None:
        assert HestonModel(S0, 0.04, R, Q, 1.5, 0.04, 0.3, -0.7).feller_satisfied
        assert not HestonModel(S0, 0.04, R, Q, 1.0, 0.04, 0.6, -0.7).feller_satisfied

    def test_parameter_validation(self) -> None:
        with pytest.raises(InvalidInputError):
            HestonModel(S0, -0.1, R, Q, 1.0, 0.04, 0.3, 0.0)
        with pytest.raises(InvalidInputError):
            HestonModel(S0, 0.04, R, Q, 1.0, 0.04, 0.3, 1.5)


class TestRainbow:
    def _engine(self, rho_assets: float) -> MonteCarloEngine:
        model = TwoHestonModel(S0, 0.04, R, Q, 1.5, 0.04, 0.3, -0.7, rho_assets)
        return MonteCarloEngine(model, steps=60)

    def test_best_of_dominates_each_leg_and_worst_of_is_dominated(self) -> None:
        engine = self._engine(0.0)
        best = engine.price(Payoff(PayoffKind.BEST_OF_CALL, 100.0), T, 100_000, seed=1)
        worst = engine.price(Payoff(PayoffKind.WORST_OF_CALL, 100.0), T, 100_000, seed=1)
        single = engine.price(Payoff(PayoffKind.CALL, 100.0), T, 100_000, seed=1)
        assert best.price > single.price + 3 * (best.std_error + single.std_error)
        assert worst.price < single.price - 3 * (worst.std_error + single.std_error)

    def test_higher_asset_correlation_lowers_the_best_of_price(self) -> None:
        payoff = Payoff(PayoffKind.BEST_OF_CALL, 100.0)
        independent = self._engine(0.0).price(payoff, T, 100_000, seed=2)
        correlated = self._engine(0.9).price(payoff, T, 100_000, seed=2)
        assert correlated.price < independent.price - 3 * (
            correlated.std_error + independent.std_error
        )

    def test_identical_perfectly_correlated_assets_make_best_equal_to_single(self) -> None:
        model = TwoHestonModel(S0, 0.04, R, Q, 1.5, 0.04, 0.0, 0.0, rho_assets=1.0)
        engine = MonteCarloEngine(model, steps=60)
        best = engine.price(Payoff(PayoffKind.BEST_OF_CALL, 100.0), T, 50_000, seed=3)
        single = engine.price(Payoff(PayoffKind.CALL, 100.0), T, 50_000, seed=3)
        assert abs(best.price - single.price) < 3 * (best.std_error + single.std_error) + 0.02

    def test_correlation_bounds_are_validated(self) -> None:
        with pytest.raises(InvalidInputError):
            TwoHestonModel(S0, 0.04, R, Q, 1.5, 0.04, 0.3, -0.7, rho_assets=1.2)


class TestResult:
    def test_convergence_history_is_consistent_with_the_final_estimate(self) -> None:
        result = MonteCarloEngine(GbmModel(S0, R, Q, SIGMA)).price(
            Payoff(PayoffKind.CALL, K), T, 50_000, checkpoints=32
        )
        assert result.counts[-1] == 50_000
        assert result.running_price[-1] == pytest.approx(result.price)
        assert result.running_std_error[-1] == pytest.approx(result.std_error)
        assert np.all(np.diff(result.counts) > 0)
        assert result.running_std_error[0] > result.running_std_error[-1]
        low, high = result.ci95
        assert low < result.price < high
        assert (high - low) / 2 == pytest.approx(1.96 * result.std_error, rel=1e-3)

    def test_single_path_has_no_standard_error(self) -> None:
        result = MonteCarloEngine(GbmModel(S0, R, Q, SIGMA), antithetic=False).price(
            Payoff(PayoffKind.CALL, K), T, 1
        )
        assert math.isnan(result.std_error)

    @pytest.mark.parametrize(
        ("t", "n"),
        [(0.0, 100), (-1.0, 100), (1.0, 0)],
    )
    def test_invalid_runs_are_rejected(self, t: float, n: int) -> None:
        with pytest.raises(InvalidInputError):
            MonteCarloEngine(GbmModel(S0, R, Q, SIGMA)).price(Payoff(PayoffKind.CALL, K), t, n)

    def test_invalid_settings(self) -> None:
        with pytest.raises(InvalidInputError):
            MonteCarloEngine(GbmModel(S0, R, Q, SIGMA), steps=0)
        with pytest.raises(InvalidInputError):
            Payoff(PayoffKind.CALL, 0.0)
        with pytest.raises(InvalidInputError):
            GbmModel(-1.0, R, Q, SIGMA)
