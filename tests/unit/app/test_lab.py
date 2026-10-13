import numpy as np
import pytest

from volsurface.app.lab import (
    VANILLA_METRICS,
    McInputs,
    NewtonInputs,
    VanillaInputs,
    build_model,
    price_vanilla,
    run_monte_carlo,
    trace_newton,
    vanilla_curves,
)
from volsurface.domain.errors import InvalidInputError
from volsurface.pricing.backends import NumbaCpuBackend
from volsurface.pricing.montecarlo import GbmModel, HestonModel, PayoffKind, TwoHestonModel

BACKEND = NumbaCpuBackend()


class TestVanilla:
    def test_european_value_greeks_and_american_premium(self) -> None:
        quote = price_vanilla(VanillaInputs(is_call=False, days=180, tree_steps=300), BACKEND)
        assert 0 < quote.bs_price < 10
        assert -1 < quote.delta < 0
        assert quote.gamma > 0 and quote.vega > 0
        assert quote.american_price >= quote.bs_price - 1e-3
        assert quote.early_exercise_premium > 0
        assert quote.american_gamma > 0

    def test_american_call_without_dividends_has_no_premium(self) -> None:
        quote = price_vanilla(VanillaInputs(is_call=True, dividend_yield=0.0), BACKEND)
        assert quote.early_exercise_premium == pytest.approx(0.0, abs=2e-2)

    def test_years_conversion(self) -> None:
        assert VanillaInputs(days=365).years == 1.0


class TestVanillaCurves:
    def test_curves_agree_with_the_point_quote_at_the_current_spot(self) -> None:
        inputs = VanillaInputs(spot=100.0, strike=100.0, is_call=False)
        curves = vanilla_curves(inputs, BACKEND, points=121)
        i = int(np.argmin(np.abs(curves.spots - inputs.spot)))
        quote = price_vanilla(inputs, BACKEND)
        assert curves.spots[i] == pytest.approx(100.0)
        assert curves.bs_value[i] == pytest.approx(quote.bs_price)
        assert curves.american_value[i] == pytest.approx(quote.american_price)
        assert curves.greeks["delta"][i] == pytest.approx(quote.delta)
        assert curves.greeks["vega"][i] == pytest.approx(quote.vega)

    def test_shapes_range_and_intrinsic_value(self) -> None:
        inputs = VanillaInputs(spot=90.0, strike=110.0, is_call=True)
        curves = vanilla_curves(inputs, BACKEND, points=61)
        assert curves.spots.shape == curves.bs_value.shape == curves.american_value.shape == (61,)
        assert curves.spots[0] == pytest.approx(0.7 * 90.0)
        assert curves.spots[-1] == pytest.approx(1.3 * 110.0)
        assert np.allclose(curves.intrinsic, np.maximum(curves.spots - 110.0, 0.0))
        assert np.all(curves.bs_value >= curves.intrinsic - 1e-9)
        assert np.all(np.diff(curves.bs_value) > 0)

    def test_american_put_is_worth_at_least_the_european_put_and_its_intrinsic(self) -> None:
        curves = vanilla_curves(VanillaInputs(is_call=False, days=365, rate=0.06), BACKEND)
        assert np.all(curves.american_value >= curves.bs_value - 1e-3)
        assert np.all(curves.american_value >= curves.intrinsic - 1e-9)

    def test_series_per_metric(self) -> None:
        curves = vanilla_curves(VanillaInputs(), BACKEND, points=21)
        assert [name for name, _ in curves.series("Value")] == [
            "Black-Scholes",
            "American tree",
            "Intrinsic",
        ]
        for metric in VANILLA_METRICS[1:]:
            ((name, values),) = curves.series(metric)
            assert name == metric
            assert values.shape == (21,)

    def test_hover_snaps_to_the_nearest_grid_spot_and_marks_every_plotted_curve(self) -> None:
        inputs = VanillaInputs(spot=100.0, strike=100.0, is_call=False)
        curves = vanilla_curves(inputs, BACKEND, points=121)
        info = curves.hover(100.04)
        assert info.x == pytest.approx(
            float(curves.spots[np.argmin(np.abs(curves.spots - 100.04))])
        )
        assert "Spot" in info.title
        assert [label for label, _ in info.rows][:4] == [
            "Black-Scholes",
            "American tree",
            "Intrinsic",
            "Time value",
        ]
        assert len(info.points) == 3
        assert all(x == info.x for x, _ in info.points)
        assert len(curves.hover(100.0, "Delta").points) == 1
        assert curves.hover(-1e9).x == curves.spots[0]
        assert curves.hover(1e9).x == curves.spots[-1]


class TestMonteCarlo:
    def test_gbm_vanilla_reports_the_black_scholes_reference(self) -> None:
        outcome = run_monte_carlo(McInputs(n_paths=200_000))
        assert outcome.model_name == "GBM"
        assert outcome.reference is not None
        assert outcome.deviation_in_se is not None
        assert abs(outcome.deviation_in_se) < 4
        assert outcome.feller_ok is None

    def test_heston_has_no_closed_form_reference_but_reports_feller(self) -> None:
        outcome = run_monte_carlo(McInputs(model="heston", n_paths=20_000, steps=40))
        assert outcome.reference is None
        assert outcome.deviation_in_se is None
        assert outcome.feller_ok is True
        bad = run_monte_carlo(McInputs(model="heston", xi=1.0, n_paths=2_000, steps=20))
        assert bad.feller_ok is False

    def test_rainbow_payoff(self) -> None:
        outcome = run_monte_carlo(
            McInputs(model="rainbow", payoff=PayoffKind.BEST_OF_CALL, n_paths=20_000, steps=30)
        )
        assert outcome.model_name == "Two-asset Heston"
        assert outcome.result.price > 0

    def test_convergence_band_is_a_95_percent_interval_around_the_running_price(self) -> None:
        outcome = run_monte_carlo(McInputs(n_paths=50_000))
        counts, price, lower, upper = outcome.convergence()
        assert counts.shape == price.shape == lower.shape == upper.shape
        assert np.all(lower < price) and np.all(price < upper)
        assert counts[-1] == 50_000
        assert counts[0] >= 500
        width = (upper - lower)[-1]
        assert width == pytest.approx(2 * 1.959963984540054 * outcome.result.std_error)

    def test_hover_snaps_to_a_plotted_checkpoint_and_reports_the_reference(self) -> None:
        outcome = run_monte_carlo(McInputs(n_paths=50_000))
        counts, price, _, _ = outcome.convergence()
        info = outcome.hover(float(counts[3]) + 1.0)
        assert info.x == counts[3]
        assert info.points == ((counts[3], price[3]),)
        labels = [label for label, _ in info.rows]
        assert labels == ["Estimate", "Std error", "95% CI", "Black-Scholes", "Off by"]
        assert "paths" in info.title

    def test_hover_without_a_closed_form_reference(self) -> None:
        outcome = run_monte_carlo(McInputs(model="heston", n_paths=20_000, steps=30))
        labels = [label for label, _ in outcome.hover(10_000.0).rows]
        assert labels == ["Estimate", "Std error", "95% CI"]

    def test_hover_before_the_first_plotted_checkpoint_clamps_to_it(self) -> None:
        outcome = run_monte_carlo(McInputs(n_paths=50_000))
        counts, _, _, _ = outcome.convergence()
        assert outcome.hover(0.0).x == counts[0]
        assert outcome.hover(1e12).x == counts[-1]

    def test_model_factory(self) -> None:
        assert isinstance(build_model(McInputs(model="gbm")), GbmModel)
        assert isinstance(build_model(McInputs(model="heston")), HestonModel)
        assert isinstance(build_model(McInputs(model="rainbow")), TwoHestonModel)
        with pytest.raises(InvalidInputError):
            build_model(McInputs(model="nope"))  # type: ignore[arg-type]


class TestNewton:
    def test_trace_converges_and_records_the_curve(self) -> None:
        trace = trace_newton(NewtonInputs())
        assert trace.converged
        assert trace.steps[0].sigma == pytest.approx(0.10)
        assert trace.steps[-1].sigma == pytest.approx(0.30, rel=1e-6)
        assert np.all(np.diff(trace.prices) > 0)
        assert trace.prices.shape == trace.vols.shape == (200,)
        assert "converged" in trace.message

    def test_failure_is_reported_not_raised(self) -> None:
        trace = trace_newton(NewtonInputs(strike=140.0, days=30, market_vol=0.6, start_vol=0.02))
        assert not trace.converged
        assert trace.steps == ()
        assert trace.message

    def test_hover_on_an_iteration_shows_that_step(self) -> None:
        trace = trace_newton(NewtonInputs())
        first = trace.steps[1]
        info = trace.hover(100 * first.sigma + 0.05)
        assert info.title == "Iteration 1"
        assert info.x == pytest.approx(100 * first.sigma)
        rows = dict(info.rows)
        assert rows["Model price"] == f"{first.model_price:.6f}"
        assert rows["Model - market"] == f"{first.error:+.6f}"
        assert float(rows["Newton goes to"].removesuffix(" %")) == pytest.approx(
            100 * trace.steps[2].sigma, abs=1e-3
        )
        assert info.points == ((pytest.approx(100 * first.sigma), first.model_price),)

    def test_hover_on_the_last_iteration_says_converged(self) -> None:
        trace = trace_newton(NewtonInputs())
        info = trace.hover(100 * trace.steps[-1].sigma)
        assert dict(info.rows)["Newton goes to"] == "converged"

    def test_hover_between_iterations_predicts_the_next_newton_guess(self) -> None:
        inputs = NewtonInputs(market_vol=0.30, start_vol=0.10)
        trace = trace_newton(inputs)
        info = trace.hover(20.0)
        rows = dict(info.rows)
        assert info.title == "Newton from this guess"
        assert info.x == pytest.approx(20.0)
        assert float(rows["Model - market"]) < 0
        assert 20.0 < float(rows["Newton goes to"].removesuffix(" %")) < 40.0

    def test_hover_only_snaps_to_iterations_already_revealed_by_the_slider(self) -> None:
        trace = trace_newton(NewtonInputs())
        later = trace.steps[2]
        shown = trace.hover(100 * later.sigma, upto=0)
        assert shown.title == "Newton from this guess"
        assert trace.hover(100 * later.sigma, upto=2).title == "Iteration 2"

    def test_hover_works_when_newton_failed(self) -> None:
        trace = trace_newton(NewtonInputs(strike=140.0, days=30, market_vol=0.6, start_vol=0.02))
        assert not trace.converged
        info = trace.hover(50.0)
        assert info.title == "Newton from this guess"
        assert dict(info.rows)["Volatility"] == "50.0000 %"

    def test_flat_part_of_the_curve_has_no_newton_step(self) -> None:
        trace = trace_newton(NewtonInputs(strike=140.0, days=30, market_vol=0.6, start_vol=0.02))
        info = trace.hover(1.0)
        assert dict(info.rows)["Newton goes to"] == "flat curve: no step"
