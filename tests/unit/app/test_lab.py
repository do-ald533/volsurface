import numpy as np
import pytest

from volsurface.app.lab import (
    McInputs,
    NewtonInputs,
    VanillaInputs,
    build_model,
    price_vanilla,
    run_monte_carlo,
    trace_newton,
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
