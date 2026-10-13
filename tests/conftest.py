"""Shared pytest configuration and fixtures.

The Qt platform must be chosen before any Qt module is imported, so it is set here, at
collection time, for every test run (CI and local) without needing a display server.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@dataclass(frozen=True, slots=True)
class QLGreeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@pytest.fixture(scope="session")
def ql_european() -> Callable[..., QLGreeks]:
    """QuantLib's analytic European engine as an independent oracle (skips if not installed).

    ``days`` is the integer number of calendar days to expiry, so T = days / 365 exactly under the
    Actual/365 (Fixed) convention used here.
    """
    ql = pytest.importorskip("QuantLib")

    def price(
        S: float, K: float, days: int, r: float, q: float, sigma: float, is_call: bool
    ) -> QLGreeks:
        today = ql.Date(2, 10, 2026)
        ql.Settings.instance().evaluationDate = today
        dc = ql.Actual365Fixed()
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
        option = ql.VanillaOption(payoff, ql.EuropeanExercise(today + days))
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc)
            ),
        )
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return QLGreeks(
            price=option.NPV(),
            delta=option.delta(),
            gamma=option.gamma(),
            vega=option.vega(),
            theta=option.theta(),
            rho=option.rho(),
        )

    return price


@pytest.fixture(scope="session")
def ql_american() -> Callable[..., tuple[float, float]]:
    """QuantLib's CRR binomial engine: ``(price, gamma)`` of an American or European option."""
    ql = pytest.importorskip("QuantLib")

    def price(
        S: float,
        K: float,
        days: int,
        r: float,
        q: float,
        sigma: float,
        is_call: bool,
        steps: int = 200,
        american: bool = True,
    ) -> tuple[float, float]:
        today = ql.Date(2, 10, 2026)
        ql.Settings.instance().evaluationDate = today
        dc = ql.Actual365Fixed()
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
        exercise = (
            ql.AmericanExercise(today, today + days)
            if american
            else ql.EuropeanExercise(today + days)
        )
        option = ql.VanillaOption(payoff, exercise)
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, q, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc)),
            ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(today, ql.NullCalendar(), sigma, dc)
            ),
        )
        option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))
        return option.NPV(), option.gamma()

    return price
