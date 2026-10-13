from datetime import UTC, date, datetime

import numpy as np
import pytest

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import ChainSnapshot, OptionContract, OptionQuote, Right
from volsurface.pricing.quote_filter import (
    AcceptAll,
    HasOpenInterest,
    MaxDaysToExpiry,
    MinTimeValue,
    MoneynessBand,
    TwoSided,
    gex_quote_spec,
    surface_quote_spec,
)

ASOF = datetime(2026, 10, 2, 14, 0, tzinfo=UTC)
NEAR, FAR = date(2026, 10, 16), date(2027, 3, 19)


def _q(
    strike: float, right: Right, bid: float, ask: float, expiry: date = NEAR, oi: int | None = 10
) -> OptionQuote:
    return OptionQuote(OptionContract("SPY", expiry, strike, right), bid, ask, open_interest=oi)


@pytest.fixture
def chain() -> ChainSnapshot:
    return ChainSnapshot.from_quotes(
        underlying="SPY",
        asof=ASOF,
        spot=100.0,
        rate=0.03,
        dividend_yield=0.0,
        quotes=[
            _q(100.0, Right.CALL, 2.0, 2.2),
            _q(100.0, Right.PUT, 0.0, 2.0),
            _q(90.0, Right.CALL, 10.0, 10.02),
            _q(110.0, Right.CALL, 0.5, 0.6, oi=None),
            _q(100.0, Right.CALL, 5.0, 5.2, expiry=FAR),
            _q(95.0, Right.PUT, 0.9, 1.0, oi=0),
        ],
    )


def test_two_sided(chain: ChainSnapshot) -> None:
    assert TwoSided().mask(chain).tolist() == [True, False, True, True, True, True]


def test_open_interest_requires_known_positive_values(chain: ChainSnapshot) -> None:
    assert HasOpenInterest().mask(chain).tolist() == [True, True, True, False, True, False]


def test_min_time_value_drops_intrinsic_priced_options(chain: ChainSnapshot) -> None:
    mask = MinTimeValue(0.05).mask(chain)
    assert not mask[2]
    assert mask[0]


def test_max_days_uses_the_new_york_date(chain: ChainSnapshot) -> None:
    mask = MaxDaysToExpiry(30).mask(chain)
    assert mask.tolist() == [True, True, True, True, False, True]


def test_moneyness_band(chain: ChainSnapshot) -> None:
    assert MoneynessBand(0.95, 1.05).mask(chain).tolist() == [True, True, False, False, True, True]


def test_composition_with_and_or_not(chain: ChainSnapshot) -> None:
    both = TwoSided() & MaxDaysToExpiry(30)
    assert both.mask(chain).tolist() == [True, False, True, True, False, True]
    either = ~TwoSided() | ~MaxDaysToExpiry(30)
    assert either.mask(chain).tolist() == [False, True, False, False, True, False]
    assert AcceptAll().mask(chain).all()


def test_default_policies(chain: ChainSnapshot) -> None:
    assert chain.select(surface_quote_spec(30).mask(chain)).size == 3
    assert chain.select(gex_quote_spec(30).mask(chain)).size == 1


def test_masks_are_boolean_arrays_of_chain_length(chain: ChainSnapshot) -> None:
    for spec in (TwoSided(), HasOpenInterest(), MinTimeValue(), MaxDaysToExpiry(5)):
        mask = spec.mask(chain)
        assert mask.dtype == np.bool_
        assert mask.shape == (chain.size,)


def test_invalid_parameters() -> None:
    with pytest.raises(InvalidInputError):
        MaxDaysToExpiry(-1)
    with pytest.raises(InvalidInputError):
        MoneynessBand(1.1, 0.9)
