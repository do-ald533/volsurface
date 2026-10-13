from datetime import UTC, date, datetime

import numpy as np
import pytest

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import (
    ChainSnapshot,
    ExerciseStyle,
    OptionContract,
    OptionQuote,
    Right,
)

ASOF = datetime(2026, 10, 2, 14, 7, 48, tzinfo=UTC)


def _contract(strike: float = 100.0, right: Right = Right.CALL) -> OptionContract:
    return OptionContract("SPY", date(2026, 10, 16), strike, right)


def _snapshot(**overrides: object) -> ChainSnapshot:
    quotes = [
        OptionQuote(_contract(95.0, Right.PUT), 1.0, 1.2, open_interest=500),
        OptionQuote(_contract(105.0, Right.CALL), 0.8, 1.0, vendor_iv=0.21),
    ]
    kwargs: dict[str, object] = {
        "underlying": "SPY",
        "asof": ASOF,
        "spot": 100.0,
        "rate": 0.04,
        "dividend_yield": 0.01,
        "quotes": quotes,
    }
    kwargs.update(overrides)
    return ChainSnapshot.from_quotes(**kwargs)  # type: ignore[arg-type]


class TestRight:
    def test_sign_convention(self) -> None:
        assert Right.CALL.sign == 1
        assert Right.PUT.sign == -1

    def test_is_call(self) -> None:
        assert Right.CALL.is_call
        assert not Right.PUT.is_call


class TestOptionContract:
    @pytest.mark.parametrize("strike", [0.0, -5.0, float("nan"), float("inf")])
    def test_rejects_bad_strike(self, strike: float) -> None:
        with pytest.raises(InvalidInputError):
            OptionContract("SPY", date(2026, 10, 16), strike, Right.CALL)

    def test_rejects_empty_underlying_and_bad_multiplier(self) -> None:
        with pytest.raises(InvalidInputError):
            OptionContract("", date(2026, 10, 16), 100.0, Right.CALL)
        with pytest.raises(InvalidInputError):
            OptionContract("SPY", date(2026, 10, 16), 100.0, Right.CALL, multiplier=0)

    def test_is_frozen(self) -> None:
        contract = _contract()
        with pytest.raises(AttributeError):
            contract.strike = 1.0  # type: ignore[misc]

    def test_orders_by_underlying_then_expiry_then_strike(self) -> None:
        assert _contract(90.0) < _contract(100.0)


class TestOptionQuote:
    def test_mid_and_two_sided(self) -> None:
        quote = OptionQuote(_contract(), 1.0, 1.4)
        assert quote.mid == pytest.approx(1.2)
        assert quote.is_two_sided
        assert not OptionQuote(_contract(), 0.0, 1.4).is_two_sided


class TestChainSnapshot:
    def test_round_trip_through_quotes(self) -> None:
        snap = _snapshot()
        rows = list(snap.iter_quotes())
        assert snap.size == 2
        assert rows[0].contract.right is Right.PUT
        assert rows[0].open_interest == 500
        assert rows[0].vendor_iv is None
        assert rows[1].vendor_iv == pytest.approx(0.21)
        assert rows[1].contract.expiry == date(2026, 10, 16)

    def test_columns_are_read_only_copies(self) -> None:
        strikes = np.array([90.0, 100.0])
        snap = ChainSnapshot(
            underlying="SPY",
            asof=ASOF,
            spot=100.0,
            rate=0.0,
            dividend_yield=0.0,
            expiry=np.array(["2026-10-16"] * 2, dtype="datetime64[D]"),
            strike=strikes,
            is_call=np.array([True, False]),
            bid=np.array([1.0, 1.0]),
            ask=np.array([1.1, 1.1]),
            open_interest=np.array([1.0, 1.0]),
            vendor_iv=np.full(2, np.nan),
        )
        strikes[0] = 1.0
        assert snap.strike[0] == 90.0
        with pytest.raises(ValueError, match="read-only"):
            snap.strike[0] = 5.0

    def test_select_returns_new_snapshot(self) -> None:
        snap = _snapshot()
        picked = snap.select([False, True])
        assert picked.size == 1
        assert picked.strike[0] == 105.0
        assert snap.size == 2

    def test_select_rejects_wrong_mask_shape(self) -> None:
        with pytest.raises(InvalidInputError):
            _snapshot().select([True])

    def test_two_sided_mask_ignores_nan(self) -> None:
        snap = _snapshot(
            quotes=[
                OptionQuote(_contract(95.0), float("nan"), 1.0),
                OptionQuote(_contract(96.0), 1.0, 1.2),
            ]
        )
        assert snap.is_two_sided.tolist() == [False, True]

    @pytest.mark.parametrize(
        "overrides",
        [
            {"asof": datetime(2026, 10, 2, 14, 0)},
            {"spot": 0.0},
            {"spot": float("nan")},
            {"rate": float("inf")},
            {"underlying": ""},
        ],
    )
    def test_rejects_invalid_scalars(self, overrides: dict[str, object]) -> None:
        with pytest.raises(InvalidInputError):
            _snapshot(**overrides)

    def test_rejects_negative_quotes(self) -> None:
        with pytest.raises(InvalidInputError):
            _snapshot(quotes=[OptionQuote(_contract(), -1.0, 1.0)])

    def test_rejects_ragged_columns(self) -> None:
        with pytest.raises(InvalidInputError):
            ChainSnapshot(
                underlying="SPY",
                asof=ASOF,
                spot=100.0,
                rate=0.0,
                dividend_yield=0.0,
                expiry=np.array(["2026-10-16"] * 2, dtype="datetime64[D]"),
                strike=np.array([90.0, 100.0]),
                is_call=np.array([True]),
                bid=np.array([1.0, 1.0]),
                ask=np.array([1.1, 1.1]),
                open_interest=np.array([1.0, 1.0]),
                vendor_iv=np.full(2, np.nan),
            )

    def test_empty_chain_is_allowed(self) -> None:
        snap = _snapshot(quotes=[])
        assert snap.size == 0
        assert snap.exercise is ExerciseStyle.AMERICAN
