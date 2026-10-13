"""Value objects and entities of the option-market domain.

Everything here is immutable. Scalar entities are slotted frozen dataclasses that validate their
invariants on construction; :class:`ChainSnapshot` is columnar (structure of arrays) because the
numerical code downstream works on whole columns, and its arrays are read-only.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Final, Self

import numpy as np
from numpy.typing import NDArray

from volsurface.domain.errors import InvalidInputError

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
DateArray = NDArray[np.datetime64]

DEFAULT_MULTIPLIER: Final = 100


class Right(StrEnum):
    """Option right."""

    CALL = "C"
    PUT = "P"

    @property
    def is_call(self) -> bool:
        return self is Right.CALL

    @property
    def sign(self) -> int:
        """+1 for a call, -1 for a put (the sign convention used by GEX)."""
        return 1 if self is Right.CALL else -1


class ExerciseStyle(StrEnum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True, slots=True, order=True)
class OptionContract:
    """A listed option contract."""

    underlying: str
    expiry: date
    strike: float
    right: Right
    multiplier: int = DEFAULT_MULTIPLIER
    exercise: ExerciseStyle = ExerciseStyle.AMERICAN

    def __post_init__(self) -> None:
        if not self.underlying:
            raise InvalidInputError("underlying must be a non-empty symbol")
        if not (math.isfinite(self.strike) and self.strike > 0):
            raise InvalidInputError(f"strike must be finite and positive, got {self.strike}")
        if self.multiplier <= 0:
            raise InvalidInputError(f"multiplier must be positive, got {self.multiplier}")


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """A two-sided quote for one contract. Missing sides are ``nan``."""

    contract: OptionContract
    bid: float
    ask: float
    open_interest: int | None = None
    vendor_iv: float | None = None

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def is_two_sided(self) -> bool:
        return self.bid > 0 and self.ask > 0


_FLOAT_COLUMNS: Final = ("strike", "bid", "ask", "open_interest", "vendor_iv")


@dataclass(frozen=True, slots=True, eq=False)
class ChainSnapshot:
    """An option chain for one underlying at one instant, stored column-wise.

    ``open_interest`` and ``vendor_iv`` use ``nan`` for "not available". Equality is by identity:
    comparing arrays element-wise is never what a caller of a frozen snapshot wants.
    """

    underlying: str
    asof: datetime
    spot: float
    rate: float
    dividend_yield: float
    expiry: DateArray
    strike: FloatArray
    is_call: BoolArray
    bid: FloatArray
    ask: FloatArray
    open_interest: FloatArray
    vendor_iv: FloatArray
    exercise: ExerciseStyle = ExerciseStyle.AMERICAN
    multiplier: int = DEFAULT_MULTIPLIER

    def __post_init__(self) -> None:
        if not self.underlying:
            raise InvalidInputError("underlying must be a non-empty symbol")
        if self.asof.tzinfo is None:
            raise InvalidInputError("asof must be timezone-aware")
        if not (math.isfinite(self.spot) and self.spot > 0):
            raise InvalidInputError(f"spot must be finite and positive, got {self.spot}")
        if not (math.isfinite(self.rate) and math.isfinite(self.dividend_yield)):
            raise InvalidInputError("rate and dividend_yield must be finite")

        expiry = np.array(self.expiry, dtype="datetime64[D]")
        n = expiry.shape[0] if expiry.ndim == 1 else -1
        if n < 0:
            raise InvalidInputError("chain columns must be one-dimensional")
        columns: dict[str, NDArray[np.generic]] = {"expiry": expiry}
        for name in _FLOAT_COLUMNS:
            columns[name] = np.array(getattr(self, name), dtype=np.float64)
        columns["is_call"] = np.array(self.is_call, dtype=np.bool_)
        for name, column in columns.items():
            if column.shape != (n,):
                raise InvalidInputError(
                    f"column {name!r} has shape {column.shape}, expected ({n},)"
                )
            column.setflags(write=False)
            object.__setattr__(self, name, column)

        if n and not (np.isfinite(self.strike).all() and (self.strike > 0).all()):
            raise InvalidInputError("strikes must be finite and positive")
        for side in (self.bid, self.ask):
            if np.any(side < 0):
                raise InvalidInputError("bid and ask must be non-negative (use nan for missing)")

    @property
    def size(self) -> int:
        return int(self.strike.shape[0])

    @property
    def mid(self) -> FloatArray:
        return 0.5 * (self.bid + self.ask)

    @property
    def is_two_sided(self) -> BoolArray:
        with np.errstate(invalid="ignore"):
            return (self.bid > 0) & (self.ask > 0)

    @property
    def right_sign(self) -> FloatArray:
        """+1.0 for calls and -1.0 for puts, aligned with the other columns."""
        return np.where(self.is_call, 1.0, -1.0)

    def select(self, mask: BoolArray | Sequence[bool]) -> Self:
        """Return a new snapshot holding only the rows where ``mask`` is true."""
        keep = np.asarray(mask, dtype=np.bool_)
        if keep.shape != (self.size,):
            raise InvalidInputError(f"mask has shape {keep.shape}, expected ({self.size},)")
        return type(self)(
            underlying=self.underlying,
            asof=self.asof,
            spot=self.spot,
            rate=self.rate,
            dividend_yield=self.dividend_yield,
            expiry=self.expiry[keep],
            strike=self.strike[keep],
            is_call=self.is_call[keep],
            bid=self.bid[keep],
            ask=self.ask[keep],
            open_interest=self.open_interest[keep],
            vendor_iv=self.vendor_iv[keep],
            exercise=self.exercise,
            multiplier=self.multiplier,
        )

    @classmethod
    def from_quotes(
        cls,
        *,
        underlying: str,
        asof: datetime,
        spot: float,
        rate: float,
        dividend_yield: float,
        quotes: Iterable[OptionQuote],
        exercise: ExerciseStyle = ExerciseStyle.AMERICAN,
        multiplier: int = DEFAULT_MULTIPLIER,
    ) -> Self:
        """Build a snapshot from row-wise quotes (adapters and tests)."""
        rows = list(quotes)
        nan = math.nan
        return cls(
            underlying=underlying,
            asof=asof,
            spot=spot,
            rate=rate,
            dividend_yield=dividend_yield,
            expiry=np.array([q.contract.expiry for q in rows], dtype="datetime64[D]"),
            strike=np.array([q.contract.strike for q in rows], dtype=np.float64),
            is_call=np.array([q.contract.right.is_call for q in rows], dtype=np.bool_),
            bid=np.array([q.bid for q in rows], dtype=np.float64),
            ask=np.array([q.ask for q in rows], dtype=np.float64),
            open_interest=np.array(
                [nan if q.open_interest is None else q.open_interest for q in rows],
                dtype=np.float64,
            ),
            vendor_iv=np.array(
                [nan if q.vendor_iv is None else q.vendor_iv for q in rows], dtype=np.float64
            ),
            exercise=exercise,
            multiplier=multiplier,
        )

    def iter_quotes(self) -> Iterable[OptionQuote]:
        """Row-wise view, mainly for tests and for the chain table."""
        for i in range(self.size):
            oi = self.open_interest[i]
            viv = self.vendor_iv[i]
            yield OptionQuote(
                contract=OptionContract(
                    underlying=self.underlying,
                    expiry=self.expiry[i].astype(object),
                    strike=float(self.strike[i]),
                    right=Right.CALL if self.is_call[i] else Right.PUT,
                    multiplier=self.multiplier,
                    exercise=self.exercise,
                ),
                bid=float(self.bid[i]),
                ask=float(self.ask[i]),
                open_interest=None if math.isnan(oi) else int(oi),
                vendor_iv=None if math.isnan(viv) else float(viv),
            )
