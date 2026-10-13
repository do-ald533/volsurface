"""Market calendar helpers: time to expiry on a fractional-year basis.

Convention (taken from the GEX notebook this project re-implements): calendar time, a year is
365 * 24 * 60 minutes, options expire at the 16:00 America/New_York close of their expiry date, and
time to expiry never drops below five minutes so that the pricing formulas stay well defined on
expiry day.
"""

from datetime import UTC, date, datetime, time
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.types import DateArray, FloatArray

MARKET_TZ: Final = ZoneInfo("America/New_York")
MARKET_CLOSE: Final = time(16, 0)
MINUTES_PER_YEAR: Final = 365 * 24 * 60
MIN_MINUTES_TO_EXPIRY: Final = 5


def expiry_instant(expiry: date) -> datetime:
    """The moment an option expiring on ``expiry`` stops trading."""
    return datetime.combine(expiry, MARKET_CLOSE, tzinfo=MARKET_TZ)


def years_to_expiry(asof: datetime, expiries: DateArray) -> FloatArray:
    """Fractional years from ``asof`` to each expiry, floored at five minutes.

    Only the distinct expiry dates are converted to datetimes, so this is cheap for chains with
    thousands of rows and a few dozen expiries.
    """
    if asof.tzinfo is None:
        raise InvalidInputError("asof must be timezone-aware")
    asof_utc = asof.astimezone(UTC)
    days = np.asarray(expiries, dtype="datetime64[D]")
    unique, inverse = np.unique(days, return_inverse=True)
    minutes = np.array(
        [
            (expiry_instant(d.astype(object)).astimezone(UTC) - asof_utc).total_seconds() / 60.0
            for d in unique
        ],
        dtype=np.float64,
    )
    clipped = np.maximum(minutes, MIN_MINUTES_TO_EXPIRY) / MINUTES_PER_YEAR
    return clipped[inverse.reshape(days.shape)]
