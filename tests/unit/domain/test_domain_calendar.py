from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from volsurface.domain.calendar import (
    MARKET_TZ,
    MINUTES_PER_YEAR,
    expiry_instant,
    years_to_expiry,
)
from volsurface.domain.clock import ManualClock, SystemClock
from volsurface.domain.errors import InvalidInputError


def _days(*iso: str) -> np.ndarray:
    return np.array(iso, dtype="datetime64[D]")


def test_same_day_expiry_counts_minutes_to_the_close() -> None:
    asof = datetime(2026, 10, 2, 10, 7, 48, tzinfo=MARKET_TZ)
    (t,) = years_to_expiry(asof, _days("2026-10-02"))
    expected_minutes = 5 * 60 + 52 + 12 / 60
    assert t == pytest.approx(expected_minutes / MINUTES_PER_YEAR)


def test_after_the_close_is_floored_at_five_minutes() -> None:
    asof = datetime(2026, 10, 2, 17, 0, tzinfo=MARKET_TZ)
    (t,) = years_to_expiry(asof, _days("2026-10-02"))
    assert t == pytest.approx(5 / MINUTES_PER_YEAR)


def test_daylight_saving_change_is_respected() -> None:
    asof = datetime(2026, 10, 30, 16, 0, tzinfo=MARKET_TZ)
    (t,) = years_to_expiry(asof, _days("2026-11-02"))
    assert t * MINUTES_PER_YEAR == pytest.approx(73 * 60)


def test_vectorised_over_repeated_and_unsorted_expiries() -> None:
    asof = datetime(2026, 10, 2, 9, 30, tzinfo=MARKET_TZ)
    days = _days("2026-10-16", "2026-10-09", "2026-10-16", "2026-10-09")
    t = years_to_expiry(asof, days)
    assert t.shape == (4,)
    assert t[0] == t[2]
    assert t[1] == t[3]
    assert t[0] > t[1] > 0


def test_naive_asof_is_rejected() -> None:
    with pytest.raises(InvalidInputError):
        years_to_expiry(datetime(2026, 10, 2, 9, 30), _days("2026-10-16"))


def test_expiry_instant_is_the_market_close() -> None:
    instant = expiry_instant(date(2026, 10, 16))
    assert instant.tzinfo == MARKET_TZ
    assert (instant.hour, instant.minute) == (16, 0)


class TestClocks:
    def test_system_clock_is_timezone_aware(self) -> None:
        assert SystemClock().now().tzinfo is not None

    def test_manual_clock_only_moves_when_told(self) -> None:
        start = datetime(2026, 10, 2, 9, 30, tzinfo=ZoneInfo("UTC"))
        clock = ManualClock(start)
        assert clock.now() == start
        clock.advance(timedelta(minutes=5))
        assert clock.now() == start + timedelta(minutes=5)

    def test_manual_clock_rejects_backwards_and_naive(self) -> None:
        clock = ManualClock(datetime(2026, 10, 2, tzinfo=ZoneInfo("UTC")))
        with pytest.raises(InvalidInputError):
            clock.advance(timedelta(seconds=-1))
        with pytest.raises(InvalidInputError):
            clock.set(datetime(2026, 10, 2))
        with pytest.raises(InvalidInputError):
            ManualClock(datetime(2026, 10, 2))
