"""Clock implementations (the :class:`~volsurface.domain.ports.Clock` port is in ``ports``)."""

from datetime import UTC, datetime, timedelta

from volsurface.domain.errors import InvalidInputError


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class SystemClock:
    """Wall-clock time in UTC."""

    def now(self) -> datetime:
        return utc_now()


class ManualClock:
    """A clock that only moves when told to; for deterministic tests and replay."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise InvalidInputError("start must be timezone-aware")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        if delta < timedelta(0):
            raise InvalidInputError("a clock cannot move backwards")
        self._now += delta

    def set(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise InvalidInputError("instant must be timezone-aware")
        self._now = instant
