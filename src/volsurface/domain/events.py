"""Domain events: facts about the outside world, published on the event bus.

Events are immutable. ``event_id`` and ``occurred_at`` are excluded from equality so that two
events with the same payload compare equal in tests.
"""

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from volsurface.domain.clock import utc_now
from volsurface.domain.types import ChainSnapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base class of every event."""

    event_id: uuid.UUID = field(default_factory=uuid.uuid4, compare=False, repr=False)
    occurred_at: datetime = field(default_factory=utc_now, compare=False, repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class ChainReceived(Event):
    """A fresh option chain arrived from a market-data source."""

    source: str
    snapshot: ChainSnapshot


@dataclass(frozen=True, slots=True, kw_only=True)
class SpotTicked(Event):
    """The underlying traded or re-quoted."""

    underlying: str
    price: float


class ConnectionState(enum.StrEnum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, kw_only=True)
class ConnectionStateChanged(Event):
    """A market-data source changed connection state."""

    source: str
    state: ConnectionState
    detail: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class HandlerFailed(Event):
    """A subscriber raised while handling an event (published by the bus, never raised)."""

    event_type: str
    handler: str
    error: str
