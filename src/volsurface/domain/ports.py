"""Ports: the interfaces the core depends on and the outer layers implement.

All of them are structural (:class:`typing.Protocol`), so adapters do not inherit from anything in
the core. Interfaces are kept small on purpose (interface segregation): a component that only
publishes asks for an :class:`EventPublisher`, not for a whole bus.
"""

import enum
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Protocol, runtime_checkable

from volsurface.domain.events import Event

type Handler[E: Event] = Callable[[E], Awaitable[None] | None]
"""A subscriber callback: synchronous or ``async``, receiving one event."""


class DeliveryPolicy(enum.Enum):
    """What a subscription does when events arrive faster than its handler runs."""

    QUEUED = "queued"
    """Keep every event in FIFO order up to ``capacity``; on overflow drop the oldest."""

    LATEST = "latest"
    """Conflate: only the newest pending event is kept. Right for "current state" streams."""


@runtime_checkable
class Subscription(Protocol):
    """Handle returned by :meth:`EventSubscriber.subscribe`."""

    @property
    def name(self) -> str: ...

    @property
    def active(self) -> bool: ...

    @property
    def pending(self) -> int:
        """Events waiting in this subscription's queue (not counting one being handled)."""
        ...

    def cancel(self) -> None:
        """Stop delivery and discard pending events. Idempotent."""
        ...


@runtime_checkable
class EventPublisher(Protocol):
    """The publishing half of an event bus."""

    def publish(self, event: Event) -> None:
        """Hand ``event`` to every matching subscriber without blocking."""
        ...

    def publish_threadsafe(self, event: Event) -> None:
        """Like :meth:`publish`, but callable from any thread."""
        ...


@runtime_checkable
class EventSubscriber(Protocol):
    """The subscribing half of an event bus."""

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: Handler[E],
        *,
        policy: DeliveryPolicy = DeliveryPolicy.QUEUED,
        capacity: int = 1024,
        name: str | None = None,
    ) -> Subscription:
        """Deliver every ``event_type`` (or subclass) event to ``handler``."""
        ...


class EventBus(EventPublisher, EventSubscriber, Protocol):
    """A full bus: both halves."""


@runtime_checkable
class Clock(Protocol):
    """Source of the current time, injected so time-dependent logic is testable."""

    def now(self) -> datetime:
        """Current timezone-aware time."""
        ...


@runtime_checkable
class MarketDataProvider(Protocol):
    """A source of market data that pushes events onto a bus (event-driven, not polled).

    Implementations: replay, synthetic and Interactive Brokers. The contract test suite in
    ``tests/contract`` runs against every implementation.
    """

    @property
    def name(self) -> str: ...

    async def start(self, publisher: EventPublisher) -> None:
        """Begin producing :class:`~volsurface.domain.events.ChainReceived` events."""
        ...

    async def stop(self) -> None:
        """Stop producing events and release resources. Idempotent."""
        ...
