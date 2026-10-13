"""Test doubles for code that publishes events."""

import threading

from volsurface.domain.events import Event


class RecordingPublisher:
    """An :class:`~volsurface.domain.ports.EventPublisher` that stores what it receives."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._lock = threading.Lock()

    def publish(self, event: Event) -> None:
        with self._lock:
            self._events.append(event)

    def publish_threadsafe(self, event: Event) -> None:
        self.publish(event)

    @property
    def events(self) -> tuple[Event, ...]:
        with self._lock:
            return tuple(self._events)

    def of_type[E: Event](self, event_type: type[E]) -> list[E]:
        """All recorded events that are instances of ``event_type``, in publish order."""
        return [e for e in self.events if isinstance(e, event_type)]
