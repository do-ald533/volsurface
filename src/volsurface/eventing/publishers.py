"""Small publisher implementations that are not the real bus."""

from volsurface.domain.events import Event


class NullPublisher:
    """Null Object: accepts and discards every event.

    Lets components take a publisher unconditionally instead of branching on ``None``.
    """

    def publish(self, event: Event) -> None:
        return None

    def publish_threadsafe(self, event: Event) -> None:
        return None
