"""An in-process, typed, asyncio event bus.

Design notes
------------
* **Observer / pub-sub.** Producers publish immutable :class:`~volsurface.domain.events.Event`
  objects; consumers subscribe by event class. Subscribing to a base class also receives its
  subclasses (the publish path walks the event's MRO).
* **One mailbox and one task per subscription.** A slow subscriber therefore never delays a fast
  one, and each subscriber sees events in publish order (FIFO per subscription).
* **Back-pressure without blocking publishers.** ``publish`` never waits. A subscription's
  :class:`~volsurface.domain.ports.DeliveryPolicy` decides what happens when it falls behind:
  ``QUEUED`` keeps a bounded FIFO (dropping the oldest on overflow), ``LATEST`` conflates to the
  newest event. Drops are counted and visible through :meth:`AsyncEventBus.stats`.
* **Failure isolation.** A handler exception is logged and republished as
  :class:`~volsurface.domain.events.HandlerFailed`; it never reaches the publisher or other
  subscribers. A failing ``HandlerFailed`` handler is only logged, so errors cannot loop.
* **Deterministic tests.** :meth:`AsyncEventBus.drain` waits until every queued event has been
  handled, including events published by handlers.
"""

import asyncio
import inspect
import logging
from collections import deque
from collections.abc import Awaitable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from volsurface.domain.errors import BusNotRunningError, InvalidInputError
from volsurface.domain.events import Event, HandlerFailed
from volsurface.domain.ports import DeliveryPolicy, Handler

_log = logging.getLogger(__name__)


class _Mailbox:
    """A bounded event queue with a conflation option, used by exactly one consumer task."""

    __slots__ = ("_capacity", "_closed", "_items", "_policy", "_wakeup")

    def __init__(self, policy: DeliveryPolicy, capacity: int) -> None:
        self._policy = policy
        self._capacity = capacity
        self._items: deque[Event] = deque()
        self._wakeup = asyncio.Event()
        self._closed = False

    def __len__(self) -> int:
        return len(self._items)

    def put(self, event: Event) -> int:
        """Enqueue ``event``; return how many older events were discarded to make room."""
        discarded = 0
        if self._policy is DeliveryPolicy.LATEST:
            discarded = len(self._items)
            self._items.clear()
        elif len(self._items) >= self._capacity:
            self._items.popleft()
            discarded = 1
        self._items.append(event)
        self._wakeup.set()
        return discarded

    async def get(self) -> Event | None:
        """Next event, or ``None`` once the mailbox is closed and empty."""
        while not self._items:
            if self._closed:
                return None
            self._wakeup.clear()
            await self._wakeup.wait()
        return self._items.popleft()

    def close(self) -> int:
        """Close the mailbox; return how many undelivered events were discarded."""
        discarded = len(self._items)
        self._items.clear()
        self._closed = True
        self._wakeup.set()
        return discarded


@dataclass(frozen=True, slots=True)
class SubscriptionStats:
    """Counters for one subscription, for the diagnostics panel and tests."""

    name: str
    event_type: str
    policy: DeliveryPolicy
    delivered: int
    dropped: int
    failed: int
    queued: int


class _Subscription:
    """Concrete :class:`~volsurface.domain.ports.Subscription`."""

    def __init__(
        self,
        *,
        bus: "AsyncEventBus",
        event_type: type[Event],
        handler: Handler[Any],
        policy: DeliveryPolicy,
        capacity: int,
        name: str,
    ) -> None:
        self._bus = bus
        self.event_type = event_type
        self.handler = handler
        self.policy = policy
        self.mailbox = _Mailbox(policy, capacity)
        self._name = name
        self.task: asyncio.Task[None] | None = None
        self.active = True
        self.delivered = 0
        self.dropped = 0
        self.failed = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def pending(self) -> int:
        return len(self.mailbox)

    def cancel(self) -> None:
        self._bus._cancel(self)

    def stats(self) -> SubscriptionStats:
        return SubscriptionStats(
            name=self._name,
            event_type=self.event_type.__name__,
            policy=self.policy,
            delivered=self.delivered,
            dropped=self.dropped,
            failed=self.failed,
            queued=len(self.mailbox),
        )


class AsyncEventBus:
    """The default :class:`~volsurface.domain.ports.EventBus` implementation.

    Use as ``async with AsyncEventBus() as bus:``. Subscriptions may be registered before the bus
    starts; their dispatcher tasks begin when it does.
    """

    def __init__(self) -> None:
        self._by_type: dict[type[Event], list[_Subscription]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._closed = False
        self._pending = 0
        self._idle = asyncio.Event()
        self._idle.set()

    async def start(self) -> None:
        if self._closed:
            raise BusNotRunningError("a stopped bus cannot be restarted")
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        self._running = True
        for sub in self._subscriptions():
            self._spawn(sub)

    async def stop(self, *, drain: bool = True) -> None:
        """Stop dispatching. By default, first wait for queued events to be handled."""
        if self._closed:
            return
        if drain and self._running:
            await self.drain()
        self._running = False
        self._closed = True
        tasks = []
        for sub in self._subscriptions():
            self._adjust(-sub.mailbox.close())
            sub.active = False
            if sub.task is not None:
                tasks.append(sub.task)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._by_type.clear()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.stop(drain=exc_type is None)

    async def drain(self, timeout: float | None = None) -> None:
        """Wait until nothing is queued or being handled (events published by handlers included)."""
        if not self._running:
            raise BusNotRunningError("drain() needs a running bus")
        await asyncio.wait_for(self._idle.wait(), timeout)

    def subscribe[E: Event](
        self,
        event_type: type[E],
        handler: Handler[E],
        *,
        policy: DeliveryPolicy = DeliveryPolicy.QUEUED,
        capacity: int = 1024,
        name: str | None = None,
    ) -> _Subscription:
        if self._closed:
            raise BusNotRunningError("cannot subscribe to a stopped bus")
        if capacity < 1:
            raise InvalidInputError(f"capacity must be >= 1, got {capacity}")
        label: str = name or str(getattr(handler, "__qualname__", repr(handler)))
        sub = _Subscription(
            bus=self,
            event_type=event_type,
            handler=handler,
            policy=policy,
            capacity=capacity,
            name=label,
        )
        self._by_type.setdefault(event_type, []).append(sub)
        if self._running:
            self._spawn(sub)
        return sub

    def publish(self, event: Event) -> None:
        if self._closed:
            raise BusNotRunningError("cannot publish to a stopped bus")
        for cls in type(event).__mro__:
            for sub in tuple(self._by_type.get(cls, ())):
                discarded = sub.mailbox.put(event)
                sub.dropped += discarded
                self._adjust(1 - discarded)

    def publish_threadsafe(self, event: Event) -> None:
        loop = self._loop
        if loop is None or not self._running:
            raise BusNotRunningError("publish_threadsafe() needs a running bus")
        loop.call_soon_threadsafe(self.publish, event)

    def stats(self) -> tuple[SubscriptionStats, ...]:
        return tuple(sub.stats() for sub in self._subscriptions())

    def _subscriptions(self) -> list[_Subscription]:
        return [sub for subs in self._by_type.values() for sub in subs]

    def _adjust(self, delta: int) -> None:
        self._pending += delta
        if self._pending <= 0:
            self._pending = 0
            self._idle.set()
        else:
            self._idle.clear()

    def _spawn(self, sub: _Subscription) -> None:
        if sub.task is None:
            sub.task = asyncio.get_running_loop().create_task(
                self._dispatch(sub), name=f"event-bus:{sub.name}"
            )

    def _cancel(self, sub: _Subscription) -> None:
        if not sub.active:
            return
        sub.active = False
        subs = self._by_type.get(sub.event_type, [])
        if sub in subs:
            subs.remove(sub)
        self._adjust(-sub.mailbox.close())
        if sub.task is not None and sub.task is not asyncio.current_task():
            sub.task.cancel()

    async def _dispatch(self, sub: _Subscription) -> None:
        while True:
            event = await sub.mailbox.get()
            if event is None:
                return
            try:
                result = sub.handler(event)
                if inspect.isawaitable(result):
                    await _await(result)
                sub.delivered += 1
            except Exception as exc:
                sub.failed += 1
                self._report_failure(sub, event, exc)
            finally:
                self._adjust(-1)

    def _report_failure(self, sub: _Subscription, event: Event, exc: Exception) -> None:
        _log.error("event handler %s failed on %s", sub.name, type(event).__name__, exc_info=exc)
        if isinstance(event, HandlerFailed) or self._closed:
            return
        self.publish(
            HandlerFailed(event_type=type(event).__name__, handler=sub.name, error=repr(exc))
        )


async def _await(awaitable: Awaitable[Any]) -> None:
    await awaitable
