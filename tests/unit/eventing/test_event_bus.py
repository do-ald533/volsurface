import asyncio
import threading
from dataclasses import dataclass

import pytest

from volsurface.domain.errors import BusNotRunningError, InvalidInputError
from volsurface.domain.events import Event, HandlerFailed
from volsurface.domain.ports import DeliveryPolicy, EventBus, EventPublisher, EventSubscriber
from volsurface.eventing import AsyncEventBus, NullPublisher
from volsurface.eventing.testing import RecordingPublisher


@dataclass(frozen=True, slots=True, kw_only=True)
class Ping(Event):
    n: int


@dataclass(frozen=True, slots=True, kw_only=True)
class SpecialPing(Ping):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class Other(Event):
    pass


async def test_delivers_to_matching_subscribers_only() -> None:
    seen: list[int] = []
    async with AsyncEventBus() as bus:
        bus.subscribe(Ping, lambda e: seen.append(e.n))
        bus.publish(Ping(n=1))
        bus.publish(Other())
        bus.publish(Ping(n=2))
        await bus.drain()
    assert seen == [1, 2]


async def test_base_class_subscription_receives_subclasses() -> None:
    kinds: list[str] = []
    async with AsyncEventBus() as bus:
        bus.subscribe(Ping, lambda e: kinds.append(type(e).__name__))
        bus.subscribe(Event, lambda e: kinds.append("any:" + type(e).__name__))
        bus.publish(SpecialPing(n=1))
        await bus.drain()
    assert sorted(kinds) == ["SpecialPing", "any:SpecialPing"]


async def test_supports_async_and_sync_handlers() -> None:
    seen: list[str] = []

    async def slow(e: Ping) -> None:
        await asyncio.sleep(0.01)
        seen.append(f"async{e.n}")

    async with AsyncEventBus() as bus:
        bus.subscribe(Ping, slow)
        bus.subscribe(Ping, lambda e: seen.append(f"sync{e.n}"))
        bus.publish(Ping(n=1))
        await bus.drain()
    assert sorted(seen) == ["async1", "sync1"]


async def test_fifo_order_per_subscription() -> None:
    seen: list[int] = []
    async with AsyncEventBus() as bus:
        bus.subscribe(Ping, lambda e: seen.append(e.n))
        for i in range(50):
            bus.publish(Ping(n=i))
        await bus.drain()
    assert seen == list(range(50))


async def test_slow_subscriber_does_not_block_a_fast_one() -> None:
    gate = asyncio.Event()
    fast: list[int] = []

    async def blocked(_: Ping) -> None:
        await gate.wait()

    async with AsyncEventBus() as bus:
        bus.subscribe(Ping, blocked)
        bus.subscribe(Ping, lambda e: fast.append(e.n))
        bus.publish(Ping(n=1))
        bus.publish(Ping(n=2))
        for _ in range(5):
            await asyncio.sleep(0)
        assert fast == [1, 2]
        gate.set()
        await bus.drain()


async def test_latest_policy_conflates_to_the_newest_event() -> None:
    gate = asyncio.Event()
    seen: list[int] = []

    async def handler(e: Ping) -> None:
        await gate.wait()
        seen.append(e.n)

    async with AsyncEventBus() as bus:
        sub = bus.subscribe(Ping, handler, policy=DeliveryPolicy.LATEST)
        bus.publish(Ping(n=0))
        await asyncio.sleep(0)
        for i in range(1, 6):
            bus.publish(Ping(n=i))
        gate.set()
        await bus.drain()
        stats = bus.stats()
    assert seen == [0, 5]
    assert sub.name in {s.name for s in stats}
    assert next(s for s in stats if s.name == sub.name).dropped == 4


async def test_queued_policy_drops_oldest_when_full() -> None:
    gate = asyncio.Event()
    seen: list[int] = []

    async def handler(e: Ping) -> None:
        await gate.wait()
        seen.append(e.n)

    async with AsyncEventBus() as bus:
        bus.subscribe(Ping, handler, capacity=3)
        bus.publish(Ping(n=0))
        await asyncio.sleep(0)
        for i in range(1, 8):
            bus.publish(Ping(n=i))
        gate.set()
        await bus.drain()
    assert seen == [0, 5, 6, 7]


async def test_handler_errors_are_isolated_and_reported() -> None:
    good: list[int] = []
    failures: list[HandlerFailed] = []

    def bad(_: Ping) -> None:
        raise RuntimeError("boom")

    async with AsyncEventBus() as bus:
        bus.subscribe(Ping, bad, name="bad-handler")
        bus.subscribe(Ping, lambda e: good.append(e.n))
        bus.subscribe(HandlerFailed, failures.append)
        bus.publish(Ping(n=1))
        bus.publish(Ping(n=2))
        await bus.drain()
        failed = {s.name: s.failed for s in bus.stats()}
    assert good == [1, 2]
    assert [f.handler for f in failures] == ["bad-handler", "bad-handler"]
    assert "boom" in failures[0].error
    assert failed["bad-handler"] == 2


async def test_failing_failure_handler_does_not_loop() -> None:
    calls = 0

    def bad(_: Ping) -> None:
        raise RuntimeError("first")

    def worse(_: HandlerFailed) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("second")

    async with AsyncEventBus() as bus:
        bus.subscribe(Ping, bad)
        bus.subscribe(HandlerFailed, worse)
        bus.publish(Ping(n=1))
        await bus.drain(timeout=2)
    assert calls == 1


async def test_cancel_stops_delivery_and_is_idempotent() -> None:
    seen: list[int] = []
    async with AsyncEventBus() as bus:
        sub = bus.subscribe(Ping, lambda e: seen.append(e.n))
        bus.publish(Ping(n=1))
        await bus.drain()
        assert sub.active
        sub.cancel()
        sub.cancel()
        assert not sub.active
        bus.publish(Ping(n=2))
        await bus.drain()
    assert seen == [1]


async def test_drain_waits_for_events_published_by_handlers() -> None:
    seen: list[str] = []
    async with AsyncEventBus() as bus:

        def relay(e: Ping) -> None:
            seen.append("ping")
            bus.publish(Other())

        bus.subscribe(Ping, relay)
        bus.subscribe(Other, lambda _: seen.append("other"))
        bus.publish(Ping(n=1))
        await bus.drain()
    assert seen == ["ping", "other"]


async def test_publish_threadsafe_from_another_thread() -> None:
    seen: list[int] = []
    async with AsyncEventBus() as bus:
        bus.subscribe(Ping, lambda e: seen.append(e.n))

        def worker() -> None:
            for i in range(20):
                bus.publish_threadsafe(Ping(n=i))

        thread = threading.Thread(target=worker)
        thread.start()
        await asyncio.to_thread(thread.join)
        await asyncio.sleep(0)
        await bus.drain()
    assert seen == list(range(20))


async def test_publish_after_stop_raises() -> None:
    bus = AsyncEventBus()
    await bus.start()
    await bus.stop()
    with pytest.raises(BusNotRunningError):
        bus.publish(Ping(n=1))
    with pytest.raises(BusNotRunningError):
        bus.subscribe(Ping, lambda _: None)


async def test_publish_threadsafe_requires_a_running_bus() -> None:
    with pytest.raises(BusNotRunningError):
        AsyncEventBus().publish_threadsafe(Ping(n=1))


async def test_events_published_before_start_are_delivered_after() -> None:
    seen: list[int] = []
    bus = AsyncEventBus()
    bus.subscribe(Ping, lambda e: seen.append(e.n))
    bus.publish(Ping(n=7))
    await bus.start()
    await bus.drain()
    await bus.stop()
    assert seen == [7]


async def test_rejects_invalid_capacity() -> None:
    async with AsyncEventBus() as bus:
        with pytest.raises(InvalidInputError):
            bus.subscribe(Ping, lambda _: None, capacity=0)


async def test_stop_without_drain_discards_pending_events() -> None:
    gate = asyncio.Event()
    seen: list[int] = []

    async def handler(e: Ping) -> None:
        await gate.wait()
        seen.append(e.n)

    bus = AsyncEventBus()
    await bus.start()
    bus.subscribe(Ping, handler)
    for i in range(3):
        bus.publish(Ping(n=i))
    await asyncio.sleep(0)
    await bus.stop(drain=False)
    assert seen == []


def test_implementations_satisfy_the_ports() -> None:
    bus = AsyncEventBus()
    assert isinstance(bus, EventPublisher)
    assert isinstance(bus, EventSubscriber)
    assert isinstance(NullPublisher(), EventPublisher)
    assert isinstance(RecordingPublisher(), EventPublisher)
    _: EventBus = bus


def test_recording_publisher_filters_by_type() -> None:
    rec = RecordingPublisher()
    rec.publish(Ping(n=1))
    rec.publish_threadsafe(Other())
    assert [e.n for e in rec.of_type(Ping)] == [1]
    assert len(rec.events) == 2


def test_null_publisher_swallows_everything() -> None:
    NullPublisher().publish(Ping(n=1))
    NullPublisher().publish_threadsafe(Ping(n=1))
