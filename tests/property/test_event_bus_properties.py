"""Property tests: bus ordering and conflation guarantees hold for arbitrary event streams."""

import asyncio
from dataclasses import dataclass

from hypothesis import given
from hypothesis import strategies as st

from volsurface.domain.events import Event
from volsurface.domain.ports import DeliveryPolicy
from volsurface.eventing import AsyncEventBus


@dataclass(frozen=True, slots=True, kw_only=True)
class Tick(Event):
    n: int


async def _run(values: list[int], policy: DeliveryPolicy) -> list[int]:
    seen: list[int] = []
    async with AsyncEventBus() as bus:
        bus.subscribe(Tick, lambda e: seen.append(e.n), policy=policy, capacity=len(values) + 1)
        for v in values:
            bus.publish(Tick(n=v))
        await bus.drain()
    return seen


@given(st.lists(st.integers(), max_size=60))
def test_queued_delivery_preserves_order_and_loses_nothing(values: list[int]) -> None:
    assert asyncio.run(_run(values, DeliveryPolicy.QUEUED)) == values


@given(st.lists(st.integers(), min_size=1, max_size=60))
def test_latest_delivery_always_ends_with_the_last_event(values: list[int]) -> None:
    seen = asyncio.run(_run(values, DeliveryPolicy.LATEST))
    assert seen[-1] == values[-1]
    assert len(seen) <= len(values)


@given(st.lists(st.integers(), max_size=40), st.integers(min_value=1, max_value=8))
def test_overflowing_a_bounded_queue_keeps_the_newest_events(
    values: list[int], capacity: int
) -> None:
    async def run() -> list[int]:
        seen: list[int] = []
        bus = AsyncEventBus()
        bus.subscribe(Tick, lambda e: seen.append(e.n), capacity=capacity)
        for v in values:
            bus.publish(Tick(n=v))
        await bus.start()
        await bus.drain()
        await bus.stop()
        return seen

    assert asyncio.run(run()) == values[-capacity:]
