"""Replay of a recording as a stream of events.

The provider walks the recorded snapshots and publishes each one as ``ChainReceived``. Pacing
follows the recorded timestamps divided by ``speed`` (``speed=0`` means "as fast as possible", used
by tests and batch jobs); a single gap is capped at ``max_gap_s`` so a recording that spans a
lunch break does not stall the demo.
"""

import asyncio
from pathlib import Path

from volsurface.domain.errors import InvalidInputError
from volsurface.domain.events import ChainReceived, ConnectionState, ConnectionStateChanged
from volsurface.domain.ports import EventPublisher
from volsurface.domain.types import ChainSnapshot
from volsurface.marketdata.recording_store import RecordingStore


class ReplayProvider:
    name = "replay"

    def __init__(
        self,
        path: Path,
        *,
        speed: float = 1.0,
        loop: bool = False,
        max_gap_s: float = 2.0,
    ) -> None:
        if speed < 0 or max_gap_s < 0:
            raise InvalidInputError("speed and max_gap_s must be non-negative")
        self._store = RecordingStore(path)
        self._speed = speed
        self._loop = loop
        self._max_gap_s = max_gap_s
        self._task: asyncio.Task[None] | None = None

    async def start(self, publisher: EventPublisher) -> None:
        if self._task is not None:
            return
        self._store.files()
        self._task = asyncio.get_running_loop().create_task(
            self._run(publisher), name="replay-provider"
        )

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def wait_finished(self) -> None:
        if self._task is not None:
            await asyncio.shield(self._task)

    async def _run(self, publisher: EventPublisher) -> None:
        publisher.publish(ConnectionStateChanged(source=self.name, state=ConnectionState.CONNECTED))
        previous: ChainSnapshot | None = None
        while True:
            for snapshot in self._store.read():
                await self._pace(previous, snapshot)
                publisher.publish(ChainReceived(source=self.name, snapshot=snapshot))
                previous = snapshot
            if not self._loop:
                break
        publisher.publish(
            ConnectionStateChanged(
                source=self.name, state=ConnectionState.DISCONNECTED, detail="end of recording"
            )
        )

    async def _pace(self, previous: ChainSnapshot | None, current: ChainSnapshot) -> None:
        if self._speed == 0:
            await asyncio.sleep(0)
            return
        if previous is None:
            return
        gap = (current.asof - previous.asof).total_seconds() / self._speed
        await asyncio.sleep(min(max(gap, 0.0), self._max_gap_s))
