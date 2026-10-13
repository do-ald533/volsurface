"""Event recorder: persists every ``ChainReceived`` seen on the bus.

Because providers publish the same events whether data is live, synthetic or replayed, recording
is a plain subscriber and replay is a plain publisher: the event log *is* the recording.
:class:`NullRecorder` is the Null Object used while recording is off.
"""

from pathlib import Path
from typing import Protocol

from volsurface.domain.events import ChainReceived
from volsurface.domain.ports import EventSubscriber, Subscription
from volsurface.marketdata.recording_store import RecordingStore, RecordingWriter


class Recorder(Protocol):
    @property
    def active(self) -> bool: ...

    def start(self, path: Path) -> None: ...

    def stop(self) -> int:
        """Stop recording and return the number of snapshots written."""
        ...


class NullRecorder:
    active = False

    def start(self, path: Path) -> None:
        return None

    def stop(self) -> int:
        return 0


class EventRecorder:
    """Subscribes to ``ChainReceived`` while active and appends each snapshot to a recording."""

    def __init__(self, bus: EventSubscriber, *, segment_size: int = 50) -> None:
        self._bus = bus
        self._segment_size = segment_size
        self._writer: RecordingWriter | None = None
        self._subscription: Subscription | None = None

    @property
    def active(self) -> bool:
        return self._writer is not None

    def start(self, path: Path) -> None:
        if self._writer is not None:
            self.stop()
        self._writer = RecordingStore(path).writer(self._segment_size)
        self._subscription = self._bus.subscribe(
            ChainReceived, self._on_chain, name="recorder.chain"
        )

    def stop(self) -> int:
        if self._subscription is not None:
            self._subscription.cancel()
            self._subscription = None
        writer, self._writer = self._writer, None
        if writer is None:
            return 0
        writer.close()
        return writer.snapshots_written

    def _on_chain(self, event: ChainReceived) -> None:
        if self._writer is not None:
            self._writer.append(event.snapshot)
