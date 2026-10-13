import asyncio
from pathlib import Path

import pytest

from helpers import small_chain, wait_until
from volsurface.domain.errors import InvalidInputError, ProviderError
from volsurface.domain.events import ChainReceived, ConnectionState, ConnectionStateChanged
from volsurface.eventing import AsyncEventBus
from volsurface.eventing.testing import RecordingPublisher
from volsurface.marketdata.recorder import EventRecorder, NullRecorder
from volsurface.marketdata.recording_store import RecordingStore
from volsurface.marketdata.replay import ReplayProvider


def _write(path: Path, n: int = 4) -> list[float]:
    chains = [small_chain(tick=i, spot=100.0 + i) for i in range(n)]
    RecordingStore(path).write_all(chains, segment_size=2)
    return [c.spot for c in chains]


class TestEventRecorder:
    async def test_records_chains_published_while_active(self, tmp_path: Path) -> None:
        async with AsyncEventBus() as bus:
            recorder = EventRecorder(bus, segment_size=2)
            assert not recorder.active
            bus.publish(ChainReceived(source="x", snapshot=small_chain(tick=0)))
            await bus.drain()
            recorder.start(tmp_path / "rec")
            assert recorder.active
            for i in (1, 2, 3):
                bus.publish(ChainReceived(source="x", snapshot=small_chain(tick=i)))
            await bus.drain()
            assert recorder.stop() == 3
            bus.publish(ChainReceived(source="x", snapshot=small_chain(tick=4)))
            await bus.drain()
        assert RecordingStore(tmp_path / "rec").count() == 3
        assert not recorder.active
        assert recorder.stop() == 0

    async def test_restarting_closes_the_previous_recording(self, tmp_path: Path) -> None:
        async with AsyncEventBus() as bus:
            recorder = EventRecorder(bus)
            recorder.start(tmp_path / "a")
            bus.publish(ChainReceived(source="x", snapshot=small_chain()))
            await bus.drain()
            recorder.start(tmp_path / "b")
            await bus.drain()
            recorder.stop()
        assert RecordingStore(tmp_path / "a").count() == 1

    def test_null_recorder_does_nothing(self, tmp_path: Path) -> None:
        recorder = NullRecorder()
        recorder.start(tmp_path)
        assert not recorder.active
        assert recorder.stop() == 0


class TestReplayProvider:
    async def test_replays_every_snapshot_in_order_then_disconnects(self, tmp_path: Path) -> None:
        spots = _write(tmp_path / "rec")
        provider = ReplayProvider(tmp_path / "rec", speed=0)
        recorder = RecordingPublisher()
        await provider.start(recorder)
        await provider.wait_finished()
        await provider.stop()
        assert [c.snapshot.spot for c in recorder.of_type(ChainReceived)] == spots
        states = [s.state for s in recorder.of_type(ConnectionStateChanged)]
        assert states == [ConnectionState.CONNECTED, ConnectionState.DISCONNECTED]
        assert all(c.source == "replay" for c in recorder.of_type(ChainReceived))

    async def test_loop_repeats_until_stopped(self, tmp_path: Path) -> None:
        _write(tmp_path / "rec", n=2)
        provider = ReplayProvider(tmp_path / "rec", speed=0, loop=True)
        recorder = RecordingPublisher()
        await provider.start(recorder)
        await wait_until(lambda: len(recorder.of_type(ChainReceived)) >= 5)
        await provider.stop()
        assert len(recorder.of_type(ChainReceived)) >= 5

    async def test_pacing_follows_recorded_timestamps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write(tmp_path / "rec", n=3)
        sleeps: list[float] = []
        real_sleep = asyncio.sleep

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)
            await real_sleep(0)

        monkeypatch.setattr("volsurface.marketdata.replay.asyncio.sleep", fake_sleep)
        provider = ReplayProvider(tmp_path / "rec", speed=10.0, max_gap_s=10.0)
        await provider.start(RecordingPublisher())
        await provider.wait_finished()
        assert sleeps == [3.0, 3.0]

    async def test_a_gap_is_capped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write(tmp_path / "rec", n=2)
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr("volsurface.marketdata.replay.asyncio.sleep", fake_sleep)
        provider = ReplayProvider(tmp_path / "rec", speed=0.001, max_gap_s=1.5)
        await provider.start(RecordingPublisher())
        await provider.wait_finished()
        assert sleeps == [1.5]

    async def test_missing_recording_fails_at_start(self, tmp_path: Path) -> None:
        provider = ReplayProvider(tmp_path / "absent")
        with pytest.raises(ProviderError, match="not found"):
            await provider.start(RecordingPublisher())

    async def test_stop_is_idempotent_and_start_twice_is_harmless(self, tmp_path: Path) -> None:
        _write(tmp_path / "rec")
        provider = ReplayProvider(tmp_path / "rec", speed=0)
        recorder = RecordingPublisher()
        await provider.start(recorder)
        await provider.start(recorder)
        await provider.wait_finished()
        await provider.stop()
        await provider.stop()
        assert len(recorder.of_type(ChainReceived)) == 4

    def test_rejects_negative_settings(self, tmp_path: Path) -> None:
        with pytest.raises(InvalidInputError):
            ReplayProvider(tmp_path, speed=-1.0)
        with pytest.raises(InvalidInputError):
            ReplayProvider(tmp_path, max_gap_s=-1.0)


async def test_record_then_replay_reproduces_the_event_stream(tmp_path: Path) -> None:
    async with AsyncEventBus() as bus:
        recorder = EventRecorder(bus, segment_size=2)
        recorder.start(tmp_path / "rec")
        originals = [small_chain(tick=i, spot=99.0 + i) for i in range(3)]
        for chain in originals:
            bus.publish(ChainReceived(source="live", snapshot=chain))
        await bus.drain()
        recorder.stop()

    replayed = RecordingPublisher()
    provider = ReplayProvider(tmp_path / "rec", speed=0)
    await provider.start(replayed)
    await provider.wait_finished()
    got = replayed.of_type(ChainReceived)
    assert [c.snapshot.spot for c in got] == [c.spot for c in originals]
    assert [c.snapshot.asof for c in got] == [c.asof for c in originals]
