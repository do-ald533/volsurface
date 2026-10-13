from pathlib import Path

import numpy as np
import polars as pl
import pytest

from helpers import small_chain
from volsurface.domain.errors import InvalidInputError, ProviderError
from volsurface.domain.types import ChainSnapshot, ExerciseStyle
from volsurface.marketdata.recording_store import (
    RecordingStore,
    RecordingWriter,
    frame_to_snapshots,
    snapshot_to_frame,
)


def _assert_same(a: ChainSnapshot, b: ChainSnapshot) -> None:
    assert a.underlying == b.underlying
    assert a.asof == b.asof
    assert (a.spot, a.rate, a.dividend_yield) == (b.spot, b.rate, b.dividend_yield)
    assert a.exercise == b.exercise
    assert a.multiplier == b.multiplier
    np.testing.assert_array_equal(a.expiry, b.expiry)
    for name in ("strike", "bid", "ask", "open_interest", "vendor_iv"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_array_equal(a.is_call, b.is_call)


def test_frame_round_trip_preserves_every_column_including_nan() -> None:
    chain = small_chain()
    nan_row = np.arange(chain.size) % 7 == 0
    chain = ChainSnapshot(
        underlying=chain.underlying,
        asof=chain.asof,
        spot=chain.spot,
        rate=chain.rate,
        dividend_yield=chain.dividend_yield,
        expiry=chain.expiry,
        strike=chain.strike,
        is_call=chain.is_call,
        bid=np.where(nan_row, np.nan, chain.bid),
        ask=chain.ask,
        open_interest=np.where(nan_row, np.nan, chain.open_interest),
        vendor_iv=np.where(nan_row, 0.2, np.nan),
        exercise=ExerciseStyle.EUROPEAN,
        multiplier=10,
    )
    (restored,) = frame_to_snapshots(snapshot_to_frame(chain, 3))
    _assert_same(chain, restored)


def test_directory_recording_round_trip_in_order(tmp_path: Path) -> None:
    chains = [small_chain(tick=i, spot=100.0 + i) for i in range(5)]
    store = RecordingStore(tmp_path / "rec")
    assert store.write_all(chains, segment_size=2) == 5
    assert len(store.files()) == 3
    assert store.count() == 5
    restored = list(store.read())
    assert [c.spot for c in restored] == [100.0, 101.0, 102.0, 103.0, 104.0]
    for original, loaded in zip(chains, restored, strict=True):
        _assert_same(original, loaded)


def test_writer_appends_new_segments_to_an_existing_directory(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path)
    store.write_all([small_chain(tick=0)], segment_size=10)
    store.write_all([small_chain(tick=1)], segment_size=10)
    assert len(store.files()) == 2
    assert store.count() == 2


def test_single_parquet_file_can_be_read(tmp_path: Path) -> None:
    chain = small_chain()
    path = tmp_path / "one.parquet"
    snapshot_to_frame(chain, 0).write_parquet(path)
    (restored,) = RecordingStore(path).read()
    _assert_same(chain, restored)
    with pytest.raises(InvalidInputError):
        RecordingStore(path).writer()


def test_missing_and_empty_recordings_are_reported(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="not found"):
        RecordingStore(tmp_path / "absent").files()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ProviderError, match="empty"):
        list(RecordingStore(empty).read())


def test_missing_columns_are_reported() -> None:
    frame = snapshot_to_frame(small_chain(), 0).drop("strike")
    with pytest.raises(ProviderError, match="missing columns"):
        list(frame_to_snapshots(frame))


def test_writer_counts_and_flushes(tmp_path: Path) -> None:
    writer = RecordingWriter(tmp_path, segment_size=3)
    for i in range(4):
        writer.append(small_chain(tick=i))
    assert writer.snapshots_written == 4
    assert len(list(tmp_path.glob("segment-*.parquet"))) == 1
    writer.close()
    assert len(list(tmp_path.glob("segment-*.parquet"))) == 2
    writer.close()
    with pytest.raises(InvalidInputError):
        RecordingWriter(tmp_path, segment_size=0)


def test_asof_is_stored_as_utc_microseconds(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path)
    store.write_all([small_chain()])
    frame = pl.read_parquet(store.files()[0])
    assert frame.schema["asof"] == pl.Datetime("us", "UTC")
