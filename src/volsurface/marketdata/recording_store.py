"""Parquet storage for recorded option chains (Repository pattern).

A recording is an ordered sequence of :class:`~volsurface.domain.types.ChainSnapshot` objects. On
disk it is either one ``.parquet`` file or a directory of numbered segment files written by
:class:`RecordingWriter`. Rows are long-format (one row per option per snapshot) with a
``snapshot_id`` column, so recordings are easy to inspect with any Parquet tool.

Real broker data must not be committed to a public repository (vendor terms generally forbid
redistribution); ``data/recordings/`` is git-ignored for that reason and only synthetic samples
live under ``data/samples/``.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl

from volsurface.domain.errors import InvalidInputError, ProviderError
from volsurface.domain.types import ChainSnapshot, ExerciseStyle

SEGMENT_PATTERN = "segment-{index:05d}.parquet"
_META_COLUMNS = (
    "snapshot_id",
    "asof",
    "underlying",
    "spot",
    "rate",
    "dividend_yield",
    "exercise",
    "multiplier",
)


def snapshot_to_frame(snapshot: ChainSnapshot, snapshot_id: int) -> pl.DataFrame:
    """One long-format frame for a snapshot."""
    n = snapshot.size
    return pl.DataFrame(
        {
            "snapshot_id": np.full(n, snapshot_id, dtype=np.int64),
            "asof": [snapshot.asof] * n,
            "underlying": [snapshot.underlying] * n,
            "spot": np.full(n, snapshot.spot),
            "rate": np.full(n, snapshot.rate),
            "dividend_yield": np.full(n, snapshot.dividend_yield),
            "exercise": [snapshot.exercise.value] * n,
            "multiplier": np.full(n, snapshot.multiplier, dtype=np.int64),
            "expiry": snapshot.expiry,
            "strike": snapshot.strike,
            "is_call": snapshot.is_call,
            "bid": snapshot.bid,
            "ask": snapshot.ask,
            "open_interest": snapshot.open_interest,
            "vendor_iv": snapshot.vendor_iv,
        },
        schema_overrides={"asof": pl.Datetime("us", "UTC")},
    )


def frame_to_snapshots(frame: pl.DataFrame) -> Iterator[ChainSnapshot]:
    """Rebuild snapshots from a long-format frame, in ``snapshot_id`` order."""
    missing = {*_META_COLUMNS, "expiry", "strike", "is_call", "bid", "ask"} - set(frame.columns)
    if missing:
        raise ProviderError(f"recording is missing columns: {sorted(missing)}")
    for _, part in frame.sort("snapshot_id", maintain_order=True).group_by(
        "snapshot_id", maintain_order=True
    ):
        head = part.row(0, named=True)
        yield ChainSnapshot(
            underlying=head["underlying"],
            asof=head["asof"],
            spot=head["spot"],
            rate=head["rate"],
            dividend_yield=head["dividend_yield"],
            expiry=part["expiry"].to_numpy().astype("datetime64[D]"),
            strike=part["strike"].to_numpy(),
            is_call=part["is_call"].to_numpy(),
            bid=part["bid"].to_numpy(),
            ask=part["ask"].to_numpy(),
            open_interest=part["open_interest"].to_numpy(),
            vendor_iv=part["vendor_iv"].to_numpy(),
            exercise=ExerciseStyle(head["exercise"]),
            multiplier=head["multiplier"],
        )


class RecordingWriter:
    """Buffers snapshots and writes a new segment file every ``segment_size`` of them."""

    def __init__(self, directory: Path, segment_size: int = 50) -> None:
        if segment_size < 1:
            raise InvalidInputError("segment_size must be at least 1")
        directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(directory.glob("segment-*.parquet"))
        self._directory = directory
        self._segment_size = segment_size
        self._next_segment = len(existing)
        self._next_snapshot = 0
        self._buffer: list[pl.DataFrame] = []
        self._buffered_snapshots = 0
        self._written = 0

    @property
    def snapshots_written(self) -> int:
        return self._written + self._buffered_snapshots

    def append(self, snapshot: ChainSnapshot) -> None:
        self._buffer.append(snapshot_to_frame(snapshot, self._next_snapshot))
        self._next_snapshot += 1
        self._buffered_snapshots += 1
        if self._buffered_snapshots >= self._segment_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        path = self._directory / SEGMENT_PATTERN.format(index=self._next_segment)
        pl.concat(self._buffer).write_parquet(path)
        self._next_segment += 1
        self._written += self._buffered_snapshots
        self._buffer.clear()
        self._buffered_snapshots = 0

    def close(self) -> None:
        self.flush()


class RecordingStore:
    """Read and write access to a recording at ``path`` (a directory or one parquet file)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def writer(self, segment_size: int = 50) -> RecordingWriter:
        if self._path.suffix == ".parquet":
            raise InvalidInputError(
                "write recordings to a directory, not to a single .parquet file"
            )
        return RecordingWriter(self._path, segment_size)

    def files(self) -> list[Path]:
        if self._path.is_file():
            return [self._path]
        if self._path.is_dir():
            return sorted(self._path.glob("segment-*.parquet"))
        raise ProviderError(f"recording not found: {self._path}")

    def read(self) -> Iterator[ChainSnapshot]:
        """Snapshots in recorded order, loading one segment at a time."""
        files = self.files()
        if not files:
            raise ProviderError(f"recording is empty: {self._path}")
        for file in files:
            yield from frame_to_snapshots(pl.read_parquet(file))

    def count(self) -> int:
        return sum(
            pl.scan_parquet(f).select(pl.col("snapshot_id").n_unique()).collect().item()
            for f in self.files()
        )

    def write_all(self, snapshots: list[ChainSnapshot], segment_size: int = 50) -> int:
        """Convenience for tests and sample generation: write ``snapshots`` and close."""
        writer = self.writer(segment_size)
        for snapshot in snapshots:
            writer.append(snapshot)
        writer.close()
        return writer.snapshots_written
