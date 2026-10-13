"""The committed synthetic samples behave as documented (they back the README demo)."""

from collections import Counter
from pathlib import Path

import pytest

from volsurface.analytics.alerts import AlertKind
from volsurface.app.config import PipelineSettings
from volsurface.app.pipeline import SurfacePipeline
from volsurface.marketdata.recording_store import RecordingStore
from volsurface.pricing.backends import NumbaCpuBackend

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"
SETTINGS = PipelineSettings(tree_steps=80, gex_tree_steps=50)

pytestmark = pytest.mark.slow


def _alerts(name: str) -> Counter[AlertKind]:
    pipeline = SurfacePipeline(NumbaCpuBackend(), SETTINGS)
    counts: Counter[AlertKind] = Counter()
    for chain in RecordingStore(SAMPLES / name).read():
        update = pipeline.process(chain)
        assert update.surface is not None
        assert update.surface.arbitrage_report().ok
        counts.update(a.kind for a in update.alerts)
    return counts


def test_the_flip_sample_crosses_the_gex_flip_and_raises_alerts() -> None:
    counts = _alerts("flip")
    assert counts[AlertKind.GEX_SIGN_CHANGE] >= 1
    assert counts[AlertKind.NEAR_FLIP] >= 1


def test_the_calm_sample_gives_arbitrage_free_surfaces_throughout() -> None:
    _alerts("calm")


def test_samples_are_small_enough_to_commit() -> None:
    for name in ("calm", "flip"):
        size = sum(f.stat().st_size for f in (SAMPLES / name).glob("*.parquet"))
        assert 10_000 < size < 1_000_000
        assert RecordingStore(SAMPLES / name).count() == 40
