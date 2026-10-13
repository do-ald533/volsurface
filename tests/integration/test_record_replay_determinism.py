"""Event-log property: replaying a recording reproduces the original numbers exactly."""

from pathlib import Path

import numpy as np

from helpers import small_chain
from volsurface.app.config import PipelineSettings
from volsurface.app.pipeline import SurfacePipeline
from volsurface.domain.events import ChainReceived
from volsurface.eventing import AsyncEventBus
from volsurface.eventing.testing import RecordingPublisher
from volsurface.marketdata.recorder import EventRecorder
from volsurface.marketdata.replay import ReplayProvider
from volsurface.pricing.backends import NumbaCpuBackend

SETTINGS = PipelineSettings(tree_steps=60, gex_tree_steps=40)


async def test_replayed_chains_produce_identical_surfaces_and_gex(tmp_path: Path) -> None:
    originals = [small_chain(tick=i, spot=100.0 + 0.4 * i) for i in range(4)]
    async with AsyncEventBus() as bus:
        recorder = EventRecorder(bus, segment_size=3)
        recorder.start(tmp_path / "rec")
        for chain in originals:
            bus.publish(ChainReceived(source="live", snapshot=chain))
        await bus.drain()
        recorder.stop()

    sink = RecordingPublisher()
    provider = ReplayProvider(tmp_path / "rec", speed=0)
    await provider.start(sink)
    await provider.wait_finished()
    replayed = [e.snapshot for e in sink.of_type(ChainReceived)]
    assert len(replayed) == len(originals)

    live_pipeline = SurfacePipeline(NumbaCpuBackend(), SETTINGS)
    replay_pipeline = SurfacePipeline(NumbaCpuBackend(), SETTINGS)
    for original, copy in zip(originals, replayed, strict=True):
        a = live_pipeline.process(original)
        b = replay_pipeline.process(copy)
        np.testing.assert_array_equal(a.implied_vol, b.implied_vol)
        assert a.surface is not None
        assert b.surface is not None
        for left, right in zip(a.surface.slices, b.surface.slices, strict=True):
            assert left.params == right.params
            assert left.fit.rmse_vol == right.fit.rmse_vol
        assert a.gex is not None
        assert b.gex is not None
        np.testing.assert_array_equal(a.gex.gex, b.gex.gex)
        assert [x.message for x in a.alerts] == [x.message for x in b.alerts]
