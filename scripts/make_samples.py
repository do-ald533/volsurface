"""Generate the synthetic sample recordings under ``data/samples``.

Both recordings come from the synthetic market (an SSVI surface priced with the American CRR
tree), so they contain no vendor data and can be published.

* ``calm``: a random-walk market, for the plain demo.
* ``flip``: the spot follows a scripted path that crosses the gamma-exposure flip level, so the
  alert rules fire (near-flip, GEX sign change).

Run from the repository root::

    uv run python scripts/make_samples.py
"""

import shutil
from pathlib import Path

import numpy as np

from volsurface.marketdata.recording_store import RecordingStore
from volsurface.marketdata.synthetic import SyntheticChain, SyntheticConfig

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
TICKS = 40


def _walk(config: SyntheticConfig, spots: list[float]) -> list:
    chain = SyntheticChain(config)
    vol_scale = 1.0
    rng = np.random.default_rng(config.seed)
    snapshots = []
    for tick, spot in enumerate(spots):
        asof = config.start + config.sim_step * tick
        vol_scale = float(np.clip(1.0 + 0.9 * (vol_scale - 1.0) + rng.normal(0, 0.008), 0.8, 1.3))
        snapshots.append(chain.snapshot(asof, spot, vol_scale, tick))
    return snapshots


def main() -> None:
    base = SyntheticConfig(seed=21, expiry_days=(7, 14, 30, 60, 90), strike_step=1.0)

    rng = np.random.default_rng(5)
    calm_spots = list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.0006, TICKS))))

    flip_spots = list(
        np.concatenate([np.linspace(97.5, 103.5, TICKS // 2), np.linspace(103.5, 98.5, TICKS // 2)])
    )

    for name, spots in (("calm", calm_spots), ("flip", flip_spots)):
        target = SAMPLES / name
        if target.exists():
            shutil.rmtree(target)
        written = RecordingStore(target).write_all(_walk(base, spots), segment_size=TICKS)
        size = sum(f.stat().st_size for f in target.glob("*.parquet")) / 1024
        print(f"{name}: {written} snapshots, {size:.0f} KiB -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
