from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from helpers import small_chain
from volsurface.app.config import PipelineSettings
from volsurface.app.diagnostics import DiagnosticsSnapshot
from volsurface.app.pipeline import SurfacePipeline
from volsurface.app.results import SessionUpdate
from volsurface.app.session_state import SessionState
from volsurface.app.viewmodels import (
    CHAIN_COLUMNS,
    chain_table,
    format_diagnostics,
    gex_view,
    iv_comparison,
    nice_ticks,
    smile_view,
    surface_axis_labels,
    surface_gl_arrays,
    surface_mesh,
    term_structure,
)
from volsurface.domain.ports import DeliveryPolicy
from volsurface.eventing.bus import SubscriptionStats
from volsurface.pricing.backends import NumbaCpuBackend


@pytest.fixture(scope="module")
def update() -> SessionUpdate:
    pipeline = SurfacePipeline(
        NumbaCpuBackend(), PipelineSettings(tree_steps=60, gex_tree_steps=40)
    )
    return pipeline.process(small_chain())


def test_surface_mesh_covers_the_fitted_range(update: SessionUpdate) -> None:
    assert update.surface is not None
    mesh = surface_mesh(update.surface, n_k=21, n_t=12)
    assert mesh is not None
    assert mesh.iv_pct.shape == (12, 21)
    assert mesh.k.size == 21 and mesh.days.size == 12
    assert np.all(np.diff(mesh.k) > 0) and np.all(np.diff(mesh.days) > 0)
    assert mesh.days[0] == pytest.approx(update.surface.slices[0].t * 365)
    assert mesh.days[-1] == pytest.approx(update.surface.slices[-1].t * 365)
    lo, hi = mesh.vol_range
    assert 5.0 < lo < hi < 80.0


def test_surface_mesh_with_one_slice_still_spans_a_time_range(update: SessionUpdate) -> None:
    assert update.surface is not None
    single = replace(update.surface, slices=update.surface.slices[:1])
    mesh = surface_mesh(single)
    assert mesh is not None
    assert mesh.days[0] < mesh.days[-1]


def test_empty_surface_has_no_mesh(update: SessionUpdate) -> None:
    assert update.surface is not None
    assert surface_mesh(replace(update.surface, slices=())) is None


def test_gl_arrays_are_normalised_and_shaped_for_the_3d_item(update: SessionUpdate) -> None:
    assert update.surface is not None
    mesh = surface_mesh(update.surface, n_k=15, n_t=9)
    assert mesh is not None
    arrays = surface_gl_arrays(mesh, extent=10.0, height=4.0)
    assert arrays.x.shape == (15,) and arrays.y.shape == (9,)
    assert arrays.z.shape == (15, 9) == arrays.unit.shape
    assert arrays.z.min() == pytest.approx(0.0)
    assert arrays.z.max() == pytest.approx(4.0)
    assert arrays.x[0] == -5.0 and arrays.x[-1] == 5.0
    assert arrays.unit.min() == pytest.approx(0.0) and arrays.unit.max() == pytest.approx(1.0)


def test_gl_arrays_for_a_flat_surface(update: SessionUpdate) -> None:
    assert update.surface is not None
    mesh = surface_mesh(update.surface)
    assert mesh is not None
    flat = replace(mesh, iv_pct=np.full_like(mesh.iv_pct, 20.0))
    arrays = surface_gl_arrays(flat)
    assert np.all(arrays.unit == 0.5)
    assert np.isfinite(arrays.z).all()


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (0.0, 1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        (-0.25, 0.25, [-0.2, -0.1, 0.0, 0.1, 0.2]),
        (7.0, 91.0, [20.0, 40.0, 60.0, 80.0]),
        (14.3, 15.9, [14.5, 15.0, 15.5]),
        (3.0, 3.0, [3.0]),
        (5.0, 1.0, []),
        (0.0, float("nan"), []),
    ],
)
def test_nice_ticks_are_round_and_inside_the_range(
    lo: float, hi: float, expected: list[float]
) -> None:
    assert nice_ticks(lo, hi) == pytest.approx(expected)


@given(
    lo=st.floats(-1e3, 1e3),
    span=st.floats(1e-3, 1e3),
    target=st.integers(2, 9),
)
def test_nice_ticks_stay_inside_and_are_evenly_spaced(lo: float, span: float, target: int) -> None:
    hi = lo + span
    ticks = nice_ticks(lo, hi, target)
    tolerance = 1e-8 * span
    assert all(lo - tolerance <= t <= hi + tolerance for t in ticks)
    assert len(ticks) <= 2 * target + 2
    if len(ticks) > 2:
        gaps = np.diff(ticks)
        assert np.allclose(gaps, gaps[0], rtol=1e-6, atol=1e-9)


def test_axis_labels_name_all_three_axes_and_sit_inside_the_box(update: SessionUpdate) -> None:
    assert update.surface is not None
    mesh = surface_mesh(update.surface)
    assert mesh is not None
    labels = surface_axis_labels(mesh, extent=10.0, height=4.0)
    titles = [label.text for label in labels if label.title]
    assert len(titles) == 3
    assert any("ln(K/F)" in t for t in titles)
    assert any("Days to expiry" in t for t in titles)
    assert any("Implied vol" in t for t in titles)
    ticks = [label for label in labels if not label.title]
    assert any(label.text == "ATM" for label in ticks)
    assert sum(label.text.endswith(" d") for label in ticks) >= 2
    assert sum(label.text.endswith("%") for label in ticks) >= 2
    for label in ticks:
        x, y, z = label.pos
        assert -7.0 <= x <= 7.0 and -7.0 <= y <= 7.0 and 0.0 <= z <= 4.0


def test_axis_tick_positions_follow_the_data(update: SessionUpdate) -> None:
    assert update.surface is not None
    mesh = surface_mesh(update.surface)
    assert mesh is not None
    labels = surface_axis_labels(mesh, extent=10.0, height=4.0)
    k_lo, k_hi = float(mesh.k[0]), float(mesh.k[-1])
    atm = next(label for label in labels if label.text == "ATM")
    assert atm.pos[0] == pytest.approx(-5.0 + 10.0 * (0.0 - k_lo) / (k_hi - k_lo))
    low, high = mesh.vol_range
    for label in labels:
        if not label.title and label.text.endswith("%"):
            vol = float(label.text.removesuffix("%"))
            assert label.pos[2] == pytest.approx(4.0 * (vol - low) / (high - low), abs=0.02)


def test_axis_labels_for_a_flat_surface_do_not_divide_by_zero(update: SessionUpdate) -> None:
    assert update.surface is not None
    mesh = surface_mesh(update.surface)
    assert mesh is not None
    flat = replace(mesh, iv_pct=np.full_like(mesh.iv_pct, 20.0))
    labels = surface_axis_labels(flat)
    vol_ticks = [label for label in labels if not label.title and label.text.endswith("%")]
    assert [label.text for label in vol_ticks] == ["20%"]
    assert vol_ticks[0].pos[2] == pytest.approx(2.0)


def test_smile_view_pairs_quotes_with_the_fitted_curve(update: SessionUpdate) -> None:
    view = smile_view(update, 1)
    assert view is not None
    assert isinstance(view.expiry, date)
    assert view.k_points.shape == view.iv_points_pct.shape == view.used.shape
    assert view.k_points.size >= 6
    assert view.k_curve.shape == view.iv_curve_pct.shape == (160,)
    assert view.butterfly_ok
    assert view.used.any()
    assert view.n_fit_points >= 6
    near = np.argmin(np.abs(view.k_curve))
    atm_points = view.iv_points_pct[np.argmin(np.abs(view.k_points))]
    assert view.iv_curve_pct[near] == pytest.approx(atm_points, abs=2.0)


def test_smile_view_index_out_of_range_or_no_surface(update: SessionUpdate) -> None:
    assert smile_view(update, 99) is None
    assert smile_view(update, -1) is None
    assert smile_view(replace(update, surface=None), 0) is None


def test_term_structure_has_one_point_per_slice(update: SessionUpdate) -> None:
    assert update.surface is not None
    ts = term_structure(update.surface)
    assert ts.days.shape == ts.atm_vol_pct.shape == ts.forward.shape == (3,)
    assert np.all(np.diff(ts.days) > 0)
    assert np.all((ts.atm_vol_pct > 5) & (ts.atm_vol_pct < 60))


def test_chain_table_cells_and_columns(update: SessionUpdate) -> None:
    table = chain_table(update)
    assert table.rows == update.chain.size
    assert len(CHAIN_COLUMNS) == 11
    row = int(np.flatnonzero(update.usable)[0])
    values = [table.value(row, c) for c in range(len(CHAIN_COLUMNS))]
    assert values[2] in {"Call", "Put"}
    assert values[10] == "yes"
    assert values[7] == pytest.approx(100.0 * update.implied_vol[row])
    assert len(table.expiries()) == 3
    with pytest.raises(IndexError):
        table.value(0, 99)


def test_iv_comparison_needs_vendor_values(update: SessionUpdate) -> None:
    assert iv_comparison(update) is None
    vendor = np.where(update.usable, update.implied_vol + 0.01, np.nan)
    chain = replace(update.chain, vendor_iv=vendor)
    comparison = iv_comparison(replace(update, chain=chain))
    assert comparison is not None
    assert comparison.n == int(np.isfinite(vendor).sum())
    assert comparison.mean_abs_diff == pytest.approx(1.0)
    assert comparison.rmse == pytest.approx(1.0)
    assert comparison.correlation > 0.99


def test_gex_view_mirrors_the_profile(update: SessionUpdate) -> None:
    view = gex_view(update)
    assert view is not None
    assert update.gex is not None
    assert view.levels.shape == view.gex.shape == (81,)
    assert view.spot == update.chain.spot
    assert view.n_options > 0
    assert gex_view(replace(update, gex=None)) is None


def test_diagnostics_rows() -> None:
    snapshot = DiagnosticsSnapshot(
        state=SessionState.STREAMING,
        updates=12,
        compute_failures=1,
        handler_failures=0,
        update_rate_hz=4.5,
        options_per_update=1234,
        stage_seconds={"filter": 0.001, "surface": 0.05, "implied_vol": 0.02, "gex": 0.3},
        backend_seconds={"american_implied_vol": 0.018},
        last_error="boom",
        subscriptions=(
            SubscriptionStats("a", "E", DeliveryPolicy.LATEST, 10, 3, 0, 0),
            SubscriptionStats("b", "E", DeliveryPolicy.QUEUED, 10, 2, 0, 0),
        ),
    )
    rows = dict(format_diagnostics(snapshot))
    assert rows["State"] == "streaming"
    assert rows["Updates"] == "12 at 4.5 Hz"
    assert rows["Options per update"] == "1,234"
    assert rows["Pipeline total"] == "371.0 ms"
    assert rows["Events dropped (conflated)"] == "5"
    assert rows["Last error"] == "boom"
    assert rows["  backend american_implied_vol"] == "18.0 ms"
