from dataclasses import replace

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from volsurface.analytics.alerts import Alert, AlertKind
from volsurface.app.diagnostics import DiagnosticsSnapshot
from volsurface.app.results import SessionUpdate
from volsurface.app.session_state import SessionState
from volsurface.app.viewmodels import (
    chain_table,
    gex_view,
    surface_axis_labels,
    surface_mesh,
    term_structure,
)
from volsurface.pricing.backends import NumbaCpuBackend
from volsurface.ui import theme
from volsurface.ui.gl_support import opengl_available
from volsurface.ui.widgets.chain import ChainTableWidget, format_cell
from volsurface.ui.widgets.diagnostics import DiagnosticsWidget
from volsurface.ui.widgets.gex import MAX_ALERTS, GexWidget
from volsurface.ui.widgets.lab import MonteCarloTab, NewtonTab, PricingLabWidget, VanillaTab
from volsurface.ui.widgets.smile import SmileWidget, TermStructureWidget
from volsurface.ui.widgets.surface import (
    HeatmapWidget,
    Surface3DWidget,
    SurfacePanel,
    _edge_data,
    _mesh_data,
    _seed_data,
    style_label,
)

pytestmark = pytest.mark.ui


def test_opengl_is_reported_unavailable_on_the_offscreen_platform(
    qapp: QtWidgets.QApplication,
) -> None:
    assert not opengl_available()


def test_theme_colours() -> None:
    lut = theme.colour_lut(64)
    assert lut.shape == (64, 4)
    assert lut.min() >= 0 and lut.max() <= 1
    values = theme.colour_values(np.array([[0.0, 0.5], [1.0, 0.25]]))
    assert values.shape == (2, 2, 4)
    assert not np.allclose(values[0, 0], values[1, 0])


class TestSurfacePanel:
    def test_heatmap_receives_the_mesh(self, qtbot: QtBot, update: SessionUpdate) -> None:
        assert update.surface is not None
        mesh = surface_mesh(update.surface)
        assert mesh is not None
        heatmap = HeatmapWidget()
        qtbot.addWidget(heatmap)
        assert not heatmap.has_data
        heatmap.set_mesh(mesh)
        assert heatmap.has_data
        image = heatmap._image.image
        assert image.shape == mesh.iv_pct.shape

    def test_flat_mesh_does_not_break_the_colour_scale(
        self, qtbot: QtBot, update: SessionUpdate
    ) -> None:
        assert update.surface is not None
        mesh = surface_mesh(update.surface)
        assert mesh is not None
        heatmap = HeatmapWidget()
        qtbot.addWidget(heatmap)
        heatmap.set_mesh(replace(mesh, iv_pct=np.full_like(mesh.iv_pct, 20.0)))
        assert heatmap.has_data

    def test_3d_view_degrades_gracefully_and_the_panel_shows_the_heatmap(
        self, qtbot: QtBot, update: SessionUpdate
    ) -> None:
        assert update.surface is not None
        panel = SurfacePanel()
        qtbot.addWidget(panel)
        assert not panel.surface3d.available
        assert panel.currentWidget() is panel.heatmap
        panel.set_mesh(surface_mesh(update.surface))
        panel.set_mesh(None)
        assert panel.heatmap.has_data

    def test_3d_widget_without_opengl_ignores_meshes(
        self, qtbot: QtBot, update: SessionUpdate
    ) -> None:
        assert update.surface is not None
        widget = Surface3DWidget()
        qtbot.addWidget(widget)
        mesh = surface_mesh(update.surface)
        assert mesh is not None
        widget.set_mesh(mesh)

    def test_gl_surface_item_accepts_the_seed_and_mesh_data(self, update: SessionUpdate) -> None:
        gl = pytest.importorskip("pyqtgraph.opengl")
        assert update.surface is not None
        mesh = surface_mesh(update.surface)
        assert mesh is not None
        item = gl.GLSurfacePlotItem(**_seed_data(), shader="shaded")
        item.setData(**_mesh_data(mesh))
        n_vertices = mesh.k.size * mesh.days.size
        assert item._meshdata.vertexes().shape == (n_vertices, 3)
        assert item._meshdata.vertexColors().shape == (n_vertices, 4)
        smaller = replace(mesh, k=mesh.k[::2], iv_pct=mesh.iv_pct[:, ::2])
        item.setData(**_mesh_data(smaller))
        assert item._meshdata.vertexes().shape == (smaller.k.size * smaller.days.size, 3)

    def test_gl_text_and_edge_items_accept_the_axis_labels(self, update: SessionUpdate) -> None:
        gl = pytest.importorskip("pyqtgraph.opengl")
        assert update.surface is not None
        mesh = surface_mesh(update.surface)
        assert mesh is not None
        tick_font, title_font = QtGui.QFont("Helvetica", 9), QtGui.QFont("Helvetica", 11)
        labels = surface_axis_labels(mesh)
        for label in labels:
            item = gl.GLTextItem(text="")
            style_label(item, label, tick_font, title_font)
            assert item.text == label.text
            assert tuple(item.pos) == label.pos
            assert item.font is (title_font if label.title else tick_font)
        edges = gl.GLLinePlotItem(**_edge_data(), mode="lines", width=2.0, antialias=True)
        assert edges.pos.shape == (6, 3)


class TestSmile:
    def test_shows_quotes_and_the_fitted_curve(self, qtbot: QtBot, update: SessionUpdate) -> None:
        widget = SmileWidget()
        qtbot.addWidget(widget)
        widget.set_update(update)
        assert widget._combo.count() == 3
        x, y = widget._curve.getData()
        assert len(x) == len(y) == 160
        assert len(widget._used.getData()[0]) > 0
        assert "RMSE" in widget._status.text()
        assert widget._status.objectName() == "badge_ok"

    def test_selection_is_kept_across_updates(self, qtbot: QtBot, update: SessionUpdate) -> None:
        widget = SmileWidget()
        qtbot.addWidget(widget)
        widget.set_update(update)
        widget._combo.setCurrentIndex(2)
        chosen = widget._selected
        widget.set_update(update)
        assert widget._combo.currentIndex() == 2
        assert widget._selected == chosen

    def test_no_surface_clears_the_choices(self, qtbot: QtBot, update: SessionUpdate) -> None:
        widget = SmileWidget()
        qtbot.addWidget(widget)
        widget.set_update(update)
        widget.set_update(replace(update, surface=None))
        assert widget._combo.count() == 0
        assert "no fitted expiry" in widget._status.text()

    def test_term_structure(self, qtbot: QtBot, update: SessionUpdate) -> None:
        assert update.surface is not None
        widget = TermStructureWidget()
        qtbot.addWidget(widget)
        widget.set_term_structure(term_structure(update.surface))
        x, y = widget._line.getData()
        assert len(x) == len(y) == 3


class TestChainTable:
    def test_rows_columns_sorting_and_filtering(self, qtbot: QtBot, update: SessionUpdate) -> None:
        widget = ChainTableWidget()
        qtbot.addWidget(widget)
        table = chain_table(update)
        widget.set_table(table)
        assert widget.model.rowCount() == update.chain.size
        assert widget.model.columnCount() == 11
        assert widget.proxy.rowCount() == int(update.usable.sum())
        widget._used.setChecked(False)
        assert widget.proxy.rowCount() == update.chain.size

        widget.view.sortByColumn(1, QtCore.Qt.SortOrder.AscendingOrder)
        strikes = [
            widget.proxy.index(r, 1).data(QtCore.Qt.ItemDataRole.UserRole)
            for r in range(widget.proxy.rowCount())
        ]
        assert strikes == sorted(strikes)

        widget._expiry.setCurrentIndex(1)
        first_expiry = str(table.expiries()[0])
        shown = {widget.proxy.index(r, 0).data() for r in range(widget.proxy.rowCount())}
        assert shown == {first_expiry}

        widget._expiry.setCurrentIndex(0)
        widget._used.setChecked(True)
        flags = {widget.proxy.index(r, 10).data() for r in range(widget.proxy.rowCount())}
        assert flags == {"yes"}
        assert widget.proxy.rowCount() == int(update.usable.sum())

    def test_same_size_refresh_keeps_the_model_instead_of_resetting(
        self, qtbot: QtBot, update: SessionUpdate
    ) -> None:
        widget = ChainTableWidget()
        qtbot.addWidget(widget)
        widget.set_table(chain_table(update))
        resets: list[int] = []
        changes: list[int] = []
        widget.model.modelReset.connect(lambda: resets.append(1))
        widget.model.dataChanged.connect(lambda *_: changes.append(1))
        widget.set_table(chain_table(update))
        assert resets == []
        assert changes

    def test_cells_are_formatted_and_styled(self, qtbot: QtBot, update: SessionUpdate) -> None:
        widget = ChainTableWidget()
        qtbot.addWidget(widget)
        widget.set_table(chain_table(update))
        unused = int(np.flatnonzero(~update.usable)[0])
        brush = widget.model.index(unused, 3).data(QtCore.Qt.ItemDataRole.ForegroundRole)
        assert brush is not None
        used = int(np.flatnonzero(update.usable)[0])
        assert widget.model.index(used, 3).data(QtCore.Qt.ItemDataRole.ForegroundRole) is None
        header = widget.model.headerData(7, QtCore.Qt.Orientation.Horizontal)
        assert header == "IV %"
        assert widget.model.headerData(0, QtCore.Qt.Orientation.Vertical) is None

    def test_format_cell(self) -> None:
        assert format_cell(1, 101.5) == "101.50"
        assert format_cell(6, 12345.0) == "12,345"
        assert format_cell(7, float("nan")) == "-"
        assert format_cell(9, 0.5) == "+0.50"
        assert format_cell(2, "Call") == "Call"

    def test_empty_model(self, qtbot: QtBot) -> None:
        widget = ChainTableWidget()
        qtbot.addWidget(widget)
        assert widget.model.rowCount() == 0
        assert widget.model.index(0, 0).data() is None


class TestGex:
    def test_profile_flip_and_summary(self, qtbot: QtBot, update: SessionUpdate) -> None:
        widget = GexWidget()
        qtbot.addWidget(widget)
        view = gex_view(update)
        widget.set_view(view)
        x, y = widget._line.getData()
        assert len(x) == len(y) == 81
        assert "GEX at spot" in widget._summary.text()
        widget.set_view(None)
        assert "open interest" in widget._summary.text()

    def test_alert_log_is_bounded_and_newest_first(
        self, qtbot: QtBot, update: SessionUpdate
    ) -> None:
        widget = GexWidget()
        qtbot.addWidget(widget)
        for i in range(MAX_ALERTS + 5):
            widget.add_alert(Alert(AlertKind.IV_JUMP, "SYN", update.chain.asof, f"alert {i}", 1.0))
        assert widget.alert_count() == MAX_ALERTS
        assert widget._alerts.item(0).text().endswith(f"alert {MAX_ALERTS + 4}")


class TestDiagnosticsWidget:
    def test_renders_rows_from_the_source(self, qtbot: QtBot) -> None:
        snapshot = DiagnosticsSnapshot(
            state=SessionState.STREAMING,
            updates=3,
            compute_failures=0,
            handler_failures=0,
            update_rate_hz=2.5,
            options_per_update=600,
            stage_seconds={"surface": 0.02},
            backend_seconds={},
            last_error="<b>x</b>",
        )
        widget = DiagnosticsWidget(lambda: snapshot, interval_ms=10_000)
        qtbot.addWidget(widget)
        text = widget._text.text()
        assert "streaming" in text
        assert "2.5 Hz" in text
        assert "&lt;b&gt;x&lt;/b&gt;" in text


class TestPricingLab:
    def test_vanilla_tab_prices_and_reacts_to_inputs(self, qtbot: QtBot) -> None:
        tab = VanillaTab(NumbaCpuBackend())
        qtbot.addWidget(tab)
        before = tab.recompute()
        assert before is not None
        tab.vol.setValue(40.0)
        after = tab.recompute()
        assert after is not None
        assert after.bs_price > before.bs_price
        assert "Black-Scholes value" in tab._output.text()
        tab.steps.setValue(3)
        assert tab.recompute() is not None

    def test_monte_carlo_tab_runs_and_reports_the_reference(self, qtbot: QtBot) -> None:
        tab = MonteCarloTab()
        qtbot.addWidget(tab)
        tab.paths.setValue(50_000)
        outcome = tab.run_blocking()
        assert outcome.reference is not None
        assert "Black-Scholes" in tab._summary.text()
        assert tab._reference.isVisible() or not tab.isVisible()
        tab.model.setCurrentText("heston")
        tab.run_blocking()
        assert "Heston" in tab._summary.text()

    def test_newton_tab_draws_each_iteration(self, qtbot: QtBot) -> None:
        tab = NewtonTab()
        qtbot.addWidget(tab)
        trace = tab.recompute()
        assert trace.converged
        assert tab.slider.maximum() == len(trace.steps) - 1
        tab.slider.setValue(0)
        assert len(tab._points.getData()[0]) == 1
        tab.slider.setValue(tab.slider.maximum())
        assert len(tab._points.getData()[0]) == len(trace.steps)
        tab.start_vol.setValue(0.5)
        tab.strike.setValue(300.0)
        tab.market_vol.setValue(5.0)
        assert tab._message.text()

    def test_lab_has_three_tabs(self, qtbot: QtBot) -> None:
        lab = PricingLabWidget(NumbaCpuBackend)
        qtbot.addWidget(lab)
        assert lab.count() == 3
        assert isinstance(lab.widget(1), QtWidgets.QWidget)
