from dataclasses import replace

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from pytestqt.qtbot import QtBot

from volsurface.analytics.alerts import Alert, AlertKind
from volsurface.app.diagnostics import DiagnosticsSnapshot
from volsurface.app.lab import HoverInfo
from volsurface.app.results import SessionUpdate
from volsurface.app.session_state import SessionState
from volsurface.app.viewmodels import (
    chain_table,
    gex_view,
    surface_axis_labels,
    surface_mesh,
    term_structure,
)
from volsurface.domain.errors import InvalidInputError
from volsurface.pricing.backends import NumbaCpuBackend
from volsurface.ui import theme
from volsurface.ui.gl_support import opengl_available
from volsurface.ui.widgets.chain import ChainTableWidget, format_cell
from volsurface.ui.widgets.diagnostics import DiagnosticsWidget
from volsurface.ui.widgets.gex import MAX_ALERTS, GexWidget
from volsurface.ui.widgets.hover import HoverProbe, format_hover_html
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


def test_lowest_surface_colour_is_visible_on_the_dark_background() -> None:
    lut = theme.colour_lut(64)
    assert lut[0, :3].max() > 0.35
    assert lut[-1, :3].max() > 0.9


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

    def test_vanilla_chart_follows_the_inputs_and_the_chosen_metric(self, qtbot: QtBot) -> None:
        tab = VanillaTab(NumbaCpuBackend())
        qtbot.addWidget(tab)
        x, value = tab._series[0].getData()
        assert len(x) == len(value) == 121
        assert len(tab._series[1].getData()[0]) == 121
        assert len(tab._series[2].getData()[0]) == 121
        assert "Black-Scholes" in tab._legend.text() and "Intrinsic" in tab._legend.text()
        assert tab._spot_line.value() == 100.0
        tab.spot.setValue(95.0)
        tab.strike.setValue(110.0)
        assert tab._spot_line.value() == 95.0 and tab._strike_line.value() == 110.0
        tab.chart_metric.setCurrentText("Gamma")
        _, gamma = tab._series[0].getData()
        assert np.all(gamma > 0)
        assert tab._series[1].getData()[0] is None or len(tab._series[1].getData()[0]) == 0
        assert "Black-Scholes" not in tab._legend.text() and "Gamma" in tab._legend.text()
        assert tab._plot.getAxis("left").labelText == "gamma"

    def test_vanilla_chart_empties_and_reports_when_pricing_fails(self, qtbot: QtBot) -> None:
        class Broken(NumbaCpuBackend):
            def american_price_gamma(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise InvalidInputError("tree exploded")

        tab = VanillaTab(Broken())
        qtbot.addWidget(tab)
        assert tab.recompute() is None
        assert "tree exploded" in tab._output.text()
        assert tab._curve_data is None
        assert tab._legend.text() == ""
        assert tab._series[0].getData()[0] is None
        assert tab._probe.show_at(100.0, 1.0) is False

    def test_vanilla_chart_survives_extreme_inputs(self, qtbot: QtBot) -> None:
        tab = VanillaTab(NumbaCpuBackend())
        qtbot.addWidget(tab)
        tab.spot.setValue(0.01)
        tab.strike.setValue(0.01)
        tab.days.setValue(0.1)
        tab.vol.setValue(500.0)
        assert tab.recompute() is not None
        assert tab._curve_data is not None
        assert np.isfinite(tab._curve_data.bs_value).all()

    def test_hover_on_each_lab_chart_shows_a_tooltip(self, qtbot: QtBot) -> None:
        lab = PricingLabWidget(NumbaCpuBackend)
        qtbot.addWidget(lab)
        lab.resize(1100, 420)
        lab.show()
        qtbot.waitExposed(lab)

        lab.setCurrentWidget(lab.vanilla)
        assert lab.vanilla._probe.show_at(104.0, 3.0)
        assert "Spot 104.00" in lab.vanilla._probe.text
        assert "American tree" in lab.vanilla._probe.text

        lab.monte_carlo.paths.setValue(50_000)
        assert not lab.monte_carlo._probe.show_at(25_000.0, 9.0)
        lab.monte_carlo.run_blocking()
        assert lab.monte_carlo._probe.show_at(25_000.0, 9.0)
        assert "paths" in lab.monte_carlo._probe.text
        assert "95% CI" in lab.monte_carlo._probe.text
        lab.monte_carlo.run_blocking()
        assert not lab.monte_carlo._probe.visible

        trace = lab.newton._trace
        assert trace is not None
        assert lab.newton._probe.show_at(100 * trace.steps[1].sigma, 3.0)
        assert "Iteration 1" in lab.newton._probe.text
        lab.newton.slider.setValue(0)
        assert not lab.newton._probe.visible
        lab.newton._probe.show_at(100 * trace.steps[1].sigma, 3.0)
        assert "Iteration" not in lab.newton._probe.text

    def test_lab_has_three_tabs(self, qtbot: QtBot) -> None:
        lab = PricingLabWidget(NumbaCpuBackend)
        qtbot.addWidget(lab)
        assert lab.count() == 3
        assert isinstance(lab.widget(1), QtWidgets.QWidget)


class TestHoverProbe:
    @staticmethod
    def _plot(qtbot: QtBot) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        qtbot.addWidget(plot)
        plot.plot([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 4.0, 9.0])
        plot.resize(500, 300)
        plot.show()
        qtbot.waitExposed(plot)
        return plot

    @staticmethod
    def _describe(x: float) -> HoverInfo | None:
        if x < 0.0:
            return None
        snapped = float(np.clip(round(x), 0, 3))
        return HoverInfo(
            snapped,
            f"x = {snapped:g}",
            (("y", f"{snapped**2:g}"), ("<b>", "&")),
            ((snapped, snapped**2),),
        )

    def test_shows_what_describe_reports_and_hides_when_there_is_nothing(
        self, qtbot: QtBot
    ) -> None:
        plot = self._plot(qtbot)
        probe = HoverProbe(plot, self._describe)
        assert not probe.visible
        assert probe.show_at(1.4, 3.0)
        assert probe.visible
        assert "x = 1" in probe.text
        assert probe._line.value() == 1.0
        assert not probe.show_at(-5.0, 3.0)
        assert not probe.visible

    def test_tooltip_text_is_escaped(self) -> None:
        html = format_hover_html(HoverInfo(0.0, "<i>t</i>", (("<b>", "&"),)))
        assert "&lt;i&gt;t&lt;/i&gt;" in html and "&lt;b&gt;" in html and "&amp;" in html

    def test_label_stays_inside_the_plot_whatever_the_cursor_position(self, qtbot: QtBot) -> None:
        plot = self._plot(qtbot)
        probe = HoverProbe(plot, self._describe)
        view_box = plot.getPlotItem().vb
        area = view_box.sceneBoundingRect()
        (x0, x1), (y0, y1) = view_box.viewRange()
        for fx in (0.0, 0.5, 1.0):
            for fy in (0.0, 0.5, 1.0):
                probe.show_at(x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))
                inside = probe._label.sceneBoundingRect()
                assert area.left() - 1 <= inside.left() and inside.right() <= area.right() + 1
                assert area.top() - 1 <= inside.top() and inside.bottom() <= area.bottom() + 1

    def test_follows_real_mouse_movement_and_hides_when_the_mouse_leaves(
        self, qtbot: QtBot
    ) -> None:
        plot = self._plot(qtbot)
        probe = HoverProbe(plot, self._describe)
        view_box = plot.getPlotItem().vb
        view_box.setRange(xRange=(0.0, 3.0), yRange=(0.0, 9.0), padding=0.0)
        target = plot.mapFromScene(view_box.mapViewToScene(QtCore.QPointF(2.0, 4.0)))
        assert plot.viewport().rect().contains(target)
        viewport = plot.viewport()
        move = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(target),
            QtCore.QPointF(viewport.mapToGlobal(target)),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        QtWidgets.QApplication.sendEvent(viewport, move)
        qtbot.waitUntil(lambda: probe.visible, timeout=3000)
        assert "x = 2" in probe.text
        QtCore.QCoreApplication.sendEvent(plot.viewport(), QtCore.QEvent(QtCore.QEvent.Type.Leave))
        assert not probe.visible


class TestClearing:
    def test_every_chart_can_be_emptied_after_showing_data(
        self, qtbot: QtBot, update: SessionUpdate
    ) -> None:
        assert update.surface is not None
        panel, smile, term = SurfacePanel(), SmileWidget(), TermStructureWidget()
        chain, gex = ChainTableWidget(), GexWidget()
        for widget in (panel, smile, term, chain, gex):
            qtbot.addWidget(widget)
        mesh = surface_mesh(update.surface)
        assert mesh is not None
        panel.set_mesh(mesh)
        smile.set_update(update)
        term.set_term_structure(term_structure(update.surface))
        chain.set_table(chain_table(update))
        gex.set_view(gex_view(update))
        assert panel.heatmap.has_data and smile._combo.count() > 0
        assert chain.model.rowCount() > 0 and len(term._line.getData()[0]) > 0

        panel.clear_data()
        for widget in (smile, term, chain, gex):
            widget.clear()
        assert not panel.heatmap.has_data
        assert panel.count() == 2
        assert panel.heatmap._image in panel.heatmap.getPlotItem().items
        assert smile._combo.count() == 0 and smile._status.text() == "no data yet"
        assert smile._curve.getData()[0] is None or len(smile._curve.getData()[0]) == 0
        assert term._line.getData()[0] is None or len(term._line.getData()[0]) == 0
        assert chain.model.rowCount() == 0 and chain._expiry.count() == 1
        assert gex._summary.text() == "no data yet" and not gex._flip.isVisible()

    def test_clearing_twice_and_before_any_data_is_harmless(self, qtbot: QtBot) -> None:
        widgets = (
            SurfacePanel(),
            SmileWidget(),
            TermStructureWidget(),
            ChainTableWidget(),
            GexWidget(),
        )
        for widget in widgets:
            qtbot.addWidget(widget)
            widget.clear()
            widget.clear()

    def test_the_surface_panel_notice(self, qtbot: QtBot) -> None:
        panel = SurfacePanel()
        qtbot.addWidget(panel)
        assert panel.notice == ""
        panel.set_notice("connecting: loading the SPY option chain")
        assert panel.notice == "connecting: loading the SPY option chain"
        panel.set_notice("")
        assert panel.notice == ""

    def test_the_chain_filter_keeps_working_after_a_clear(
        self, qtbot: QtBot, update: SessionUpdate
    ) -> None:
        chain = ChainTableWidget()
        qtbot.addWidget(chain)
        chain.set_table(chain_table(update))
        chain.clear()
        chain.set_table(chain_table(update))
        assert chain.model.rowCount() > 0
        assert chain._expiry.count() > 1
