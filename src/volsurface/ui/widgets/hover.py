"""Hover tooltips for pyqtgraph charts: a crosshair, marker dots and a small info box.

The widget is deliberately dumb. It asks a ``describe(x)`` callback what is at the cursor's x
position and draws the :class:`~volsurface.app.lab.HoverInfo` it gets back, so what is shown is
computed (and unit-tested) outside Qt.
"""

from collections.abc import Callable
from html import escape

import pyqtgraph as pg
from PySide6 import QtCore, QtGui
from shiboken6 import isValid

from volsurface.app.lab import HoverInfo
from volsurface.ui import theme

type Describe = Callable[[float], HoverInfo | None]

MOUSE_RATE_HZ = 60
CURSOR_GAP = 14.0
EDGE_MARGIN = 4.0


def format_hover_html(info: HoverInfo) -> str:
    """The tooltip body: a bold title over label/value rows."""
    rows = "".join(
        f"<tr><td style='color:{theme.MUTED}'>{escape(label)}&nbsp;&nbsp;</td>"
        f"<td align='right' style='color:{theme.TEXT}'>{escape(value)}</td></tr>"
        for label, value in info.rows
    )
    return (
        f"<div style='font-size:12px'><b style='color:{theme.ACCENT}'>{escape(info.title)}</b>"
        f"<table cellspacing='0' cellpadding='1'>{rows}</table></div>"
    )


class HoverProbe(QtCore.QObject):
    """Follows the mouse over ``plot`` and shows what ``describe`` reports for its x position."""

    def __init__(self, plot: pg.PlotWidget, describe: Describe) -> None:
        super().__init__(plot)
        self._plot = plot
        self._describe = describe
        self._line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(theme.MUTED, width=1, style=QtCore.Qt.PenStyle.DashLine),
        )
        self._dots = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(theme.ACCENT_2), pen=pg.mkPen(theme.BACKGROUND, width=1.5)
        )
        fill = QtGui.QColor(theme.PANEL_ALT)
        fill.setAlpha(235)
        self._label = pg.TextItem(
            anchor=(0.0, 0.0), border=pg.mkPen(theme.BORDER), fill=pg.mkBrush(fill)
        )
        for item, z in ((self._line, 20), (self._dots, 21), (self._label, 22)):
            item.setZValue(z)
            plot.addItem(item, ignoreBounds=True)
        self._proxy = pg.SignalProxy(
            plot.scene().sigMouseMoved, rateLimit=MOUSE_RATE_HZ, slot=self._on_mouse
        )
        plot.viewport().installEventFilter(self)
        self.hide()

    @property
    def alive(self) -> bool:
        """False once Qt has deleted the chart (events can still arrive while a window closes)."""
        return all(isValid(obj) for obj in (self._plot, self._line, self._dots, self._label))

    @property
    def visible(self) -> bool:
        return self.alive and bool(self._label.isVisible())

    @property
    def text(self) -> str:
        return str(self._label.toPlainText())

    def hide(self) -> None:
        if not self.alive:
            return
        for item in (self._line, self._dots, self._label):
            item.setVisible(False)

    def show_at(self, x: float, y: float) -> bool:
        """Show the tooltip for data position ``(x, y)``; returns whether there was anything."""
        if not self.alive:
            return False
        info = self._describe(x)
        if info is None:
            self.hide()
            return False
        self._line.setValue(info.x)
        self._dots.setData([p[0] for p in info.points], [p[1] for p in info.points])
        self._label.setHtml(format_hover_html(info))
        self._place_label(info.x, y)
        for item in (self._line, self._dots, self._label):
            item.setVisible(True)
        return True

    def _place_label(self, x: float, y: float) -> None:
        """Put the box beside the cursor, flipping sides so it always stays inside the plot."""
        view_box = self._plot.getPlotItem().vb
        area = view_box.sceneBoundingRect()
        size = self._label.boundingRect()
        anchor = view_box.mapViewToScene(QtCore.QPointF(x, y))
        left = anchor.x() + CURSOR_GAP
        if left + size.width() > area.right() - EDGE_MARGIN:
            left = anchor.x() - CURSOR_GAP - size.width()
        top = anchor.y() - CURSOR_GAP - size.height()
        if top < area.top() + EDGE_MARGIN:
            top = anchor.y() + CURSOR_GAP
        left = max(area.left() + EDGE_MARGIN, min(left, area.right() - size.width() - EDGE_MARGIN))
        top = max(area.top() + EDGE_MARGIN, min(top, area.bottom() - size.height() - EDGE_MARGIN))
        self._label.setAnchor((0.0, 0.0))
        self._label.setPos(view_box.mapSceneToView(QtCore.QPointF(left, top)))

    def _on_mouse(self, args: tuple[QtCore.QPointF]) -> None:
        if not self.alive:
            return
        scene_pos = args[0]
        view_box = self._plot.getPlotItem().vb
        if not view_box.sceneBoundingRect().contains(scene_pos):
            self.hide()
            return
        point = view_box.mapSceneToView(scene_pos)
        self.show_at(float(point.x()), float(point.y()))

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Leave:
            self.hide()
        return False
