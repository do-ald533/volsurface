"""Gamma-exposure panel: profile against spot, flip level and the alert log."""

import math

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from volsurface.analytics.alerts import Alert
from volsurface.app.viewmodels import GexView
from volsurface.ui import theme

MAX_ALERTS = 200


class GexWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "underlying price")
        self._plot.setLabel("left", "GEX  bn$ per 1% move")
        theme.style_plot(self._plot)
        self._line = self._plot.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self._zero = pg.InfiniteLine(pos=0.0, angle=0, pen=pg.mkPen(theme.MUTED))
        self._spot = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(theme.ACCENT_2, width=2),
            label="spot",
            labelOpts={"position": 0.92},
        )
        self._flip = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(theme.NEGATIVE, width=2, style=QtCore.Qt.PenStyle.DashLine),
            label="flip",
            labelOpts={"position": 0.8},
        )
        for item in (self._zero, self._spot, self._flip):
            self._plot.addItem(item)
        self._flip.setVisible(False)
        self._summary = QtWidgets.QLabel("GEX needs a source that provides open interest")
        self._summary.setObjectName("muted")
        self._summary.setWordWrap(True)
        self._alerts = QtWidgets.QListWidget()
        self._alerts.setMaximumHeight(72)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._plot, 3)
        layout.addWidget(self._summary)
        layout.addWidget(self._alerts)

    def clear(self) -> None:
        self._line.setData([], [])
        self._flip.setVisible(False)
        self._summary.setText("no data yet")

    def set_view(self, view: GexView | None) -> None:
        if view is None:
            self._line.setData([], [])
            self._flip.setVisible(False)
            self._summary.setText("GEX needs a source that provides open interest")
            return
        self._line.setData(view.levels, view.gex)
        self._spot.setValue(view.spot)
        has_flip = math.isfinite(view.flip)
        self._flip.setVisible(has_flip)
        if has_flip:
            self._flip.setValue(view.flip)
        sign = (
            "positive: moves tend to be damped"
            if view.gex_at_spot > 0
            else ("negative: moves tend to be amplified")
        )
        flip_text = (
            f"flip {view.flip:.2f} ({view.distance_to_flip_pct:+.2f}% from spot)"
            if has_flip
            else "no flip within the grid"
        )
        self._summary.setText(
            f"GEX at spot {view.gex_at_spot:+.3f} bn$/1% ({sign})   {flip_text}   "
            f"{view.n_options} options"
        )

    def add_alert(self, alert: Alert) -> None:
        self._alerts.insertItem(0, f"{alert.asof:%H:%M:%S}  {alert.message}")
        while self._alerts.count() > MAX_ALERTS:
            self._alerts.takeItem(self._alerts.count() - 1)

    def alert_count(self) -> int:
        return self._alerts.count()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(560, 260)
