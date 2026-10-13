"""Smile and term structure plots."""

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from volsurface.app.results import SessionUpdate
from volsurface.app.viewmodels import SmileView, TermStructure, smile_view
from volsurface.ui import theme

if TYPE_CHECKING:
    from datetime import date


def _plot(x_label: str, y_label: str) -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setLabel("bottom", x_label)
    plot.setLabel("left", y_label)
    theme.style_plot(plot)
    return plot


class SmileWidget(QtWidgets.QWidget):
    """Market implied vols (ours and the broker's) next to the fitted SVI curve for one expiry."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._selected: date | None = None
        self._expiries: list[date] = []
        self._combo = QtWidgets.QComboBox()
        self._status = QtWidgets.QLabel("no data yet")
        self._status.setObjectName("muted")
        self._plot = _plot("log-moneyness  ln(K / F)", "implied vol %")
        self._curve = self._plot.plot(pen=pg.mkPen(theme.ACCENT, width=2), name="SVI fit")
        self._used = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(None), brush=pg.mkBrush(theme.ACCENT_2), name="quotes used"
        )
        self._unused = pg.ScatterPlotItem(
            size=6, pen=pg.mkPen(theme.MUTED), brush=pg.mkBrush(None), name="other quotes"
        )
        self._vendor = pg.ScatterPlotItem(
            size=8, symbol="x", pen=pg.mkPen(theme.MAGENTA, width=1.5), name="broker model"
        )
        for item in (self._unused, self._used, self._vendor):
            self._plot.addItem(item)
        self._plot.addItem(
            pg.InfiniteLine(
                pos=0.0, angle=90, pen=pg.mkPen(theme.MUTED, style=QtCore.Qt.PenStyle.DashLine)
            )
        )
        self._plot.addLegend(offset=(-10, 10))
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Expiry"))
        top.addWidget(self._combo, 1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(top)
        layout.addWidget(self._plot, 1)
        layout.addWidget(self._status)
        self._combo.currentIndexChanged.connect(self._on_choice)
        self._last: SessionUpdate | None = None

    def set_update(self, update: SessionUpdate) -> None:
        self._last = update
        surface = update.surface
        expiries = [s.expiry for s in surface.slices] if surface else []
        if expiries != self._expiries:
            self._expiries = expiries
            self._combo.blockSignals(True)
            self._combo.clear()
            for s in surface.slices if surface else ():
                self._combo.addItem(f"{s.expiry:%Y-%m-%d}  ({s.t * 365:.0f} d)")
            index = expiries.index(self._selected) if self._selected in expiries else 0
            self._combo.setCurrentIndex(index if expiries else -1)
            self._combo.blockSignals(False)
        self._draw(smile_view(update, max(self._combo.currentIndex(), 0)))

    def _on_choice(self, index: int) -> None:
        if 0 <= index < len(self._expiries):
            self._selected = self._expiries[index]
        if self._last is not None:
            self._draw(smile_view(self._last, index))

    def _draw(self, view: SmileView | None) -> None:
        if view is None:
            self._status.setText("no fitted expiry")
            return
        self._curve.setData(view.k_curve, view.iv_curve_pct)
        used = view.used
        self._used.setData(view.k_points[used], view.iv_points_pct[used])
        self._unused.setData(view.k_points[~used], view.iv_points_pct[~used])
        has_vendor = ~np.isnan(view.vendor_iv_pct)
        self._vendor.setData(view.k_points[has_vendor], view.vendor_iv_pct[has_vendor])
        ok = "OK" if view.butterfly_ok else "VIOLATED"
        lee = "OK" if view.lee_ok else "VIOLATED"
        self._status.setText(
            f"{view.n_fit_points} points   RMSE {view.rmse_vol_pct:.2f} vol pts   "
            f"butterfly g_min {view.min_g:.3f} {ok}   Lee wings {lee}"
        )
        self._status.setObjectName("badge_ok" if view.butterfly_ok and view.lee_ok else "badge_bad")
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(560, 330)


class TermStructureWidget(QtWidgets.QWidget):
    """At-the-money implied volatility by expiry."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._plot = _plot("days to expiry", "ATM implied vol %")
        self._line = self._plot.plot(
            pen=pg.mkPen(theme.ACCENT, width=2),
            symbol="o",
            symbolSize=8,
            symbolBrush=theme.ACCENT_2,
            symbolPen=None,
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._plot)

    def set_term_structure(self, structure: TermStructure) -> None:
        self._line.setData(structure.days, structure.atm_vol_pct)

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(560, 330)
