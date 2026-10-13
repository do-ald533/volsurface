"""Dark theme: Qt palette, stylesheet and plotting defaults, in one place."""

from typing import Final

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

from volsurface.domain.types import FloatArray

BACKGROUND: Final = "#0b0d0f"
PANEL: Final = "#12161b"
PANEL_ALT: Final = "#1a2028"
BORDER: Final = "#2b333d"
TEXT: Final = "#e6edf3"
MUTED: Final = "#8b949e"
ACCENT: Final = "#00d8e6"
ACCENT_2: Final = "#ff9f1a"
POSITIVE: Final = "#3fb950"
NEGATIVE: Final = "#f85149"
WARNING: Final = "#d29922"
MAGENTA: Final = "#d070ff"

COLORMAP: Final = "magma"
COLORMAP_START: Final = 0.3

_STYLESHEET: Final = f"""
QWidget {{ background-color: {PANEL}; color: {TEXT}; font-size: 12px; }}
QMainWindow, QDockWidget {{ background-color: {BACKGROUND}; }}
QDockWidget::title {{
    background: {PANEL_ALT}; padding: 5px 8px; border-bottom: 1px solid {BORDER};
}}
QToolBar {{ background: {PANEL_ALT}; border: none; spacing: 6px; padding: 4px; }}
QToolButton, QPushButton {{
    background: {PANEL_ALT}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 10px;
}}
QToolButton:hover, QPushButton:hover {{ border-color: {ACCENT}; }}
QToolButton:checked, QPushButton:checked {{ background: #123036; border-color: {ACCENT}; }}
QToolButton:disabled, QPushButton:disabled {{ color: {MUTED}; }}
QComboBox, QLineEdit, QSpinBox {{
    background: {BACKGROUND}; border: 1px solid {BORDER}; border-radius: 4px; padding: 3px 6px;
}}
QTabWidget::pane {{ border: 1px solid {BORDER}; }}
QTabBar::tab {{ background: {PANEL_ALT}; padding: 5px 12px; border: 1px solid {BORDER}; }}
QTabBar::tab:selected {{ background: {PANEL}; border-bottom-color: {PANEL}; color: {ACCENT}; }}
QHeaderView::section {{ background: {PANEL_ALT}; border: none; padding: 4px; color: {MUTED}; }}
QTableView {{ gridline-color: {BORDER}; alternate-background-color: {PANEL_ALT}; }}
QStatusBar {{ background: {PANEL_ALT}; color: {MUTED}; }}
QLabel#badge_ok {{ color: {POSITIVE}; }}
QLabel#badge_bad {{ color: {NEGATIVE}; }}
QLabel#muted {{ color: {MUTED}; }}
"""


def apply_theme(app: QtWidgets.QApplication) -> None:
    """Install the dark Fusion style and configure pyqtgraph to match."""
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    roles = {
        QtGui.QPalette.ColorRole.Window: PANEL,
        QtGui.QPalette.ColorRole.WindowText: TEXT,
        QtGui.QPalette.ColorRole.Base: BACKGROUND,
        QtGui.QPalette.ColorRole.AlternateBase: PANEL_ALT,
        QtGui.QPalette.ColorRole.Text: TEXT,
        QtGui.QPalette.ColorRole.Button: PANEL_ALT,
        QtGui.QPalette.ColorRole.ButtonText: TEXT,
        QtGui.QPalette.ColorRole.Highlight: "#164b52",
        QtGui.QPalette.ColorRole.HighlightedText: TEXT,
        QtGui.QPalette.ColorRole.ToolTipBase: PANEL_ALT,
        QtGui.QPalette.ColorRole.ToolTipText: TEXT,
    }
    for role, colour in roles.items():
        palette.setColor(role, QtGui.QColor(colour))
    app.setPalette(palette)
    app.setStyleSheet(_STYLESHEET)
    pg.setConfigOptions(antialias=True, background=BACKGROUND, foreground=TEXT)


def style_plot(plot: pg.PlotWidget) -> None:
    """Plain axis numbers (no SI prefixes such as "x0.001") and a light grid."""
    for name in ("bottom", "left"):
        plot.getAxis(name).enableAutoSIPrefix(False)
    plot.showGrid(x=True, y=True, alpha=0.15)


def colormap() -> pg.ColorMap:
    """The surface colour map without its near-black low end, which vanishes on the dark plot."""
    stops = pg.colormap.get(COLORMAP).getLookupTable(COLORMAP_START, 1.0, 64, alpha=True)
    return pg.ColorMap(np.linspace(0.0, 1.0, 64), stops)


def colour_lut(size: int = 256) -> FloatArray:
    """``(size, 4)`` RGBA lookup table in ``[0, 1]`` for the surface colouring."""
    table = np.asarray(colormap().getLookupTable(0.0, 1.0, size, alpha=True), dtype=np.float64)
    return table / 255.0


def colour_values(unit: FloatArray, size: int = 256) -> FloatArray:
    """RGBA colours (``unit.shape + (4,)``) for values in ``[0, 1]``."""
    lut = colour_lut(size)
    index = np.clip((unit * (size - 1)).astype(np.intp), 0, size - 1)
    return lut[index]
