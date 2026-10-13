"""The surface panel: a 3D OpenGL view and a 2D heatmap of the same implied-volatility grid.

The 3D view is only built when OpenGL is available (see :mod:`volsurface.ui.gl_support`); the
heatmap is plain 2D and always works. Both are fed the same :class:`SurfaceMesh`, and updating
only replaces data, so the user's camera angle survives every refresh.
"""

from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from volsurface.app.viewmodels import AxisLabel, SurfaceMesh, surface_axis_labels, surface_gl_arrays
from volsurface.ui import theme
from volsurface.ui.gl_support import opengl_available


def _seed_data() -> dict[str, np.ndarray]:
    """Arrays for the empty placeholder surface; pyqtgraph needs ndarrays, not lists."""
    axis = np.array([-5.0, 5.0])
    return {"x": axis, "y": axis.copy(), "z": np.zeros((2, 2))}


def _mesh_data(mesh: SurfaceMesh) -> dict[str, np.ndarray]:
    """``GLSurfacePlotItem.setData`` arguments; colours go in as one RGBA row per vertex."""
    arrays = surface_gl_arrays(mesh)
    colours = theme.colour_values(arrays.unit).reshape(-1, 4).astype(np.float32)
    return {"x": arrays.x, "y": arrays.y, "z": arrays.z, "colors": colours}


def _edge_data(extent: float = 10.0, height: float = 4.0) -> dict[str, object]:
    """Three bold box edges (strike, expiry, volatility) that the axis labels sit along."""
    half = extent / 2
    segments = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, -half, 0.0],
            [-half, -half, height],
        ]
    )
    return {"pos": segments, "color": pg.mkColor(theme.MUTED).getRgbF()}


_ALIGNMENT = {
    "left": QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
    "center": QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter,
    "right": QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
}


def style_label(
    item: Any, label: AxisLabel, tick_font: QtGui.QFont, title_font: QtGui.QFont
) -> None:
    """Point a ``GLTextItem`` at ``label``: titles bright and bold, ticks small and muted."""
    item.setData(
        pos=label.pos,
        text=label.text,
        color=theme.TEXT if label.title else theme.MUTED,
        font=title_font if label.title else tick_font,
        alignment=_ALIGNMENT[label.align],
    )


class HeatmapWidget(pg.PlotWidget):  # type: ignore[misc]
    """Implied volatility as colour over log-moneyness (x) and days to expiry (y)."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setLabel("bottom", "log-moneyness  ln(K / F)")
        self.setLabel("left", "days to expiry")
        theme.style_plot(self)
        self._image = pg.ImageItem(axisOrder="row-major")
        self._image.setColorMap(theme.colormap())
        self.addItem(self._image)
        self._bar = pg.ColorBarItem(
            values=(0.0, 1.0), colorMap=theme.colormap(), interactive=False, label="implied vol %"
        )
        self._bar.setImageItem(self._image, insert_in=self.getPlotItem())
        self._has_data = False

    @property
    def has_data(self) -> bool:
        return self._has_data

    def set_mesh(self, mesh: SurfaceMesh) -> None:
        low, high = mesh.vol_range
        if high - low < 1e-9:
            high = low + 1e-6
        self._image.setImage(mesh.iv_pct, autoLevels=False, levels=(low, high))
        self._bar.setLevels((low, high))
        k0, k1 = float(mesh.k[0]), float(mesh.k[-1])
        d0, d1 = float(mesh.days[0]), float(mesh.days[-1])
        self._image.setRect(QtCore.QRectF(k0, d0, k1 - k0, max(d1 - d0, 1e-9)))
        self.setRange(xRange=(k0, k1), yRange=(d0, max(d1, d0 + 1e-6)), padding=0.02)
        self._has_data = True


class Surface3DWidget(QtWidgets.QWidget):
    """OpenGL surface plot, or an explanatory message when OpenGL is not available."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._view: Any | None = None
        self._surface: Any | None = None
        self._labels: list[Any] = []
        if not opengl_available():
            note = QtWidgets.QLabel(
                "3D view unavailable: this Qt platform has no OpenGL support.\n"
                "The heatmap tab shows the same surface."
            )
            note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            note.setObjectName("muted")
            layout.addWidget(note)
            return
        import pyqtgraph.opengl as gl  # noqa: PLC0415

        view = gl.GLViewWidget()
        view.setBackgroundColor(theme.BACKGROUND)
        view.setCameraPosition(distance=24, elevation=26, azimuth=-58)
        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(1, 1)
        view.addItem(grid)
        surface = gl.GLSurfacePlotItem(**_seed_data(), shader="shaded")
        view.addItem(surface)
        view.addItem(gl.GLLinePlotItem(**_edge_data(), mode="lines", width=2.0, antialias=True))
        layout.addWidget(view)
        self._view, self._surface = view, surface
        self._gl = gl
        self._tick_font = QtGui.QFont("Helvetica", 9)
        self._title_font = QtGui.QFont("Helvetica", 11, QtGui.QFont.Weight.Bold)

    @property
    def available(self) -> bool:
        return self._view is not None

    def set_mesh(self, mesh: SurfaceMesh) -> None:
        if self._surface is None:
            return
        self._surface.setData(**_mesh_data(mesh))
        self._draw_labels(surface_axis_labels(mesh))

    def _draw_labels(self, labels: list[AxisLabel]) -> None:
        if self._view is None:
            return
        while len(self._labels) < len(labels):
            item = self._gl.GLTextItem(text="")
            self._view.addItem(item)
            self._labels.append(item)
        for index, item in enumerate(self._labels):
            if index < len(labels):
                style_label(item, labels[index], self._tick_font, self._title_font)
            else:
                item.setData(text="")


class SurfacePanel(QtWidgets.QTabWidget):
    """Tabs: 3D surface, heatmap."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.surface3d = Surface3DWidget()
        self.heatmap = HeatmapWidget()
        self.addTab(self.surface3d, "3D surface")
        self.addTab(self.heatmap, "Heatmap")
        if not self.surface3d.available:
            self.setCurrentWidget(self.heatmap)

    def set_mesh(self, mesh: SurfaceMesh | None) -> None:
        if mesh is None:
            return
        self.surface3d.set_mesh(mesh)
        self.heatmap.set_mesh(mesh)
