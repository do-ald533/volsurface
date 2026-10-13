"""The surface panel: a 3D OpenGL view and a 2D heatmap of the same implied-volatility grid.

The 3D view is only built when OpenGL is available (see :mod:`volsurface.ui.gl_support`); the
heatmap is plain 2D and always works. Both are fed the same :class:`SurfaceMesh`, and updating
only replaces data, so the user's camera angle survives every refresh.
"""

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from volsurface.app.viewmodels import SurfaceMesh, surface_gl_arrays
from volsurface.ui import theme
from volsurface.ui.gl_support import opengl_available


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
        self._view: object | None = None
        self._surface: object | None = None
        self._labels: list[object] = []
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
        surface = gl.GLSurfacePlotItem(
            x=[-5.0, 5.0], y=[-5.0, 5.0], z=[[0.0, 0.0], [0.0, 0.0]], shader="shaded"
        )
        view.addItem(surface)
        for _ in range(6):
            label = gl.GLTextItem(text="", color=QtGui.QColor(theme.MUTED))
            view.addItem(label)
            self._labels.append(label)
        layout.addWidget(view)
        self._view, self._surface = view, surface
        self._gl = gl

    @property
    def available(self) -> bool:
        return self._view is not None

    def set_mesh(self, mesh: SurfaceMesh) -> None:
        if self._surface is None:
            return
        arrays = surface_gl_arrays(mesh)
        colours = theme.colour_values(arrays.unit)
        self._surface.setData(x=arrays.x, y=arrays.y, z=arrays.z, colors=colours)  # type: ignore[attr-defined]
        low, high = mesh.vol_range
        marks = [
            ((-5.6, -5.6, 0.0), f"k {mesh.k[0]:+.2f}"),
            ((5.6, -5.6, 0.0), f"k {mesh.k[-1]:+.2f}"),
            ((-5.8, -5.0, 0.0), f"{mesh.days[0]:.0f} d"),
            ((-5.8, 5.0, 0.0), f"{mesh.days[-1]:.0f} d"),
            ((-6.0, -6.0, 0.0), f"IV {low:.1f}%"),
            ((-6.0, -6.0, 4.2), f"IV {high:.1f}%"),
        ]
        for label, (pos, text) in zip(self._labels, marks, strict=True):
            label.setData(pos=pos, text=text)  # type: ignore[attr-defined]


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
