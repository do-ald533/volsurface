"""Detect whether an OpenGL 3D view can be created in the current Qt platform."""

from PySide6 import QtGui

_UNSUPPORTED_PLATFORMS = frozenset({"offscreen", "minimal", "vnc"})
MINIMUM_VERSION = (2, 1)


def opengl_available() -> bool:
    """True when a desktop OpenGL context of at least version 2.1 can be created.

    The ``offscreen`` and ``minimal`` Qt platforms (used by the test suite and CI) report that
    ``QOpenGLWidget`` is unsupported, so the 3D tab falls back to a message and the heatmap tab
    carries the surface.
    """
    if QtGui.QGuiApplication.platformName() in _UNSUPPORTED_PLATFORMS:
        return False
    context = QtGui.QOpenGLContext()
    if not context.create():
        return False
    return tuple(context.format().version()) >= MINIMUM_VERSION
