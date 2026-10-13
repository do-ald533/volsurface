from PySide6 import QtGui

from volsurface.app.warmup import warm_up_numerics
from volsurface.pricing.backends import NumbaCpuBackend
from volsurface.ui import gl_support


def test_warmup_runs_every_compiled_routine() -> None:
    warm_up_numerics(NumbaCpuBackend())


class _Context:
    def __init__(self, ok: bool, version: tuple[int, int]) -> None:
        self._ok = ok
        self._version = version

    def create(self) -> bool:
        return self._ok

    def format(self) -> "_Context":
        return self

    def version(self) -> tuple[int, int]:
        return self._version


def test_opengl_detection_follows_the_platform_and_the_context(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(QtGui.QGuiApplication, "platformName", staticmethod(lambda: "xcb"))
    monkeypatch.setattr(gl_support.QtGui, "QOpenGLContext", lambda: _Context(False, (0, 0)))
    assert not gl_support.opengl_available()
    monkeypatch.setattr(gl_support.QtGui, "QOpenGLContext", lambda: _Context(True, (2, 0)))
    assert not gl_support.opengl_available()
    monkeypatch.setattr(gl_support.QtGui, "QOpenGLContext", lambda: _Context(True, (4, 6)))
    assert gl_support.opengl_available()
