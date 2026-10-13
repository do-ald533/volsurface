"""Runtime diagnostics panel, refreshed on a timer from the diagnostics collector."""

from collections.abc import Callable
from html import escape

from PySide6 import QtCore, QtWidgets

from volsurface.app.diagnostics import DiagnosticsSnapshot
from volsurface.app.viewmodels import format_diagnostics
from volsurface.ui import theme


class DiagnosticsWidget(QtWidgets.QWidget):
    def __init__(
        self,
        source: Callable[[], DiagnosticsSnapshot],
        *,
        interval_ms: int = 500,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._text = QtWidgets.QLabel()
        self._text.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._text.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft
        )
        self._text.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._text, 1)
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(interval_ms)
        self.refresh()

    def refresh(self) -> None:
        rows = format_diagnostics(self._source())
        body = "".join(
            f"<tr><td style='color:{theme.MUTED}'>{escape(k)}</td>"
            f"<td align='right'>&nbsp;&nbsp;{escape(v)}</td></tr>"
            for k, v in rows
        )
        self._text.setText(f"<table>{body}</table>")

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(270, 400)
