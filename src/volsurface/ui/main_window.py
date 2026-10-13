"""The main window: toolbar, docked panels and the wiring between them.

The window issues *commands* through the dispatcher and listens to the *bridge's signals*; it never
touches the session, the bus or a provider directly.
"""

import asyncio
from collections.abc import Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from volsurface import __version__
from volsurface.analytics.alerts import Alert
from volsurface.app.commands import Command, Freeze, Resume, SetRecording, StartSession, StopSession
from volsurface.app.container import AppContainer
from volsurface.app.events import SessionStateChanged
from volsurface.app.results import SessionUpdate
from volsurface.app.session_state import SessionState
from volsurface.app.sources import SourceSpec
from volsurface.app.viewmodels import chain_table, gex_view, surface_mesh, term_structure
from volsurface.pricing.backends import default_registry
from volsurface.ui.qt_bridge import EventBridge
from volsurface.ui.widgets.chain import ChainTableWidget
from volsurface.ui.widgets.diagnostics import DiagnosticsWidget
from volsurface.ui.widgets.gex import GexWidget
from volsurface.ui.widgets.lab import PricingLabWidget
from volsurface.ui.widgets.smile import SmileWidget, TermStructureWidget
from volsurface.ui.widgets.surface import SurfacePanel

_IDLE_STATES = frozenset({SessionState.IDLE, SessionState.STOPPED, SessionState.FAILED})
_LIVE_STATES = frozenset({SessionState.STREAMING, SessionState.FROZEN})
SETTINGS_ORG = "volsurface"
SETTINGS_APP = "desk"


class MainWindow(QtWidgets.QMainWindow):
    closed = QtCore.Signal()

    def __init__(
        self,
        container: AppContainer,
        bridge: EventBridge,
        *,
        settings: QtCore.QSettings | None = None,
        initial_source: str = "synthetic",
    ) -> None:
        super().__init__()
        self._container = container
        self._bridge = bridge
        self._settings = settings or QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._pending: SessionUpdate | None = None
        self._state = SessionState.IDLE
        self._replay_path: Path | None = None
        self.last_error = ""
        self.frames_rendered = 0
        self.setWindowTitle(f"Volsurface Desk {__version__}")
        self.resize(1500, 920)

        self.surface_panel = SurfacePanel()
        self.setCentralWidget(self.surface_panel)
        self.smile = SmileWidget()
        self.term = TermStructureWidget()
        self.gex = GexWidget()
        self.chain = ChainTableWidget()
        self.diagnostics = DiagnosticsWidget(
            lambda: container.diagnostics.snapshot(container.bus.stats())
        )
        self.lab = PricingLabWidget(lambda: default_registry().create(container.settings.backend))

        self._docks: dict[str, QtWidgets.QDockWidget] = {}
        self._build_docks()
        self._build_toolbar(initial_source)
        self._build_status_bar()
        self._connect_signals()
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.timeout.connect(self.flush_pending)
        self._render_timer.start(int(1000 / container.settings.ui_hz))
        self._needs_default_layout = not self._restore_layout()
        self._apply_state(SessionState.IDLE)

    def _build_docks(self) -> None:
        areas = QtCore.Qt.DockWidgetArea
        placements = (
            ("Smile and SVI fit", self.smile, areas.RightDockWidgetArea),
            ("Term structure", self.term, areas.RightDockWidgetArea),
            ("Gamma exposure", self.gex, areas.RightDockWidgetArea),
            ("Option chain", self.chain, areas.BottomDockWidgetArea),
            ("Pricing lab", self.lab, areas.BottomDockWidgetArea),
            ("Diagnostics", self.diagnostics, areas.LeftDockWidgetArea),
        )
        for title, widget, area in placements:
            dock = QtWidgets.QDockWidget(title, self)
            dock.setObjectName(title)
            dock.setWidget(widget)
            self.addDockWidget(area, dock)
            self._docks[title] = dock
        self.tabifyDockWidget(self._docks["Smile and SVI fit"], self._docks["Term structure"])
        self._docks["Smile and SVI fit"].raise_()
        self.tabifyDockWidget(self._docks["Option chain"], self._docks["Pricing lab"])
        self._docks["Option chain"].raise_()
        self._docks["Diagnostics"].setMaximumWidth(330)
        view_menu = self.menuBar().addMenu("&View")
        for dock in self._docks.values():
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        reset = view_menu.addAction("Reset layout")
        reset.triggered.connect(self._reset_layout)

    def _build_toolbar(self, initial_source: str) -> None:
        bar = self.addToolBar("Controls")
        bar.setObjectName("Controls")
        bar.setMovable(False)
        self.source = QtWidgets.QComboBox()
        self.source.addItems(["synthetic", "replay", "ibkr"])
        self.source.setCurrentText(SourceSpec.parse(initial_source).kind)
        self.symbol = QtWidgets.QLineEdit(SourceSpec.parse(initial_source).symbol)
        self.symbol.setMaximumWidth(90)
        self.browse = QtWidgets.QPushButton("Recording ...")
        self.start_action = QtGui.QAction("Start", self)
        self.stop_action = QtGui.QAction("Stop", self)
        self.freeze_action = QtGui.QAction("Freeze", self, checkable=True)
        self.record_action = QtGui.QAction("Record", self, checkable=True)
        bar.addWidget(QtWidgets.QLabel(" Source "))
        bar.addWidget(self.source)
        bar.addWidget(self.symbol)
        bar.addWidget(self.browse)
        bar.addSeparator()
        for action in (self.start_action, self.stop_action, self.freeze_action, self.record_action):
            bar.addAction(action)
        self._on_source_changed(self.source.currentText())

    def _build_status_bar(self) -> None:
        self._state_label = QtWidgets.QLabel("idle")
        self._rate_label = QtWidgets.QLabel("")
        self._source_label = QtWidgets.QLabel("")
        status = self.statusBar()
        status.addPermanentWidget(self._source_label)
        status.addPermanentWidget(self._rate_label)
        status.addPermanentWidget(self._state_label)

    def _connect_signals(self) -> None:
        self._bridge.update_ready.connect(self.on_update)
        self._bridge.state_changed.connect(self.on_state)
        self._bridge.alert_raised.connect(self.on_alert)
        self._bridge.failure.connect(self._show_error)
        self.source.currentTextChanged.connect(self._on_source_changed)
        self.browse.clicked.connect(self._choose_recording)
        self.start_action.triggered.connect(self._start)
        self.stop_action.triggered.connect(lambda: self.issue(StopSession()))
        self.freeze_action.toggled.connect(self._on_freeze_toggled)
        self.record_action.toggled.connect(self._on_record_toggled)

    def _on_source_changed(self, kind: str) -> None:
        self.symbol.setEnabled(kind != "replay")
        self.browse.setVisible(kind == "replay")

    def _choose_recording(self) -> None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(self, "Recording directory")
        if chosen:
            self._replay_path = Path(chosen)
            self.browse.setText(self._replay_path.name)

    def current_spec(self) -> SourceSpec:
        kind = self.source.currentText()
        if kind == "replay":
            if self._replay_path is None:
                raise ValueError("choose a recording directory first")
            return SourceSpec("replay", path=self._replay_path)
        return SourceSpec.parse(f"{kind}:{self.symbol.text().strip()}")

    def _start(self) -> None:
        try:
            spec = self.current_spec()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self.issue(StartSession(spec))

    def _on_freeze_toggled(self, frozen: bool) -> None:
        if frozen and self._state is SessionState.STREAMING:
            self.issue(Freeze())
        elif not frozen and self._state is SessionState.FROZEN:
            self.issue(Resume())

    def _on_record_toggled(self, enabled: bool) -> None:
        if not enabled:
            self.issue(SetRecording(None))
            return
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = QtWidgets.QFileDialog.getExistingDirectory(self, "Save recording in")
        if not base:
            self.record_action.setChecked(False)
            return
        self.issue(SetRecording(Path(base) / f"recording-{stamp}"))

    def issue(self, command: Command) -> asyncio.Future[None]:
        """Dispatch ``command`` on the event loop; failures are shown, not raised."""
        future = asyncio.ensure_future(self._container.dispatcher.dispatch(command))
        future.add_done_callback(self._after_command)
        return future

    def _after_command(self, future: asyncio.Future[Any]) -> None:
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            self._show_error(str(error))

    def spawn(self, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Future[Any]:
        return asyncio.ensure_future(coroutine)

    def on_update(self, update: SessionUpdate) -> None:
        self._pending = update

    def flush_pending(self) -> bool:
        """Render the newest queued update; returns whether anything was drawn."""
        update, self._pending = self._pending, None
        if update is None:
            return False
        self.draw_update(update)
        return True

    def draw_update(self, update: SessionUpdate) -> None:
        self.frames_rendered += 1
        if update.surface is not None:
            self.surface_panel.set_mesh(surface_mesh(update.surface))
            self.term.set_term_structure(term_structure(update.surface))
        self.smile.set_update(update)
        self.chain.set_table(chain_table(update))
        self.gex.set_view(gex_view(update))
        self._source_label.setText(
            f"{update.source}  {update.chain.underlying}  spot {update.chain.spot:.2f}  "
        )
        snapshot = self._container.diagnostics.snapshot()
        self._rate_label.setText(f"{snapshot.update_rate_hz:.1f} Hz  ")
        if update.warnings:
            self.statusBar().showMessage("; ".join(update.warnings), 4000)

    def on_state(self, event: SessionStateChanged) -> None:
        self._apply_state(event.state, event.detail)

    def _apply_state(self, state: SessionState, detail: str = "") -> None:
        self._state = state
        self._state_label.setText(state.value + (f" ({detail})" if detail else ""))
        self.start_action.setEnabled(state in _IDLE_STATES)
        self.stop_action.setEnabled(state not in _IDLE_STATES)
        self.freeze_action.setEnabled(state in _LIVE_STATES)
        with QtCore.QSignalBlocker(self.freeze_action):
            self.freeze_action.setChecked(state is SessionState.FROZEN)
        self.source.setEnabled(state in _IDLE_STATES)
        self.symbol.setEnabled(state in _IDLE_STATES and self.source.currentText() != "replay")

    def on_alert(self, alert: Alert) -> None:
        self.gex.add_alert(alert)
        self.statusBar().showMessage(alert.message, 6000)

    def _show_error(self, message: str) -> None:
        self.last_error = message
        self.statusBar().showMessage(f"error: {message}", 10000)

    def _restore_layout(self) -> bool:
        geometry = self._settings.value("window/geometry")
        state = self._settings.value("window/state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is None:
            return False
        return bool(self.restoreState(state))

    def _apply_default_layout(self) -> None:
        vertical, horizontal = QtCore.Qt.Orientation.Vertical, QtCore.Qt.Orientation.Horizontal
        docks = self._docks
        self.resizeDocks([docks["Option chain"]], [250], vertical)
        self.resizeDocks(
            [docks["Smile and SVI fit"], docks["Gamma exposure"]], [420, 280], vertical
        )
        self.resizeDocks([docks["Diagnostics"], docks["Smile and SVI fit"]], [270, 560], horizontal)

    def _reset_layout(self) -> None:
        self._settings.remove("window/geometry")
        self._settings.remove("window/state")
        for dock in self._docks.values():
            dock.setFloating(False)
            dock.show()
        self._apply_default_layout()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._settings.setValue("window/geometry", self.saveGeometry())
        self._settings.setValue("window/state", self.saveState())
        self._render_timer.stop()
        self._bridge.close()
        super().closeEvent(event)
        self.closed.emit()
