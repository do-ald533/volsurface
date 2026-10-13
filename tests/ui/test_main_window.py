import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PySide6 import QtCore

from helpers import wait_until
from volsurface.analytics.alerts import Alert, AlertKind
from volsurface.app.commands import StartSession
from volsurface.app.config import PipelineSettings, Settings
from volsurface.app.container import open_container
from volsurface.app.session_state import SessionState
from volsurface.app.sources import SourceSpec
from volsurface.marketdata.factory import ProviderFactory
from volsurface.marketdata.synthetic import SyntheticConfig, SyntheticProvider
from volsurface.ui.main_window import MainWindow
from volsurface.ui.qt_bridge import EventBridge

pytestmark = pytest.mark.ui


def _settings() -> Settings:
    return Settings(ui_hz=20.0, pipeline=PipelineSettings(tree_steps=50, gex_tree_steps=30))


def _factory() -> ProviderFactory:
    config = SyntheticConfig(expiry_days=(14, 30, 60), strike_step=2.0, interval_s=0.03, seed=3)
    factory = ProviderFactory()
    factory.register("synthetic", lambda spec, options: SyntheticProvider(config))
    return factory


def _ini(tmp_path: Path) -> QtCore.QSettings:
    return QtCore.QSettings(str(tmp_path / "window.ini"), QtCore.QSettings.Format.IniFormat)


def test_window_streams_renders_and_responds_to_the_toolbar(run_qt, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> tuple[int, str, str]:
        async with open_container(_settings(), factory=_factory(), warmup=False) as app:
            bridge = EventBridge()
            bridge.bind(app.bus)
            window = MainWindow(app, bridge, settings=_ini(tmp_path))
            window.show()
            assert window.start_action.isEnabled()
            assert not window.stop_action.isEnabled()

            window.start_action.trigger()
            await wait_until(lambda: window.frames_rendered >= 3, timeout=20)
            assert app.session.state is SessionState.STREAMING
            assert window.stop_action.isEnabled()
            assert not window.start_action.isEnabled()
            assert window.freeze_action.isEnabled()
            assert window.smile._combo.count() == 3
            assert window.chain.model.rowCount() > 100
            assert window.surface_panel.heatmap.has_data
            assert "streaming" in window._state_label.text()
            shot = tmp_path / "window.png"
            assert window.grab().save(str(shot))

            window.freeze_action.setChecked(True)
            await wait_until(lambda: app.session.state is SessionState.FROZEN)
            await asyncio.sleep(0.3)
            frozen_frames = window.frames_rendered
            await asyncio.sleep(0.3)
            assert window.frames_rendered == frozen_frames
            window.freeze_action.setChecked(False)
            await wait_until(lambda: app.session.state is SessionState.STREAMING)
            await wait_until(lambda: window.frames_rendered > frozen_frames, timeout=20)

            window.stop_action.trigger()
            await wait_until(lambda: app.session.state is SessionState.STOPPED)
            assert window.start_action.isEnabled()
            state_text = window._state_label.text()
            window.close()
            return window.frames_rendered, state_text, str(shot)

    frames, state_text, shot = run_qt(scenario)
    assert frames >= 3
    assert "stopped" in state_text
    assert Path(shot).stat().st_size > 5000
    assert (tmp_path / "window.ini").exists()


def test_alerts_and_input_errors_are_surfaced(run_qt, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> tuple[int, str]:
        async with open_container(_settings(), factory=_factory(), warmup=False) as app:
            bridge = EventBridge()
            bridge.bind(app.bus)
            window = MainWindow(app, bridge, settings=_ini(tmp_path))
            window.on_alert(Alert(AlertKind.NEAR_FLIP, "SYN", datetime.now(UTC), "near flip", 0.1))
            window.source.setCurrentText("replay")
            window.start_action.trigger()
            await asyncio.sleep(0)
            return window.gex.alert_count(), window.last_error

    count, error = run_qt(scenario)
    assert count == 1
    assert "recording directory" in error


def test_starting_a_missing_replay_reports_the_error(run_qt, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    async def scenario() -> str:
        async with open_container(_settings(), warmup=False) as app:
            bridge = EventBridge()
            bridge.bind(app.bus)
            window = MainWindow(app, bridge, settings=_ini(tmp_path))
            window.issue(StartSession(SourceSpec("replay", path=tmp_path / "absent")))
            await wait_until(lambda: bool(window.last_error))
            return window.last_error

    assert "not found" in run_qt(scenario)


def test_gui_entry_runs_cleanly_and_exits_after_the_window_closes(run_qt, themed_app) -> None:  # type: ignore[no-untyped-def]
    from volsurface.ui.app import _main  # noqa: PLC0415

    frames: list[int] = []

    def when_ready(window: MainWindow) -> None:
        def close() -> None:
            frames.append(window.frames_rendered)
            window.close()

        QtCore.QTimer.singleShot(2500, close)

    settings = Settings(ui_hz=20.0, pipeline=PipelineSettings(tree_steps=50, gex_tree_steps=30))
    exit_code = run_qt(
        lambda: _main(themed_app, settings, SourceSpec.parse("synthetic:SYN"), on_ready=when_ready)
    )
    assert exit_code == 0
    assert frames
    assert frames[0] >= 1
