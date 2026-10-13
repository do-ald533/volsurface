"""Entry point of the desktop application: Qt and asyncio on one event loop."""

import asyncio
import sys
from collections.abc import Callable

import qasync
from PySide6 import QtWidgets

from volsurface.app.commands import StartSession
from volsurface.app.config import Settings
from volsurface.app.container import open_container
from volsurface.app.logs import configure_logging, get_logger
from volsurface.app.sources import SourceSpec
from volsurface.app.warmup import warm_up_numerics
from volsurface.ui.main_window import MainWindow
from volsurface.ui.qt_bridge import EventBridge
from volsurface.ui.theme import apply_theme

_log = get_logger(__name__)


async def _main(
    app: QtWidgets.QApplication,
    settings: Settings,
    source: SourceSpec | None,
    *,
    on_ready: Callable[[MainWindow], None] | None = None,
) -> int:
    """Run the window until it is closed, then shut down in order and quit.

    The application quits only after the asynchronous cleanup has finished: ending the Qt loop
    first would stop the asyncio loop with the event bus and the session still running.
    """
    done = asyncio.Event()
    app.setQuitOnLastWindowClosed(False)
    app.aboutToQuit.connect(done.set)
    async with open_container(settings, warmup=False) as container:
        bridge = EventBridge()
        bridge.bind(container.bus)
        window = MainWindow(
            container, bridge, initial_source=str(source) if source else "synthetic"
        )
        window.closed.connect(done.set)
        window.show()
        if on_ready is not None:
            on_ready(window)
        window.statusBar().showMessage("compiling numerical kernels ...")
        await asyncio.to_thread(warm_up_numerics, container.backend)
        window.statusBar().showMessage("ready", 3000)
        if source is not None:
            await container.dispatcher.dispatch(StartSession(source))
        await done.wait()
        bridge.close()
    app.quit()
    return 0


def run_gui(settings: Settings, source: SourceSpec | None = None) -> int:
    """Run the desktop application until its window closes; returns the process exit code."""
    configure_logging(settings.log_level, json=settings.log_json)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    assert isinstance(app, QtWidgets.QApplication)
    apply_theme(app)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    _log.info("gui_start", source=str(source) if source else None)
    with loop:
        exit_code: int = loop.run_until_complete(_main(app, settings, source))
    return exit_code
