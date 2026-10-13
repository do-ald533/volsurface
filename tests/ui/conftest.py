import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import pytest
import qasync
from PySide6 import QtWidgets

from helpers import small_chain
from volsurface.app.config import PipelineSettings
from volsurface.app.pipeline import SurfacePipeline
from volsurface.app.results import SessionUpdate
from volsurface.pricing.backends import NumbaCpuBackend
from volsurface.ui.theme import apply_theme

T = TypeVar("T")


@pytest.fixture(scope="session")
def themed_app(qapp: QtWidgets.QApplication) -> QtWidgets.QApplication:
    apply_theme(qapp)
    return qapp


@pytest.fixture(scope="session")
def update() -> SessionUpdate:
    pipeline = SurfacePipeline(
        NumbaCpuBackend(), PipelineSettings(tree_steps=60, gex_tree_steps=40)
    )
    return pipeline.process(small_chain(), seq=1, source="synthetic")


@pytest.fixture
def run_qt(themed_app: QtWidgets.QApplication) -> Callable[[Callable[[], Awaitable[T]]], T]:
    """Run a coroutine on a qasync loop, the way the real application does."""

    def run(factory: Callable[[], Awaitable[T]]) -> T:
        loop = qasync.QEventLoop(themed_app)
        asyncio.set_event_loop(loop)
        themed_app.setQuitOnLastWindowClosed(False)
        try:
            return loop.run_until_complete(factory())
        finally:
            themed_app.setQuitOnLastWindowClosed(True)
            loop.close()
            asyncio.set_event_loop(None)

    return run
