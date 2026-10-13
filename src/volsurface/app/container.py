"""Composition root: the one place where concrete classes are chosen and wired together.

Everything else receives its collaborators through constructors (dependency injection by hand, no
framework). ``open_container`` is an async context manager so that start-up and shutdown order are
explicit and exception-safe, using :class:`contextlib.AsyncExitStack`.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from volsurface.app.commands import (
    CommandDispatcher,
    Freeze,
    Resume,
    SetRecording,
    StartSession,
    StopSession,
)
from volsurface.app.config import Settings
from volsurface.app.diagnostics import DiagnosticsCollector
from volsurface.app.events import StageTimed
from volsurface.app.pipeline import SurfacePipeline
from volsurface.app.session import MarketSession
from volsurface.eventing import AsyncEventBus
from volsurface.marketdata.factory import ProviderFactory, ProviderOptions, default_factory
from volsurface.marketdata.options import IbkrOptions
from volsurface.marketdata.recorder import EventRecorder
from volsurface.pricing.backends import PricingBackend, TimedBackend, default_registry


@dataclass(frozen=True, slots=True)
class AppContainer:
    settings: Settings
    bus: AsyncEventBus
    backend: PricingBackend
    session: MarketSession
    dispatcher: CommandDispatcher
    diagnostics: DiagnosticsCollector
    recorder: EventRecorder


def provider_options(settings: Settings) -> ProviderOptions:
    return ProviderOptions(ibkr=IbkrOptions(**settings.ibkr.model_dump()))


@asynccontextmanager
async def open_container(
    settings: Settings | None = None,
    *,
    factory: ProviderFactory | None = None,
    options: ProviderOptions | None = None,
    warmup: bool = True,
) -> AsyncIterator[AppContainer]:
    cfg = settings or Settings()
    async with AsyncExitStack() as stack:
        bus = await stack.enter_async_context(AsyncEventBus())

        def report(operation: str, seconds: float, count: int) -> None:
            bus.publish_threadsafe(
                StageTimed(stage=f"backend.{operation}", seconds=seconds, count=count)
            )

        backend: PricingBackend = TimedBackend(default_registry().create(cfg.backend), report)
        if warmup:
            await asyncio.to_thread(backend.warmup)

        pipeline = SurfacePipeline(backend, cfg.pipeline)
        session = MarketSession(
            bus, pipeline, factory or default_factory(), options or provider_options(cfg)
        )
        stack.push_async_callback(session.close)

        dispatcher = CommandDispatcher()

        async def start(command: StartSession) -> None:
            await session.start(command.source)

        async def stop(_: StopSession) -> None:
            await session.stop()

        async def freeze(_: Freeze) -> None:
            await session.freeze()

        async def resume(_: Resume) -> None:
            await session.resume()

        recorder = EventRecorder(bus)
        stack.callback(recorder.stop)

        async def record(command: SetRecording) -> None:
            if command.path is None:
                recorder.stop()
            else:
                recorder.start(command.path)

        dispatcher.register(StartSession, start)
        dispatcher.register(StopSession, stop)
        dispatcher.register(Freeze, freeze)
        dispatcher.register(Resume, resume)
        dispatcher.register(SetRecording, record)

        yield AppContainer(
            settings=cfg,
            bus=bus,
            backend=backend,
            session=session,
            dispatcher=dispatcher,
            diagnostics=DiagnosticsCollector(bus),
            recorder=recorder,
        )
