"""The market session: wires a provider, the pipeline and the event bus together.

Flow, all event-driven: the provider publishes ``ChainReceived``; this session handles it on a
``LATEST`` subscription (so a slow pipeline conflates stale chains instead of falling behind),
runs the pipeline on a worker thread and publishes ``SurfaceComputed``, ``StageTimed`` and
``AlertRaised``. Lifecycle changes go through :class:`SessionStateMachine` and surface as
``SessionStateChanged`` events.
"""

import asyncio
import itertools

from volsurface.app.events import (
    AlertRaised,
    ComputeFailed,
    SessionStateChanged,
    StageTimed,
    SurfaceComputed,
)
from volsurface.app.logs import get_logger
from volsurface.app.pipeline import SurfacePipeline
from volsurface.app.session_state import SessionState, SessionStateMachine
from volsurface.domain.errors import IllegalTransitionError, ProviderError
from volsurface.domain.events import ChainReceived, ConnectionState, ConnectionStateChanged
from volsurface.domain.ports import DeliveryPolicy, EventBus, MarketDataProvider
from volsurface.marketdata.factory import ProviderFactory, ProviderOptions, SourceSpec

_log = get_logger(__name__)

_CONNECTION_TO_SESSION = {
    ConnectionState.CONNECTED: SessionState.STREAMING,
    ConnectionState.RECONNECTING: SessionState.RECONNECTING,
    ConnectionState.FAILED: SessionState.FAILED,
    ConnectionState.DISCONNECTED: SessionState.STOPPED,
}


class MarketSession:
    def __init__(
        self,
        bus: EventBus,
        pipeline: SurfacePipeline,
        factory: ProviderFactory,
        options: ProviderOptions | None = None,
    ) -> None:
        self._bus = bus
        self._pipeline = pipeline
        self._factory = factory
        self._options = options or ProviderOptions()
        self._machine = SessionStateMachine(self._on_state_change)
        self._provider: MarketDataProvider | None = None
        self._sequence = itertools.count(1)
        self._processing = False
        self._chain_subscription = bus.subscribe(
            ChainReceived,
            self._on_chain,
            policy=DeliveryPolicy.LATEST,
            name="session.pipeline",
        )
        self._subscriptions = [
            self._chain_subscription,
            bus.subscribe(ConnectionStateChanged, self._on_connection, name="session.connection"),
        ]

    @property
    def state(self) -> SessionState:
        return self._machine.state

    @property
    def source(self) -> str | None:
        return self._provider.name if self._provider else None

    async def start(self, spec: SourceSpec) -> None:
        if self._machine.is_running:
            raise IllegalTransitionError(f"a session is already {self._machine.state}")
        self._machine.move_to(SessionState.CONNECTING, str(spec))
        self._pipeline.reset()
        self._sequence = itertools.count(1)
        try:
            provider = self._factory.create(spec, self._options)
            self._provider = provider
            await provider.start(self._bus)
        except Exception as exc:
            self._provider = None
            if self._machine.can_move_to(SessionState.FAILED):
                self._machine.move_to(SessionState.FAILED, str(exc))
            raise ProviderError(f"could not start {spec}: {exc}") from exc

    async def stop(self) -> None:
        provider, self._provider = self._provider, None
        if provider is not None:
            await provider.stop()
        if self._machine.can_move_to(SessionState.STOPPED):
            self._machine.move_to(SessionState.STOPPED, "stopped by user")

    async def freeze(self) -> None:
        self._machine.move_to(SessionState.FROZEN, "frozen by user")

    async def resume(self) -> None:
        self._machine.move_to(SessionState.STREAMING, "resumed by user")

    async def close(self) -> None:
        await self.stop()
        for subscription in self._subscriptions:
            subscription.cancel()

    def _on_state_change(self, previous: SessionState, state: SessionState, detail: str) -> None:
        _log.info("session_state", previous=previous.value, state=state.value, detail=detail)
        self._bus.publish(SessionStateChanged(previous=previous, state=state, detail=detail))

    async def _on_connection(self, event: ConnectionStateChanged) -> None:
        target = _CONNECTION_TO_SESSION[event.state]
        if target is SessionState.STOPPED:
            await self._wait_for_pipeline()
        if self._machine.state is target or not self._machine.can_move_to(target):
            return
        if target is SessionState.STREAMING and self._machine.state is SessionState.FROZEN:
            return
        self._machine.move_to(target, event.detail or event.source)

    async def _wait_for_pipeline(self) -> None:
        """Let chains that arrived before the source ended finish processing."""
        while self._chain_subscription.pending or self._processing:
            await asyncio.sleep(0.005)

    async def _on_chain(self, event: ChainReceived) -> None:
        if self._machine.state is not SessionState.STREAMING:
            return
        sequence = next(self._sequence)
        self._processing = True
        try:
            update = await asyncio.to_thread(
                self._pipeline.process, event.snapshot, seq=sequence, source=event.source
            )
        except Exception as exc:
            _log.exception("pipeline_failed", source=event.source)
            self._bus.publish(ComputeFailed(source=event.source, error=repr(exc)))
            return
        finally:
            self._processing = False
        if self._machine.state is not SessionState.STREAMING:
            return
        self._bus.publish(SurfaceComputed(update=update))
        for stage, seconds in update.timings.items():
            self._bus.publish(StageTimed(stage=stage, seconds=seconds, count=update.chain.size))
        for alert in update.alerts:
            self._bus.publish(AlertRaised(alert=alert))
