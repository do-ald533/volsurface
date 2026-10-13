"""Runtime diagnostics gathered from events, for the diagnostics panel.

The collector is a pure subscriber: it listens to the same events as everything else and keeps
smoothed timings and counters. Nothing else in the application knows it exists (Observer).
"""

import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from volsurface.app.events import (
    ComputeFailed,
    SessionStateChanged,
    StageTimed,
    SurfaceComputed,
)
from volsurface.app.session_state import SessionState
from volsurface.domain.events import HandlerFailed
from volsurface.domain.ports import EventSubscriber
from volsurface.eventing.bus import SubscriptionStats

_EMA_ALPHA = 0.2
_RATE_WINDOW = 20


@dataclass(frozen=True, slots=True)
class DiagnosticsSnapshot:
    state: SessionState
    updates: int
    compute_failures: int
    handler_failures: int
    update_rate_hz: float
    options_per_update: int
    stage_seconds: Mapping[str, float]
    backend_seconds: Mapping[str, float]
    last_error: str
    subscriptions: tuple[SubscriptionStats, ...] = field(default=())

    @property
    def total_seconds(self) -> float:
        return sum(v for k, v in self.stage_seconds.items() if k in _PIPELINE_STAGES)


_PIPELINE_STAGES = frozenset({"filter", "implied_vol", "surface", "gex"})


class DiagnosticsCollector:
    def __init__(
        self, bus: EventSubscriber, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._clock = clock
        self._state = SessionState.IDLE
        self._updates = 0
        self._compute_failures = 0
        self._handler_failures = 0
        self._rows = 0
        self._stage: dict[str, float] = {}
        self._backend: dict[str, float] = {}
        self._arrivals: deque[float] = deque(maxlen=_RATE_WINDOW)
        self._last_error = ""
        bus.subscribe(SurfaceComputed, self._on_update, name="diagnostics.update")
        bus.subscribe(StageTimed, self._on_stage, name="diagnostics.stage")
        bus.subscribe(ComputeFailed, self._on_compute_failed, name="diagnostics.compute_failed")
        bus.subscribe(HandlerFailed, self._on_handler_failed, name="diagnostics.handler_failed")
        bus.subscribe(SessionStateChanged, self._on_state, name="diagnostics.state")

    def snapshot(self, subscriptions: tuple[SubscriptionStats, ...] = ()) -> DiagnosticsSnapshot:
        return DiagnosticsSnapshot(
            state=self._state,
            updates=self._updates,
            compute_failures=self._compute_failures,
            handler_failures=self._handler_failures,
            update_rate_hz=self._rate(),
            options_per_update=self._rows,
            stage_seconds=dict(self._stage),
            backend_seconds=dict(self._backend),
            last_error=self._last_error,
            subscriptions=subscriptions,
        )

    def _rate(self) -> float:
        if len(self._arrivals) < 2:
            return 0.0
        span = self._arrivals[-1] - self._arrivals[0]
        return (len(self._arrivals) - 1) / span if span > 0 else 0.0

    def _on_update(self, event: SurfaceComputed) -> None:
        self._updates += 1
        self._rows = event.update.chain.size
        self._arrivals.append(self._clock())

    def _on_stage(self, event: StageTimed) -> None:
        target = self._backend if event.stage.startswith("backend.") else self._stage
        key = event.stage.removeprefix("backend.")
        previous = target.get(key)
        target[key] = (
            event.seconds
            if previous is None
            else (1 - _EMA_ALPHA) * previous + _EMA_ALPHA * event.seconds
        )

    def _on_compute_failed(self, event: ComputeFailed) -> None:
        self._compute_failures += 1
        self._last_error = event.error

    def _on_handler_failed(self, event: HandlerFailed) -> None:
        self._handler_failures += 1
        self._last_error = f"{event.handler}: {event.error}"

    def _on_state(self, event: SessionStateChanged) -> None:
        self._state = event.state
