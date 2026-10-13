"""The session lifecycle as a state machine with guarded transitions.

::

    IDLE -> CONNECTING -> STREAMING <-> FROZEN
                 |            |
                 v            v
              FAILED <- RECONNECTING

Every state can also go to STOPPED; a stopped or failed session can be started again. Illegal
transitions raise instead of silently corrupting the lifecycle, which keeps the UI and the
session from disagreeing about what is running.
"""

import enum
from collections.abc import Callable
from typing import Final

from volsurface.domain.errors import IllegalTransitionError


class SessionState(enum.StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    FROZEN = "frozen"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    STOPPED = "stopped"


_ALLOWED: Final[dict[SessionState, frozenset[SessionState]]] = {
    SessionState.IDLE: frozenset({SessionState.CONNECTING, SessionState.STOPPED}),
    SessionState.CONNECTING: frozenset(
        {SessionState.STREAMING, SessionState.FAILED, SessionState.STOPPED}
    ),
    SessionState.STREAMING: frozenset(
        {
            SessionState.FROZEN,
            SessionState.RECONNECTING,
            SessionState.FAILED,
            SessionState.STOPPED,
        }
    ),
    SessionState.FROZEN: frozenset(
        {
            SessionState.STREAMING,
            SessionState.RECONNECTING,
            SessionState.FAILED,
            SessionState.STOPPED,
        }
    ),
    SessionState.RECONNECTING: frozenset(
        {SessionState.STREAMING, SessionState.FAILED, SessionState.STOPPED}
    ),
    SessionState.FAILED: frozenset({SessionState.CONNECTING, SessionState.STOPPED}),
    SessionState.STOPPED: frozenset({SessionState.CONNECTING}),
}

type ChangeListener = Callable[[SessionState, SessionState, str], None]


class SessionStateMachine:
    """Holds the current state and validates every move."""

    def __init__(self, on_change: ChangeListener | None = None) -> None:
        self._state = SessionState.IDLE
        self._on_change = on_change

    @property
    def state(self) -> SessionState:
        return self._state

    def can_move_to(self, target: SessionState) -> bool:
        return target in _ALLOWED[self._state]

    def move_to(self, target: SessionState, detail: str = "") -> None:
        if not self.can_move_to(target):
            raise IllegalTransitionError(f"cannot go from {self._state} to {target}")
        previous, self._state = self._state, target
        if self._on_change is not None:
            self._on_change(previous, target, detail)

    @property
    def is_running(self) -> bool:
        return self._state in {
            SessionState.CONNECTING,
            SessionState.STREAMING,
            SessionState.FROZEN,
            SessionState.RECONNECTING,
        }
