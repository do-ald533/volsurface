"""Application events: results of computation and changes of session state.

These complement the domain events in :mod:`volsurface.domain.events`, which describe facts from
the outside world. They live in the application layer because they carry types (surfaces, GEX
profiles) that the domain layer must not depend on.
"""

from dataclasses import dataclass

from volsurface.analytics.alerts import Alert
from volsurface.app.results import SessionUpdate
from volsurface.app.session_state import SessionState
from volsurface.domain.events import Event


@dataclass(frozen=True, slots=True, kw_only=True)
class SurfaceComputed(Event):
    """A new chain was processed into a surface (and GEX, when open interest is available)."""

    update: SessionUpdate


@dataclass(frozen=True, slots=True, kw_only=True)
class AlertRaised(Event):
    alert: Alert


@dataclass(frozen=True, slots=True, kw_only=True)
class StageTimed(Event):
    """A named piece of work finished."""

    stage: str
    seconds: float
    count: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ComputeFailed(Event):
    """The pipeline raised while processing a chain; the session keeps running."""

    source: str
    error: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionStateChanged(Event):
    previous: SessionState
    state: SessionState
    detail: str = ""
