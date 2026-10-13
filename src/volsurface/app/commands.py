"""User intents as command objects, routed to one handler each (Command pattern).

The GUI, the CLI and tests all talk to the session through the same small vocabulary, so none of
them reaches into session internals and each action can be exercised without a window.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from volsurface.domain.errors import ConfigurationError
from volsurface.marketdata.factory import SourceSpec


@dataclass(frozen=True, slots=True)
class Command:
    """Base class of every command."""


@dataclass(frozen=True, slots=True)
class StartSession(Command):
    source: SourceSpec


@dataclass(frozen=True, slots=True)
class SwitchSource(Command):
    """Stop the running session, if any, and start ``source`` in its place."""

    source: SourceSpec


@dataclass(frozen=True, slots=True)
class StopSession(Command):
    pass


@dataclass(frozen=True, slots=True)
class Freeze(Command):
    pass


@dataclass(frozen=True, slots=True)
class Resume(Command):
    pass


@dataclass(frozen=True, slots=True)
class SetRecording(Command):
    """Start recording incoming chains to ``path``, or stop when ``path`` is ``None``."""

    path: Path | None


type CommandHandler[C: Command] = Callable[[C], Awaitable[None]]


class CommandDispatcher:
    """Maps each command type to exactly one handler."""

    def __init__(self) -> None:
        self._handlers: dict[type[Command], CommandHandler[Command]] = {}

    def register[C: Command](self, command_type: type[C], handler: CommandHandler[C]) -> None:
        if command_type in self._handlers:
            raise ConfigurationError(f"{command_type.__name__} already has a handler")
        self._handlers[command_type] = handler  # type: ignore[assignment]

    async def dispatch(self, command: Command) -> None:
        handler = self._handlers.get(type(command))
        if handler is None:
            raise ConfigurationError(f"no handler registered for {type(command).__name__}")
        await handler(command)
