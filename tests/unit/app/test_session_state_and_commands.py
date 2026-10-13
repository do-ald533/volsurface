from pathlib import Path

import pytest

from volsurface.app.commands import (
    CommandDispatcher,
    Freeze,
    Resume,
    SetRecording,
    StartSession,
    StopSession,
)
from volsurface.app.session_state import SessionState, SessionStateMachine
from volsurface.domain.errors import ConfigurationError, IllegalTransitionError
from volsurface.marketdata.factory import SourceSpec

S = SessionState


class TestStateMachine:
    def test_happy_path(self) -> None:
        machine = SessionStateMachine()
        assert machine.state is S.IDLE
        for target in (S.CONNECTING, S.STREAMING, S.FROZEN, S.STREAMING, S.STOPPED):
            machine.move_to(target)
        assert machine.state is S.STOPPED

    @pytest.mark.parametrize(
        ("start", "target"),
        [
            (S.IDLE, S.STREAMING),
            (S.IDLE, S.FROZEN),
            (S.CONNECTING, S.FROZEN),
            (S.FROZEN, S.CONNECTING),
            (S.STOPPED, S.STREAMING),
            (S.FAILED, S.STREAMING),
        ],
    )
    def test_illegal_transitions_raise(self, start: SessionState, target: SessionState) -> None:
        machine = SessionStateMachine()
        machine._state = start
        assert not machine.can_move_to(target)
        with pytest.raises(IllegalTransitionError):
            machine.move_to(target)
        assert machine.state is start

    def test_every_state_can_be_stopped_except_a_stopped_one(self) -> None:
        for state in S:
            machine = SessionStateMachine()
            machine._state = state
            assert machine.can_move_to(S.STOPPED) == (state is not S.STOPPED)

    def test_failed_and_stopped_sessions_can_be_restarted(self) -> None:
        for state in (S.FAILED, S.STOPPED):
            machine = SessionStateMachine()
            machine._state = state
            assert machine.can_move_to(S.CONNECTING)

    def test_listener_sees_every_change_with_detail(self) -> None:
        seen: list[tuple[SessionState, SessionState, str]] = []
        machine = SessionStateMachine(lambda a, b, d: seen.append((a, b, d)))
        machine.move_to(S.CONNECTING, "spy")
        machine.move_to(S.STREAMING)
        assert seen == [(S.IDLE, S.CONNECTING, "spy"), (S.CONNECTING, S.STREAMING, "")]

    def test_is_running(self) -> None:
        running = {S.CONNECTING, S.STREAMING, S.FROZEN, S.RECONNECTING}
        for state in S:
            machine = SessionStateMachine()
            machine._state = state
            assert machine.is_running == (state in running)


class TestCommandDispatcher:
    async def test_routes_each_command_to_its_handler(self) -> None:
        calls: list[str] = []
        dispatcher = CommandDispatcher()

        async def on_start(command: StartSession) -> None:
            calls.append(f"start {command.source}")

        async def on_freeze(_: Freeze) -> None:
            calls.append("freeze")

        dispatcher.register(StartSession, on_start)
        dispatcher.register(Freeze, on_freeze)
        await dispatcher.dispatch(StartSession(SourceSpec.parse("synthetic:SYN")))
        await dispatcher.dispatch(Freeze())
        assert calls == ["start synthetic:SYN", "freeze"]

    async def test_unknown_command_is_an_error(self) -> None:
        with pytest.raises(ConfigurationError, match="no handler"):
            await CommandDispatcher().dispatch(Resume())

    async def test_one_handler_per_command(self) -> None:
        dispatcher = CommandDispatcher()

        async def handler(_: StopSession) -> None:
            return None

        dispatcher.register(StopSession, handler)
        with pytest.raises(ConfigurationError, match="already has a handler"):
            dispatcher.register(StopSession, handler)

    def test_commands_are_immutable_values(self) -> None:
        assert SetRecording(None) == SetRecording(None)
        with pytest.raises(AttributeError):
            SetRecording(None).path = Path("elsewhere")  # type: ignore[misc]
