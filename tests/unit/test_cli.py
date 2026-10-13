from pathlib import Path

import pytest
from typer.testing import CliRunner

import volsurface
from volsurface.cli import app
from volsurface.marketdata.recording_store import RecordingStore

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == volsurface.__version__


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("gui", "demo", "record", "bench", "version"):
        assert command in result.stdout


def test_demo_prints_the_surface_gex_and_timings() -> None:
    result = runner.invoke(app, ["demo", "--ticks", "1"])
    assert result.exit_code == 0, result.output
    assert "ATM vol%" in result.stdout
    assert "static arbitrage: none found" in result.stdout
    assert "GEX at spot" in result.stdout
    assert "timings:" in result.stdout


def test_record_then_replay_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "rec"
    recorded = runner.invoke(app, ["record", str(out), "--seconds", "1.5"])
    assert recorded.exit_code == 0, recorded.output
    assert "recorded" in recorded.stdout
    assert RecordingStore(out).count() >= 2

    replayed = runner.invoke(app, ["demo", "--source", f"replay:{out}", "--ticks", "2"])
    assert replayed.exit_code == 0, replayed.output
    assert replayed.stdout.count("expiry") >= 2


@pytest.mark.parametrize(
    "args",
    [
        ["demo", "--source", "csv:x"],
        ["demo", "--source", "replay"],
        ["demo", "--source", f"replay:{Path('/nonexistent/recording')}"],
        ["gui", "--source", "nonsense"],
    ],
)
def test_bad_sources_fail_with_a_clear_message(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "error" in result.output.lower()


def test_missing_config_file_is_reported(tmp_path: Path) -> None:
    result = runner.invoke(app, ["demo", "--config", str(tmp_path / "absent.toml")])
    assert result.exit_code == 1
    assert "not found" in result.output
