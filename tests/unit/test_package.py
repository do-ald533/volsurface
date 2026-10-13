from typer.testing import CliRunner

import volsurface
from volsurface.cli import app


def test_version_is_exposed() -> None:
    assert volsurface.__version__


def test_cli_version_command_prints_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == volsurface.__version__
