from pathlib import Path

import pytest

from volsurface.app.config import IbkrSettings, PipelineSettings, Settings
from volsurface.domain.errors import ConfigurationError


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in list(__import__("os").environ):
        if name.startswith("VOLSURFACE_"):
            monkeypatch.delenv(name)


def test_defaults_match_the_documented_values() -> None:
    settings = Settings()
    assert settings.backend == "numba-cpu"
    assert settings.ibkr.port == 7497
    assert settings.ibkr.client_id == 35
    assert settings.pipeline.max_days_to_expiry == 120
    assert settings.pipeline.tree_steps == 200
    assert settings.pipeline.flip_grid_points == 81


def test_environment_overrides_nested_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOLSURFACE_IBKR__PORT", "4001")
    monkeypatch.setenv("VOLSURFACE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("VOLSURFACE_PIPELINE__TREE_STEPS", "150")
    settings = Settings()
    assert settings.ibkr.port == 4001
    assert settings.log_level == "DEBUG"
    assert settings.pipeline.tree_steps == 150


def test_dotenv_file_in_the_working_directory_is_read(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("VOLSURFACE_IBKR__HOST=10.1.2.3\n")
    assert Settings().ibkr.host == "10.1.2.3"


def test_toml_file_takes_priority_over_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VOLSURFACE_IBKR__PORT", "4001")
    path = tmp_path / "volsurface.toml"
    path.write_text('log_level = "WARNING"\n[ibkr]\nport = 4002\n[pipeline]\ncompute_gex = false\n')
    settings = Settings.load(path)
    assert settings.ibkr.port == 4002
    assert settings.log_level == "WARNING"
    assert settings.pipeline.compute_gex is False


def test_load_without_a_file_uses_the_environment() -> None:
    assert Settings.load().ibkr.port == 7497


def test_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        Settings.load(tmp_path / "absent.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml")
    with pytest.raises(ConfigurationError, match="invalid TOML"):
        Settings.load(bad)


@pytest.mark.parametrize(
    "toml",
    [
        "[ibkr]\nport = 70000\n",
        "[pipeline]\nflip_grid_points = 80\n",
        'log_level = "LOUD"\n',
        "[pipeline]\ntree_steps = 2\n",
    ],
)
def test_invalid_values_are_rejected_with_a_configuration_error(toml: str, tmp_path: Path) -> None:
    path = tmp_path / "volsurface.toml"
    path.write_text(toml)
    with pytest.raises(ConfigurationError, match="invalid settings"):
        Settings.load(path)


def test_settings_are_immutable() -> None:
    settings = Settings()
    with pytest.raises(Exception, match="frozen"):
        settings.log_level = "DEBUG"  # type: ignore[misc]
    with pytest.raises(Exception, match="frozen"):
        IbkrSettings().port = 1  # type: ignore[misc]
    assert PipelineSettings().compute_gex is True
