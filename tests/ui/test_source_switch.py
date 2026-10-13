import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from PySide6 import QtCore

from helpers import MarketLikeGateway, fake_options, wait_until
from volsurface.app.config import PipelineSettings, Settings
from volsurface.app.container import open_container
from volsurface.app.session_state import SessionState
from volsurface.marketdata.factory import ProviderFactory, ProviderOptions
from volsurface.marketdata.ibkr import IbkrProvider
from volsurface.marketdata.synthetic import SyntheticConfig, SyntheticProvider
from volsurface.ui.main_window import MainWindow
from volsurface.ui.qt_bridge import EventBridge

pytestmark = pytest.mark.ui


def _settings() -> Settings:
    return Settings(ui_hz=20.0, pipeline=PipelineSettings(tree_steps=50, gex_tree_steps=30))


def _factory(*, open_interest: float = float("nan"), connect_delay: float = 0.0) -> ProviderFactory:
    config = SyntheticConfig(expiry_days=(14, 30, 60), strike_step=2.0, interval_s=0.05, seed=3)
    factory = ProviderFactory()
    factory.register("synthetic", lambda spec, options: SyntheticProvider(config))
    factory.register(
        "ibkr",
        lambda spec, options: IbkrProvider(
            spec.symbol,
            options.ibkr,
            gateway_factory=lambda symbol, opts: MarketLikeGateway(
                open_interest=open_interest, connect_delay=connect_delay
            ),
            poll_interval_s=0.1,
            warmup_s=0.1,
        ),
    )
    return factory


def _ini(tmp_path: Path) -> QtCore.QSettings:
    return QtCore.QSettings(str(tmp_path / "window.ini"), QtCore.QSettings.Format.IniFormat)


@pytest.mark.parametrize("open_interest", [float("nan"), 800.0])
def test_switching_to_a_market_like_source_keeps_the_charts_alive(
    run_qt, tmp_path: Path, open_interest: float
) -> None:  # type: ignore[no-untyped-def]
    options = ProviderOptions(ibkr=replace(ProviderOptions().ibkr, max_lines=60, expiries=4))

    async def scenario() -> dict[str, object]:
        async with open_container(
            _settings(),
            factory=_factory(open_interest=open_interest),
            options=options,
            warmup=False,
        ) as app:
            bridge = EventBridge()
            bridge.bind(app.bus)
            window = MainWindow(app, bridge, settings=_ini(tmp_path))
            window.show()
            window.start_action.trigger()
            await wait_until(lambda: window.frames_rendered >= 3, timeout=30)
            before = window._source_label.text()
            before_image = window.surface_panel.heatmap._image.image.copy()

            window.source.setCurrentText("ibkr")
            window.start_action.trigger()
            await wait_until(lambda: "ibkr" in window._source_label.text(), timeout=30)
            frames_at_switch = window.frames_rendered
            await wait_until(lambda: window.frames_rendered >= frames_at_switch + 4, timeout=30)
            await asyncio.sleep(0.2)
            result = {
                "before": before,
                "after": window._source_label.text(),
                "state": app.session.state,
                "error": window.last_error,
                "chain_rows": window.chain.model.rowCount(),
                "surface_changed": not (
                    window.surface_panel.heatmap._image.image.shape == before_image.shape
                    and (window.surface_panel.heatmap._image.image == before_image).all()
                ),
                "gex_text": window.gex._summary.text() if hasattr(window.gex, "_summary") else "",
                "smile_combo": window.smile._combo.count(),
                "status": window.statusBar().currentMessage(),
            }
            window.close()
            return result

    result = run_qt(scenario)
    print(result)
    assert result["state"] is SessionState.STREAMING
    assert "synthetic" in str(result["before"])
    assert "ibkr" in str(result["after"])
    assert result["error"] == ""
    assert result["surface_changed"]
    assert int(result["chain_rows"]) > 20
    assert int(result["smile_combo"]) >= 2
    assert fake_options()


def test_a_switch_empties_the_old_charts_and_reports_progress(run_qt, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    options = ProviderOptions(ibkr=replace(ProviderOptions().ibkr, max_lines=60, expiries=4))

    async def scenario() -> dict[str, object]:
        async with open_container(
            _settings(), factory=_factory(connect_delay=0.8), options=options, warmup=False
        ) as app:
            bridge = EventBridge()
            bridge.bind(app.bus)
            window = MainWindow(app, bridge, settings=_ini(tmp_path))
            window.show()
            window.start_action.trigger()
            await wait_until(lambda: window.surface_panel.heatmap.has_data, timeout=30)
            assert window.chain.model.rowCount() > 0 and window.smile._combo.count() > 0

            window.source.setCurrentText("ibkr")
            window.start_action.trigger()
            await wait_until(lambda: "connecting to TWS" in window.surface_panel.notice, timeout=10)
            while_connecting = {
                "heatmap": window.surface_panel.heatmap.has_data,
                "rows": window.chain.model.rowCount(),
                "smile": window.smile._combo.count(),
                "term": window.term._line.getData()[0],
                "gex": window.gex._summary.text(),
                "notice": window.surface_panel.notice,
                "state": window._state_label.text(),
            }
            await wait_until(lambda: window.surface_panel.heatmap.has_data, timeout=30)
            window.close()
            return {
                "while_connecting": while_connecting,
                "notice_after": window.surface_panel.notice,
                "source": window._source_label.text(),
            }

    result = run_qt(scenario)
    waiting = result["while_connecting"]
    assert isinstance(waiting, dict)
    assert waiting["heatmap"] is False
    assert waiting["rows"] == 0 and waiting["smile"] == 0
    assert waiting["term"] is None or len(waiting["term"]) == 0
    assert waiting["gex"] == "no data yet"
    assert str(waiting["notice"]).startswith("connecting to TWS")
    assert str(waiting["state"]).startswith("connecting")
    assert result["notice_after"] == ""
    assert "ibkr" in str(result["source"])
