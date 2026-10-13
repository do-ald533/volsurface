from pathlib import Path

import pytest

from volsurface.domain.errors import ConfigurationError
from volsurface.domain.ports import MarketDataProvider
from volsurface.marketdata.factory import (
    ProviderFactory,
    ProviderOptions,
    SourceSpec,
    default_factory,
)
from volsurface.marketdata.ibkr import IbkrProvider
from volsurface.marketdata.replay import ReplayProvider
from volsurface.marketdata.synthetic import SyntheticConfig, SyntheticProvider


class TestSourceSpec:
    @pytest.mark.parametrize(
        ("text", "kind", "symbol"),
        [
            ("synthetic", "synthetic", "SYN"),
            ("synthetic:abc", "synthetic", "ABC"),
            ("ibkr", "ibkr", "SPY"),
            ("ibkr:qqq", "ibkr", "QQQ"),
            ("  ibkr:iwm  ", "ibkr", "IWM"),
        ],
    )
    def test_parses_symbol_sources(self, text: str, kind: str, symbol: str) -> None:
        spec = SourceSpec.parse(text)
        assert (spec.kind, spec.symbol) == (kind, symbol)

    def test_parses_replay_paths(self) -> None:
        spec = SourceSpec.parse("replay:/data/rec")
        assert spec.kind == "replay"
        assert spec.path == Path("/data/rec")
        assert str(spec) == "replay:/data/rec"

    def test_string_round_trip(self) -> None:
        for text in ("synthetic:SYN", "ibkr:SPY"):
            assert str(SourceSpec.parse(text)) == text

    @pytest.mark.parametrize("text", ["", "csv:x", "replay", "replay:"])
    def test_rejects_bad_sources(self, text: str) -> None:
        with pytest.raises(ConfigurationError):
            SourceSpec.parse(text)


class TestFactory:
    def test_builds_each_kind_of_provider(self, tmp_path: Path) -> None:
        factory = default_factory()
        synthetic = factory.create(SourceSpec.parse("synthetic:ABC"))
        replay = factory.create(SourceSpec("replay", path=tmp_path))
        ibkr = factory.create(SourceSpec.parse("ibkr:QQQ"))
        assert isinstance(synthetic, SyntheticProvider)
        assert isinstance(replay, ReplayProvider)
        assert isinstance(ibkr, IbkrProvider)
        for provider in (synthetic, replay, ibkr):
            assert isinstance(provider, MarketDataProvider)

    def test_synthetic_symbol_overrides_the_config(self) -> None:
        options = ProviderOptions(synthetic=SyntheticConfig(underlying="OLD", spot0=250.0))
        provider = default_factory().create(SourceSpec.parse("synthetic:NEW"), options)
        assert isinstance(provider, SyntheticProvider)
        assert provider.config.underlying == "NEW"
        assert provider.config.spot0 == 250.0

    def test_unknown_kind_and_custom_builders(self) -> None:
        empty = ProviderFactory()
        with pytest.raises(ConfigurationError, match="no provider registered"):
            empty.create(SourceSpec.parse("synthetic"))
        empty.register("synthetic", lambda spec, options: SyntheticProvider())
        assert isinstance(empty.create(SourceSpec.parse("synthetic")), SyntheticProvider)

    def test_replay_without_a_path_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="needs a path"):
            default_factory().create(SourceSpec("replay"))
