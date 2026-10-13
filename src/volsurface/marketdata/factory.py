"""Provider factory: turns a textual source description into a ``MarketDataProvider``.

Source strings, as typed on the command line or chosen in the GUI:

* ``synthetic`` or ``synthetic:SYMBOL``
* ``replay:/path/to/recording.parquet``
* ``ibkr:SYMBOL``

Concrete providers register a builder, so adding a source means adding a builder and nothing else
(Factory + Registry). The Interactive Brokers builder imports ``ib_async`` lazily, which keeps the
rest of the application importable without that optional dependency.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from volsurface.domain.errors import ConfigurationError
from volsurface.domain.ports import MarketDataProvider
from volsurface.marketdata.options import IbkrOptions
from volsurface.marketdata.replay import ReplayProvider
from volsurface.marketdata.synthetic import SyntheticConfig, SyntheticProvider

SourceKind = Literal["synthetic", "replay", "ibkr"]
KINDS: tuple[SourceKind, ...] = ("synthetic", "replay", "ibkr")


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Which market-data source to use."""

    kind: SourceKind
    symbol: str = "SPY"
    path: Path | None = None

    @classmethod
    def parse(cls, text: str) -> "SourceSpec":
        kind, _, rest = text.strip().partition(":")
        if kind not in KINDS:
            raise ConfigurationError(f"unknown source {kind!r}; expected one of {', '.join(KINDS)}")
        if kind == "replay":
            if not rest:
                raise ConfigurationError("replay needs a file: replay:/path/to/recording.parquet")
            return cls("replay", path=Path(rest).expanduser())
        return cls(kind, symbol=rest.upper() or ("SYN" if kind == "synthetic" else "SPY"))

    def __str__(self) -> str:
        if self.kind == "replay":
            return f"replay:{self.path}"
        return f"{self.kind}:{self.symbol}"


@dataclass(frozen=True, slots=True)
class ProviderOptions:
    """Settings the builders need that do not belong in a :class:`SourceSpec`."""

    ibkr: IbkrOptions = field(default_factory=IbkrOptions)
    synthetic: SyntheticConfig | None = None
    replay_speed: float = 1.0


type Builder = Callable[[SourceSpec, ProviderOptions], MarketDataProvider]


def _build_synthetic(spec: SourceSpec, options: ProviderOptions) -> MarketDataProvider:
    base = options.synthetic or SyntheticConfig()
    return SyntheticProvider(replace(base, underlying=spec.symbol))


def _build_replay(spec: SourceSpec, options: ProviderOptions) -> MarketDataProvider:
    if spec.path is None:
        raise ConfigurationError("replay source needs a path")
    return ReplayProvider(spec.path, speed=options.replay_speed)


def _build_ibkr(spec: SourceSpec, options: ProviderOptions) -> MarketDataProvider:
    from volsurface.marketdata.ibkr import IbkrProvider  # noqa: PLC0415

    return IbkrProvider(spec.symbol, options.ibkr)


class ProviderFactory:
    """Registry of builders keyed by source kind."""

    def __init__(self, builders: Mapping[str, Builder] | None = None) -> None:
        self._builders: dict[str, Builder] = dict(builders or {})

    def register(self, kind: str, builder: Builder) -> None:
        self._builders[kind] = builder

    def create(
        self, spec: SourceSpec, options: ProviderOptions | None = None
    ) -> MarketDataProvider:
        try:
            builder = self._builders[spec.kind]
        except KeyError:
            raise ConfigurationError(f"no provider registered for source {spec.kind!r}") from None
        return builder(spec, options or ProviderOptions())


def default_factory() -> ProviderFactory:
    return ProviderFactory(
        {"synthetic": _build_synthetic, "replay": _build_replay, "ibkr": _build_ibkr}
    )
