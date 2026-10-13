"""Pricing backends: the Strategy that lets the CPU tree be swapped for another implementation.

* :class:`PricingBackend` is the structural interface callers depend on.
* :class:`NumbaCpuBackend` is the default implementation.
* :class:`TimedBackend` is a Decorator that reports how long each call took, without the wrapped
  backend or its callers knowing.
* :class:`BackendRegistry` is a Registry + Factory. Third-party packages (for example a CUDA
  backend) register themselves through the ``volsurface.pricing_backends`` entry-point group, so
  adding one needs no change to this code base.
"""

import logging
import time
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from volsurface.domain.errors import ConfigurationError
from volsurface.domain.types import FloatArray
from volsurface.pricing import american as american_pricing

_log = logging.getLogger(__name__)

ENTRY_POINT_GROUP: Final = "volsurface.pricing_backends"
DEFAULT_BACKEND: Final = "numba-cpu"


@runtime_checkable
class PricingBackend(Protocol):
    """Everything the pipeline needs from an American-option pricing engine."""

    @property
    def name(self) -> str: ...

    def warmup(self) -> None:
        """Pay one-off compilation costs now rather than on the first real call."""
        ...

    def american_price_gamma(
        self,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: ArrayLike,
        q: ArrayLike,
        sigma: ArrayLike,
        is_call: ArrayLike,
        *,
        steps: int,
        american: bool = True,
    ) -> tuple[FloatArray, FloatArray]: ...

    def american_implied_vol(
        self,
        price: ArrayLike,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: ArrayLike,
        q: ArrayLike,
        is_call: ArrayLike,
        *,
        steps: int,
        american: bool = True,
    ) -> FloatArray: ...

    def american_gamma_grid(
        self,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: ArrayLike,
        q: ArrayLike,
        sigma: ArrayLike,
        is_call: ArrayLike,
        levels: ArrayLike,
        *,
        steps: int,
        american: bool = True,
    ) -> FloatArray: ...


class NumbaCpuBackend:
    """Multi-threaded Numba implementation (the default)."""

    name = DEFAULT_BACKEND

    def warmup(self) -> None:
        price, _ = self.american_price_gamma(100.0, 100.0, 0.25, 0.03, 0.0, 0.2, True, steps=8)
        self.american_implied_vol(price, 100.0, 100.0, 0.25, 0.03, 0.0, True, steps=8)
        self.american_gamma_grid(100.0, 100.0, [0.25], 0.03, 0.0, 0.2, True, [1.0], steps=8)

    def american_price_gamma(
        self,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: ArrayLike,
        q: ArrayLike,
        sigma: ArrayLike,
        is_call: ArrayLike,
        *,
        steps: int,
        american: bool = True,
    ) -> tuple[FloatArray, FloatArray]:
        return american_pricing.american_price_gamma(
            S, K, T, r, q, sigma, is_call, steps=steps, american=american
        )

    def american_implied_vol(
        self,
        price: ArrayLike,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: ArrayLike,
        q: ArrayLike,
        is_call: ArrayLike,
        *,
        steps: int,
        american: bool = True,
    ) -> FloatArray:
        return american_pricing.american_implied_vol(
            price, S, K, T, r, q, is_call, steps=steps, american=american
        )

    def american_gamma_grid(
        self,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: ArrayLike,
        q: ArrayLike,
        sigma: ArrayLike,
        is_call: ArrayLike,
        levels: ArrayLike,
        *,
        steps: int,
        american: bool = True,
    ) -> FloatArray:
        return american_pricing.american_gamma_grid(
            S, K, T, r, q, sigma, is_call, levels, steps=steps, american=american
        )


type TimingCallback = Callable[[str, float, int], None]
"""Receives ``(operation, seconds, number_of_options)`` after each timed call."""


class TimedBackend:
    """Decorator: forwards every call to ``inner`` and reports its duration."""

    def __init__(self, inner: PricingBackend, on_timing: TimingCallback) -> None:
        self._inner = inner
        self._on_timing = on_timing

    @property
    def name(self) -> str:
        return self._inner.name

    def warmup(self) -> None:
        self._inner.warmup()

    def _timed[T](self, operation: str, count: int, call: Callable[[], T]) -> T:
        start = time.perf_counter()
        try:
            return call()
        finally:
            self._on_timing(operation, time.perf_counter() - start, count)

    def american_price_gamma(
        self,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: ArrayLike,
        q: ArrayLike,
        sigma: ArrayLike,
        is_call: ArrayLike,
        *,
        steps: int,
        american: bool = True,
    ) -> tuple[FloatArray, FloatArray]:
        return self._timed(
            "american_price_gamma",
            _size(K),
            lambda: self._inner.american_price_gamma(
                S, K, T, r, q, sigma, is_call, steps=steps, american=american
            ),
        )

    def american_implied_vol(
        self,
        price: ArrayLike,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: ArrayLike,
        q: ArrayLike,
        is_call: ArrayLike,
        *,
        steps: int,
        american: bool = True,
    ) -> FloatArray:
        return self._timed(
            "american_implied_vol",
            _size(price),
            lambda: self._inner.american_implied_vol(
                price, S, K, T, r, q, is_call, steps=steps, american=american
            ),
        )

    def american_gamma_grid(
        self,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        r: ArrayLike,
        q: ArrayLike,
        sigma: ArrayLike,
        is_call: ArrayLike,
        levels: ArrayLike,
        *,
        steps: int,
        american: bool = True,
    ) -> FloatArray:
        return self._timed(
            "american_gamma_grid",
            _size(K),
            lambda: self._inner.american_gamma_grid(
                S, K, T, r, q, sigma, is_call, levels, steps=steps, american=american
            ),
        )


def _size(values: ArrayLike) -> int:
    return int(np.size(values))


type BackendFactory = Callable[[], PricingBackend]


class BackendRegistry:
    """Maps backend names to factories."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory, *, replace: bool = False) -> None:
        if name in self._factories and not replace:
            raise ConfigurationError(f"pricing backend {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name: str = DEFAULT_BACKEND) -> PricingBackend:
        try:
            factory = self._factories[name]
        except KeyError:
            known = ", ".join(sorted(self._factories)) or "none"
            raise ConfigurationError(
                f"unknown pricing backend {name!r} (registered: {known})"
            ) from None
        return factory()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def load_entry_points(self, group: str = ENTRY_POINT_GROUP) -> list[str]:
        """Register backends advertised by installed packages; a broken plugin is skipped."""
        loaded: list[str] = []
        for ep in entry_points(group=group):
            try:
                self.register(ep.name, ep.load())
            except Exception:
                _log.warning("could not load pricing backend plugin %r", ep.name, exc_info=True)
            else:
                loaded.append(ep.name)
        return loaded


def default_registry() -> BackendRegistry:
    """A registry with the built-in CPU backend plus any installed plugins."""
    registry = BackendRegistry()
    registry.register(DEFAULT_BACKEND, NumbaCpuBackend)
    registry.load_entry_points()
    return registry
