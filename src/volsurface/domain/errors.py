"""Exception hierarchy. Every error raised on purpose derives from :class:`VolsurfaceError`."""


class VolsurfaceError(Exception):
    """Base class for all errors raised deliberately by this package."""


class InvalidInputError(VolsurfaceError, ValueError):
    """An argument or data set violates a documented precondition."""


class PricingError(VolsurfaceError):
    """A pricing or calibration routine could not produce a result."""


class ConvergenceError(PricingError):
    """An iterative solver exhausted its budget without meeting its tolerance."""


class ProviderError(VolsurfaceError):
    """A market-data provider failed."""


class ConnectionLostError(ProviderError):
    """The link to a market-data source was lost."""


class MarketDataUnavailableError(ProviderError):
    """The source refuses to send the requested data, so retrying cannot help."""


class ConfigurationError(VolsurfaceError):
    """Configuration is missing or inconsistent."""


class BusNotRunningError(VolsurfaceError, RuntimeError):
    """An event was published to a bus that is not running (not started, or already stopped)."""


class IllegalTransitionError(VolsurfaceError):
    """A state machine was asked to make a transition that its rules do not allow."""
