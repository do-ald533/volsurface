"""Draw-ready data for the user interface, computed without Qt.

Each function turns a :class:`~volsurface.app.results.SessionUpdate` (or part of one) into plain
arrays and numbers that a widget can hand straight to a plot. Keeping this logic out of the widgets
means it is unit-tested without a display, and the widgets stay thin glue.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np

from volsurface.app.diagnostics import DiagnosticsSnapshot
from volsurface.app.results import SessionUpdate
from volsurface.domain.calendar import years_to_expiry
from volsurface.domain.types import BoolArray, FloatArray
from volsurface.surface.surface import VolSurface

DAYS_PER_YEAR = 365.0
DEFAULT_K_LIMIT = 0.25


@dataclass(frozen=True, slots=True, eq=False)
class SurfaceMesh:
    """Implied volatility (in percent) on a log-moneyness by days-to-expiry grid."""

    k: FloatArray
    days: FloatArray
    iv_pct: FloatArray

    @property
    def vol_range(self) -> tuple[float, float]:
        return float(np.nanmin(self.iv_pct)), float(np.nanmax(self.iv_pct))


def surface_mesh(
    surface: VolSurface, *, n_k: int = 41, n_t: int = 30, k_limit: float = DEFAULT_K_LIMIT
) -> SurfaceMesh | None:
    """Sample ``surface`` on a regular grid inside the range its slices actually cover."""
    if not surface.slices:
        return None
    k_lo = max(-k_limit, min(s.k_min for s in surface.slices))
    k_hi = min(k_limit, max(s.k_max for s in surface.slices))
    if k_hi - k_lo < 0.02:
        k_lo, k_hi = -0.1, 0.1
    times = surface.maturities
    t_lo, t_hi = (times[0], times[-1]) if len(times) > 1 else (times[0] * 0.5, times[0] * 1.5)
    k = np.linspace(k_lo, k_hi, n_k)
    t = np.linspace(t_lo, t_hi, n_t)
    return SurfaceMesh(k=k, days=t * DAYS_PER_YEAR, iv_pct=100.0 * surface.iv_grid(k, t))


@dataclass(frozen=True, slots=True, eq=False)
class SurfaceGlArrays:
    """Arrays for ``pyqtgraph.opengl.GLSurfacePlotItem``: ``z[i, j]`` sits at ``(x[i], y[j])``."""

    x: FloatArray
    y: FloatArray
    z: FloatArray
    unit: FloatArray


def surface_gl_arrays(
    mesh: SurfaceMesh, *, extent: float = 10.0, height: float = 4.0
) -> SurfaceGlArrays:
    """Normalise a mesh into a box ``extent`` wide and ``height`` tall for the 3D view.

    ``unit`` is the volatility rescaled to ``[0, 1]`` (for colouring); ``z`` is the same scaled to
    ``[0, height]``. A flat surface maps to the middle of the range instead of dividing by zero.
    """
    lo, hi = mesh.vol_range
    span = hi - lo
    unit = np.full_like(mesh.iv_pct, 0.5) if span < 1e-9 else (mesh.iv_pct - lo) / span
    x = np.linspace(-extent / 2, extent / 2, mesh.k.size)
    y = np.linspace(-extent / 2, extent / 2, mesh.days.size)
    return SurfaceGlArrays(x=x, y=y, z=(unit * height).T, unit=unit.T)


@dataclass(frozen=True, slots=True, eq=False)
class SmileView:
    """One expiry: market quotes next to the fitted SVI curve."""

    expiry: date
    days: float
    k_points: FloatArray
    iv_points_pct: FloatArray
    vendor_iv_pct: FloatArray
    used: BoolArray
    k_curve: FloatArray
    iv_curve_pct: FloatArray
    rmse_vol_pct: float
    min_g: float
    butterfly_ok: bool
    lee_ok: bool
    n_fit_points: int


def smile_view(update: SessionUpdate, index: int) -> SmileView | None:
    """The smile of the ``index``-th fitted expiry, or ``None`` if there is none."""
    surface = update.surface
    if surface is None or not 0 <= index < len(surface.slices):
        return None
    chain, piece = update.chain, surface.slices[index]
    rows = (
        (chain.expiry == np.datetime64(piece.expiry, "D"))
        & np.isfinite(update.implied_vol)
        & update.usable
    )
    forward = piece.forward
    k_points = np.log(chain.strike[rows] / forward)
    otm = chain.is_call[rows] == (chain.strike[rows] >= forward)
    k_curve = np.linspace(piece.k_min - 0.05, piece.k_max + 0.05, 160)
    iv_curve = np.sqrt(np.maximum(piece.total_variance(k_curve), 0.0) / piece.t)
    return SmileView(
        expiry=piece.expiry,
        days=piece.t * DAYS_PER_YEAR,
        k_points=k_points,
        iv_points_pct=100.0 * update.implied_vol[rows],
        vendor_iv_pct=100.0 * chain.vendor_iv[rows],
        used=otm,
        k_curve=k_curve,
        iv_curve_pct=100.0 * iv_curve,
        rmse_vol_pct=100.0 * piece.fit.rmse_vol,
        min_g=piece.fit.min_g,
        butterfly_ok=piece.fit.butterfly_free,
        lee_ok=piece.fit.lee_ok,
        n_fit_points=piece.fit.n_points,
    )


@dataclass(frozen=True, slots=True, eq=False)
class TermStructure:
    days: FloatArray
    atm_vol_pct: FloatArray
    forward: FloatArray


def term_structure(surface: VolSurface) -> TermStructure:
    """At-the-money (forward) volatility and forward price per fitted expiry."""
    return TermStructure(
        days=surface.maturities * DAYS_PER_YEAR,
        atm_vol_pct=np.array([100.0 * s.atm_vol() for s in surface.slices]),
        forward=np.array([s.forward for s in surface.slices]),
    )


CHAIN_COLUMNS = (
    "Expiry",
    "Strike",
    "Type",
    "Bid",
    "Ask",
    "Mid",
    "OI",
    "IV %",
    "Vendor IV %",
    "Diff pts",
    "Used",
)


type CellReader = Callable[[SessionUpdate, int], str | float]


def _iv_pct(u: SessionUpdate, r: int) -> float:
    return 100.0 * float(u.implied_vol[r])


def _vendor_pct(u: SessionUpdate, r: int) -> float:
    return 100.0 * float(u.chain.vendor_iv[r])


_COLUMN_READERS: tuple[CellReader, ...] = (
    lambda u, r: str(u.chain.expiry[r]),
    lambda u, r: float(u.chain.strike[r]),
    lambda u, r: "Call" if u.chain.is_call[r] else "Put",
    lambda u, r: float(u.chain.bid[r]),
    lambda u, r: float(u.chain.ask[r]),
    lambda u, r: float(u.chain.mid[r]),
    lambda u, r: float(u.chain.open_interest[r]),
    _iv_pct,
    _vendor_pct,
    lambda u, r: _iv_pct(u, r) - _vendor_pct(u, r),
    lambda u, r: "yes" if u.usable[r] else "no",
)


@dataclass(frozen=True, slots=True, eq=False)
class ChainTable:
    """Column-wise chain data plus the formatting rules of each column."""

    update: SessionUpdate
    days: FloatArray

    @property
    def rows(self) -> int:
        return self.update.chain.size

    def expiries(self) -> list[date]:
        return [d.astype(object) for d in np.unique(self.update.chain.expiry)]

    def value(self, row: int, column: int) -> str | float:
        """Sortable value of a cell: numbers stay numbers, labels are strings."""
        return _COLUMN_READERS[column](self.update, row)


def chain_table(update: SessionUpdate) -> ChainTable:
    chain = update.chain
    return ChainTable(update=update, days=years_to_expiry(chain.asof, chain.expiry) * DAYS_PER_YEAR)


@dataclass(frozen=True, slots=True, eq=False)
class IvComparison:
    """Our implied volatility against the broker's model value, in volatility points."""

    ours_pct: FloatArray
    vendor_pct: FloatArray
    mean_abs_diff: float
    rmse: float
    correlation: float

    @property
    def n(self) -> int:
        return int(self.ours_pct.size)


def iv_comparison(update: SessionUpdate) -> IvComparison | None:
    """Pairs where both volatilities exist; ``None`` if there are fewer than three."""
    ours = 100.0 * update.implied_vol
    vendor = 100.0 * update.chain.vendor_iv
    both = np.isfinite(ours) & np.isfinite(vendor)
    if both.sum() < 3:
        return None
    a, b = ours[both], vendor[both]
    diff = a - b
    corr = float(np.corrcoef(a, b)[0, 1]) if np.ptp(a) > 0 and np.ptp(b) > 0 else float("nan")
    return IvComparison(
        ours_pct=a,
        vendor_pct=b,
        mean_abs_diff=float(np.mean(np.abs(diff))),
        rmse=float(np.sqrt(np.mean(diff**2))),
        correlation=corr,
    )


@dataclass(frozen=True, slots=True, eq=False)
class GexView:
    levels: FloatArray
    gex: FloatArray
    spot: float
    flip: float
    gex_at_spot: float
    size: float
    distance_to_flip_pct: float
    n_options: int


def gex_view(update: SessionUpdate) -> GexView | None:
    profile = update.gex
    if profile is None:
        return None
    return GexView(
        levels=profile.levels,
        gex=profile.gex,
        spot=profile.spot,
        flip=profile.flip,
        gex_at_spot=profile.gex_at_spot,
        size=profile.size,
        distance_to_flip_pct=profile.distance_to_flip_pct,
        n_options=profile.n_options,
    )


def format_diagnostics(snapshot: DiagnosticsSnapshot) -> list[tuple[str, str]]:
    """Label/value rows for the diagnostics panel."""
    rows = [
        ("State", snapshot.state.value),
        ("Updates", f"{snapshot.updates} at {snapshot.update_rate_hz:.1f} Hz"),
        ("Options per update", f"{snapshot.options_per_update:,}"),
    ]
    rows += [(f"  {name}", f"{1000 * s:.1f} ms") for name, s in snapshot.stage_seconds.items()]
    rows.append(("Pipeline total", f"{1000 * snapshot.total_seconds:.1f} ms"))
    rows += [
        (f"  backend {name}", f"{1000 * s:.1f} ms") for name, s in snapshot.backend_seconds.items()
    ]
    dropped = sum(s.dropped for s in snapshot.subscriptions)
    rows.append(("Compute failures", str(snapshot.compute_failures)))
    rows.append(("Handler failures", str(snapshot.handler_failures)))
    rows.append(("Events dropped (conflated)", str(dropped)))
    if snapshot.last_error:
        rows.append(("Last error", snapshot.last_error))
    return rows
