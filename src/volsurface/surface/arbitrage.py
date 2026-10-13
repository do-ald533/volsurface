"""Static no-arbitrage diagnostics for a set of SVI slices.

Two kinds of static arbitrage can exist in a volatility surface (Gatheral and Jacquier,
arXiv:1204.0646, section 2):

* **Butterfly**: a slice's implied density is negative somewhere. Checked per slice through
  ``g(k) >= 0``, plus Roger Lee's wing bound as a guard against heavy-tail mispricing.
* **Calendar spread**: total variance decreases with maturity at some log-moneyness, i.e.
  ``dw/dt < 0``. For a finite set of slices this is checked pairwise between neighbours: if every
  neighbouring pair is ordered, the whole chain is.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from volsurface.surface.svi import (
    CALENDAR_TOLERANCE,
    SviParams,
    check_slice,
    is_lee_compliant,
    total_variance,
)


@dataclass(frozen=True, slots=True)
class SliceReport:
    """Butterfly diagnostics for one expiry."""

    t: float
    min_g: float
    butterfly_free: bool
    lee_ok: bool

    @property
    def ok(self) -> bool:
        return self.butterfly_free and self.lee_ok


@dataclass(frozen=True, slots=True)
class CalendarViolation:
    """A neighbouring pair of expiries where the longer one has less total variance somewhere."""

    t_short: float
    t_long: float
    max_gap: float
    k_at_max: float


@dataclass(frozen=True, slots=True)
class ArbitrageReport:
    slices: tuple[SliceReport, ...]
    calendar: tuple[CalendarViolation, ...]

    @property
    def butterfly_violations(self) -> int:
        return sum(1 for s in self.slices if not s.butterfly_free)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.slices) and not self.calendar


def check_calendar(
    slices: Sequence[tuple[float, SviParams]],
    k_min: float,
    k_max: float,
    *,
    tolerance: float = CALENDAR_TOLERANCE,
) -> tuple[CalendarViolation, ...]:
    """Pairwise calendar check over ``[k_min, k_max]`` for slices ordered by maturity."""
    ordered = sorted(slices, key=lambda item: item[0])
    grid = np.linspace(k_min, k_max, 201)
    found: list[CalendarViolation] = []
    for (t_short, p_short), (t_long, p_long) in pairwise(ordered):
        gap = total_variance(grid, p_short) - total_variance(grid, p_long)
        worst = int(np.argmax(gap))
        if gap[worst] > tolerance:
            found.append(CalendarViolation(t_short, t_long, float(gap[worst]), float(grid[worst])))
    return tuple(found)


def build_report(
    slices: Sequence[tuple[float, SviParams, float, float]],
) -> ArbitrageReport:
    """Report for ``(t, params, k_min, k_max)`` tuples, one per fitted expiry.

    Calendar consistency is checked where both neighbouring slices have quotes, that is over the
    overlap of their log-moneyness ranges; beyond it a slice is an extrapolation, not a market.
    """
    per_slice = []
    for t, params, k_lo, k_hi in slices:
        min_g, butterfly_ok = check_slice(params, k_lo, k_hi)
        per_slice.append(SliceReport(t, min_g, butterfly_ok, is_lee_compliant(params)))
    if not slices:
        return ArbitrageReport((), ())
    ordered = sorted(slices, key=lambda item: item[0])
    calendar: list[CalendarViolation] = []
    for (t0, p0, lo0, hi0), (t1, p1, lo1, hi1) in pairwise(ordered):
        lo, hi = max(lo0, lo1), min(hi0, hi1)
        if hi > lo:
            calendar.extend(check_calendar([(t0, p0), (t1, p1)], lo, hi))
    return ArbitrageReport(tuple(per_slice), tuple(calendar))
