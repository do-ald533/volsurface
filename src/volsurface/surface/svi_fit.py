"""Calibration of raw SVI slices to observed total variances.

The optimiser works in ``(s, b, rho, m, sigma)`` where ``s = a + b*sigma*sqrt(1-rho^2)`` is the
minimum total variance, so non-negativity of ``w`` becomes the box constraint ``s >= 0``.
Residuals and the analytic Jacobian are compiled with Numba (:mod:`volsurface.surface.svi_kernels`).

Fitting a slice is a small strategy (:class:`_Fitter`):

1. **warm**: polish the previous parameters briefly; accept if arbitrage-free and as good as before;
2. **cold**: otherwise screen four starting points, polish the best;
3. **repair**: if the result violates the butterfly condition, Lee's bound or calendar ordering,
   refit with penalty residuals and keep the result only if it is better and still fits;
4. **fallback**: if violations remain, fit the three-parameter SSVI slice, which satisfies the
   butterfly condition by construction (:mod:`volsurface.surface.ssvi_fit`).
"""

import math
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, least_squares

from volsurface.domain.errors import InvalidInputError, PricingError
from volsurface.domain.types import FloatArray
from volsurface.surface import svi_kernels
from volsurface.surface.ssvi_fit import fit_ssvi_slice
from volsurface.surface.svi import (
    CALENDAR_TOLERANCE,
    CHECK_PAD,
    LEE_SLOPE_BOUND,
    SliceFunction,
    SviParams,
    butterfly_density_factor,
    check_slice,
    is_lee_compliant,
    total_variance,
)

BUTTERFLY_MARGIN = 0.02
REPAIR_RMSE_FACTOR = 4.0
REPAIR_RMSE_FLOOR = 1e-6
FALLBACK_RMSE_FACTOR = 10.0
WARM_START_ACCEPT = 0.01
WARM_START_NOISE_FACTOR = 2.0
_SCREEN_NFEV = 40
_WARM_NFEV = 40
_REPAIR_NFEV = 200
_B_MAX = 2.0
_SIGMA_MIN = 1e-3
_SIGMA_MAX = 2.0
_RHO_MAX = 0.999
_MIN_POINTS = 5


@dataclass(frozen=True, slots=True)
class SviFit:
    """Result of fitting one slice."""

    params: SviParams
    rmse_variance: float
    rmse_vol: float
    n_points: int
    min_g: float
    butterfly_free: bool
    lee_ok: bool
    nfev: int

    @property
    def arbitrage_free(self) -> bool:
        return self.butterfly_free and self.lee_ok


@dataclass(frozen=True, slots=True)
class _Problem:
    """Data, bounds and residual definitions for one slice fit."""

    k: FloatArray
    w: FloatArray
    sqrt_weight: FloatArray
    lower: FloatArray
    upper: FloatArray
    penalty_grid: FloatArray
    penalty_scale: float
    lower_slice: SliceFunction | None

    def residuals(self, x: FloatArray) -> FloatArray:
        res, _ = svi_kernels.residuals_and_jacobian(x, self.k, self.w, self.sqrt_weight)
        return cast("FloatArray", res)

    def jacobian(self, x: FloatArray) -> FloatArray:
        _, jac = svi_kernels.residuals_and_jacobian(x, self.k, self.w, self.sqrt_weight)
        return cast("FloatArray", jac)

    def penalised(self, x: FloatArray) -> FloatArray:
        p = SviParams.from_minimum_variance(*x)
        g = butterfly_density_factor(self.penalty_grid, p)
        g_pen = np.minimum(np.where(np.isfinite(g), g, -1.0) - BUTTERFLY_MARGIN, 0.0)
        slopes = np.array(
            [
                max(p.left_slope - LEE_SLOPE_BOUND, 0.0),
                max(p.right_slope - LEE_SLOPE_BOUND, 0.0),
            ]
        )
        scale = self.penalty_scale
        parts = [self.residuals(x), scale * g_pen / math.sqrt(g_pen.size), scale * slopes]
        if self.lower_slice is not None:
            gap = np.maximum(
                self.lower_slice(self.penalty_grid) - total_variance(self.penalty_grid, p), 0.0
            )
            parts.append(scale * gap / math.sqrt(gap.size))
        return np.concatenate(parts)

    def clip(self, x: FloatArray) -> FloatArray:
        return np.clip(x, self.lower + 1e-9, self.upper - 1e-9)

    def solve(
        self, x0: FloatArray, *, penalised: bool = False, max_nfev: int = 200
    ) -> OptimizeResult:
        return least_squares(
            self.penalised if penalised else self.residuals,
            self.clip(x0),
            jac="2-point" if penalised else self.jacobian,
            bounds=(self.lower, self.upper),
            method="trf",
            loss="soft_l1",
            f_scale=max(float(self.w.std()), 1e-4),
            x_scale="jac",
            max_nfev=max_nfev,
        )

    def starting_points(self) -> list[FloatArray]:
        k, w = self.k, self.w
        k_span = max(float(k.max() - k.min()), 1e-3)
        s0 = max(float(w.min()), 1e-6)
        b0 = float(np.clip((w.max() - w.min()) / k_span, 0.02, 1.0))
        m0 = float(k[int(np.argmin(w))])
        return [
            np.array([s0, b0, -0.4, m0, 0.10]),
            np.array([s0, b0, 0.0, 0.0, 0.30]),
            np.array([s0, b0 * 0.5, -0.8, m0, 0.05]),
            np.array([s0, b0, -0.2, m0, 0.50]),
        ]

    def rmse(self, p: SviParams) -> float:
        fitted = cast(
            "FloatArray",
            svi_kernels.total_variance_grid(self.k, p.a, p.b, p.rho, p.m, p.sigma),
        )
        weight = self.sqrt_weight**2
        return float(np.sqrt(np.average((fitted - self.w) ** 2, weights=weight)))


def _build_problem(
    k: ArrayLike,
    w: ArrayLike,
    weights: ArrayLike | None,
    lower_slice: SliceFunction | None,
    penalty_scale: float,
) -> _Problem:
    kk = np.asarray(k, dtype=np.float64).reshape(-1)
    ww = np.asarray(w, dtype=np.float64).reshape(-1)
    if kk.shape != ww.shape:
        raise InvalidInputError("k and w must have the same length")
    wt = np.ones_like(kk) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if wt.shape != kk.shape or (wt < 0).any():
        raise InvalidInputError("weights must be non-negative and match k")
    keep = np.isfinite(kk) & np.isfinite(ww) & np.isfinite(wt) & (ww > 0) & (wt > 0)
    kk, ww, wt = kk[keep], ww[keep], wt[keep]
    if kk.size < _MIN_POINTS:
        raise InvalidInputError(f"need at least {_MIN_POINTS} usable points, got {kk.size}")
    wt = wt / wt.mean()
    lower = np.array([0.0, 1e-6, -_RHO_MAX, float(kk.min()) - 1.0, _SIGMA_MIN])
    upper = np.array(
        [max(2.0 * float(ww.max()), 1e-4), _B_MAX, _RHO_MAX, float(kk.max()) + 1.0, _SIGMA_MAX]
    )
    return _Problem(
        k=np.ascontiguousarray(kk),
        w=np.ascontiguousarray(ww),
        sqrt_weight=np.ascontiguousarray(np.sqrt(wt)),
        lower=lower,
        upper=upper,
        penalty_grid=np.linspace(kk.min() - CHECK_PAD, kk.max() + CHECK_PAD, 121),
        penalty_scale=penalty_scale,
        lower_slice=lower_slice,
    )


@dataclass(frozen=True, slots=True)
class _Status:
    """Arbitrage standing of a candidate slice."""

    min_g: float
    butterfly_ok: bool
    lee_ok: bool
    calendar_gap: float

    @property
    def violations(self) -> int:
        return (
            (not self.butterfly_ok) + (not self.lee_ok) + (self.calendar_gap > CALENDAR_TOLERANCE)
        )

    def better_than(self, other: "_Status") -> bool:
        if self.violations != other.violations:
            return self.violations < other.violations
        return self.min_g > other.min_g and self.calendar_gap <= other.calendar_gap


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A fitted parameter vector with its quality and the effort it took."""

    x: FloatArray
    params: SviParams
    status: _Status
    rmse: float
    nfev: int


class _Fitter:
    """Strategy for fitting one slice: warm start when possible, multi-start otherwise."""

    def __init__(self, problem: _Problem, *, enforce: bool) -> None:
        self._problem = problem
        self._enforce = enforce
        self._k_lo = float(problem.k.min())
        self._k_hi = float(problem.k.max())

    def fit(self, initial: SviParams | None, initial_rmse: float | None) -> _Candidate:
        if initial is not None:
            warm = self._warm(initial, initial_rmse)
            if warm is not None:
                return warm
        return self._cold()

    def _assess(self, x: FloatArray, nfev: int) -> _Candidate:
        params = SviParams.from_minimum_variance(*x)
        return _Candidate(x, params, self._status(params), self._problem.rmse(params), nfev)

    def _status(self, params: SviParams) -> _Status:
        min_g, butterfly_ok = check_slice(params, self._k_lo, self._k_hi)
        gap = 0.0
        floor = self._problem.lower_slice
        if floor is not None:
            grid = np.linspace(self._k_lo, self._k_hi, 81)
            gap = float(np.max(np.maximum(floor(grid) - total_variance(grid, params), 0.0)))
        return _Status(min_g, butterfly_ok, is_lee_compliant(params), gap)

    def _repair(self, candidate: _Candidate, max_nfev: int) -> _Candidate:
        if not self._enforce or not candidate.status.violations:
            return candidate
        try:
            refit = self._problem.solve(candidate.x, penalised=True, max_nfev=max_nfev)
        except (ValueError, FloatingPointError):
            return candidate
        repaired = self._assess(cast("FloatArray", refit.x), candidate.nfev + int(refit.nfev))
        affordable = repaired.rmse <= max(REPAIR_RMSE_FACTOR * candidate.rmse, REPAIR_RMSE_FLOOR)
        if affordable and repaired.status.better_than(candidate.status):
            return repaired
        return replace(candidate, nfev=repaired.nfev)

    def _fallback(self, candidate: _Candidate) -> _Candidate:
        if not self._enforce or not candidate.status.violations:
            return candidate
        problem = self._problem
        try:
            fit = fit_ssvi_slice(
                problem.k,
                problem.w,
                problem.sqrt_weight,
                lower_slice=problem.lower_slice,
                penalty_scale=problem.penalty_scale,
            )
        except PricingError:
            return candidate
        alternative = self._assess(fit.params.as_fit_vector(), candidate.nfev + fit.nfev)
        acceptable = alternative.rmse <= max(
            FALLBACK_RMSE_FACTOR * candidate.rmse, REPAIR_RMSE_FLOOR
        )
        if acceptable and alternative.status.better_than(candidate.status):
            return alternative
        return replace(candidate, nfev=alternative.nfev)

    def _warm(self, initial: SviParams, initial_rmse: float | None) -> _Candidate | None:
        problem = self._problem
        result = _try_solve(problem, initial.as_fit_vector(), max_nfev=_WARM_NFEV)
        if result is None:
            return None
        candidate = self._repair(
            self._assess(cast("FloatArray", result.x), int(result.nfev)), _WARM_NFEV * 2
        )
        tolerance = WARM_START_ACCEPT * float(problem.w.mean())
        if initial_rmse is not None:
            tolerance = max(tolerance, WARM_START_NOISE_FACTOR * initial_rmse)
        clean = candidate.status.violations == 0 or not self._enforce
        return candidate if clean and candidate.rmse <= tolerance else None

    def _cold(self) -> _Candidate:
        problem = self._problem
        screened = [
            r
            for x0 in problem.starting_points()
            if (r := _try_solve(problem, x0, max_nfev=_SCREEN_NFEV)) is not None
        ]
        if not screened:
            raise PricingError("SVI fit failed from every starting point")
        nfev = sum(int(r.nfev) for r in screened)
        leader = min(screened, key=lambda r: r.cost)
        polished = _try_solve(problem, cast("FloatArray", leader.x), max_nfev=_REPAIR_NFEV)
        best = polished if polished is not None else leader
        nfev += int(polished.nfev) if polished is not None else 0
        assessed = self._assess(cast("FloatArray", best.x), nfev)
        return self._fallback(self._repair(assessed, _REPAIR_NFEV))


def _try_solve(problem: _Problem, x0: FloatArray, *, max_nfev: int) -> OptimizeResult | None:
    try:
        return problem.solve(x0, max_nfev=max_nfev)
    except (ValueError, FloatingPointError):
        return None


def fit_svi_slice(
    k: ArrayLike,
    w: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    t: float | None = None,
    initial: SviParams | None = None,
    initial_rmse: float | None = None,
    enforce_no_arbitrage: bool = True,
    lower_slice: SliceFunction | None = None,
    penalty_scale: float = 50.0,
) -> SviFit:
    """Fit raw SVI to observed total variances.

    Args:
        k: log-moneyness of the observations.
        w: observed total variance ``iv^2 * T`` at those points.
        weights: non-negative weights on the residuals (default: all ones).
        t: expiry in years; only used to report the volatility-space RMSE.
        initial: warm start (for example last update's parameters). A short polish from there is
            accepted when it is arbitrage-free and fits to within one percent of the mean
            variance, or within twice ``initial_rmse`` (the previous fit's residual, which
            captures the noise level); otherwise the multi-start search runs.
        initial_rmse: variance RMSE of the fit that produced ``initial``.
        enforce_no_arbitrage: add butterfly and wing-slope penalties and refit if the plain fit
            violates them.
        lower_slice: ``k -> w`` of the previous (shorter) expiry; adds a calendar penalty so the
            fitted slice does not cross below it.
        penalty_scale: weight of the arbitrage penalties relative to data residuals.

    Raises:
        InvalidInputError: fewer than five usable points, or inconsistent inputs.
        PricingError: the optimiser failed from every start.
    """
    problem = _build_problem(k, w, weights, lower_slice, penalty_scale)
    best = _Fitter(problem, enforce=enforce_no_arbitrage).fit(initial, initial_rmse)
    rmse_vol = math.nan
    if t is not None and t > 0:
        fitted = total_variance(problem.k, best.params)
        diff = np.sqrt(np.maximum(fitted, 0.0) / t) - np.sqrt(problem.w / t)
        rmse_vol = float(np.sqrt(np.average(diff**2, weights=problem.sqrt_weight**2)))
    return SviFit(
        params=best.params,
        rmse_variance=best.rmse,
        rmse_vol=rmse_vol,
        n_points=int(problem.k.size),
        min_g=best.status.min_g,
        butterfly_free=best.status.butterfly_ok,
        lee_ok=best.status.lee_ok,
        nfev=best.nfev,
    )
