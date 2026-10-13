# SVI slices, arbitrage checks and the SSVI fallback

Code: `surface/svi.py` (model and checks), `surface/svi_fit.py` (calibration),
`surface/ssvi.py` and `surface/ssvi_fit.py`, `surface/arbitrage.py`, `surface/builder.py`.
Reference: Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*,
[arXiv:1204.0646](https://arxiv.org/abs/1204.0646). The formulas below were read from the paper,
not from memory.

## Raw SVI

For one expiry, total implied variance $w(k)=\sigma_{imp}(k)^2T$ as a function of log-moneyness
$k=\ln(K/F)$, with forward $F=Se^{(r-q)T}$:

$$
w(k)=a+b\Bigl(\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\Bigr),\qquad b\ge0,\ |\rho|<1,\ \sigma>0,\ a+b\sigma\sqrt{1-\rho^2}\ge0 .
$$

The optimiser works in $(s,b,\rho,m,\sigma)$ with $s=a+b\sigma\sqrt{1-\rho^2}$ (the minimum
variance), so positivity is a box constraint. Residuals and the analytic Jacobian are Numba kernels
(checked against finite differences in the tests).

## No-arbitrage conditions

* **Butterfly** (Lemma 2.2): a slice is free of butterfly arbitrage iff $g(k)\ge0$ for all $k$ (and
  $d_+(k)\to-\infty$), where

  $$
  g(k)=\Bigl(1-\frac{k\,w'}{2w}\Bigr)^2-\frac{w'^2}{4}\Bigl(\frac1w+\frac14\Bigr)+\frac{w''}2 .
  $$

* **Calendar spread** (Lemma 2.1): $\partial w(k,t)/\partial t\ge0$ for every $k$.
* **Wing slopes**: increasing $b$ steepens both wings; the asymptotic slopes of $w$ are
  $b(1\pm\rho)$ (derived here from the formula), and the paper relates the bound $\theta\varphi(1+|\rho|)\le4$
  to Roger Lee's upper bound of 2 on the asymptotic slope. The code requires $b(1+|\rho|)\le2$.

The paper gives **no explicit parameter conditions for raw SVI** (it calls finding them
"seemingly impossible" and shows a counter-example, Example 3.1, `a=-0.0410, b=0.1331,
m=0.3586, rho=0.3060, sigma=0.4153`, which has negative density; it is a test fixture here). Its
explicit conditions are for SSVI. So the implementation checks $g\ge0$ numerically on a grid
padded by 0.5 beyond the data and uses Lee's bound for the tails.

## SSVI

$$
w(k,\theta_t)=\frac{\theta_t}{2}\Bigl\{1+\rho\,\varphi(\theta_t)k+\sqrt{(\varphi(\theta_t)k+\rho)^2+1-\rho^2}\Bigr\},\qquad\theta_t=\sigma^2_{ATM}(t)\,t .
$$

Sufficient conditions (Theorems 4.1 and 4.2): $\theta_t$ non-decreasing; $0\le\partial_\theta(\theta\varphi)\le\frac{1}{\rho^2}(1+\sqrt{1-\rho^2})\varphi$;
$\theta\varphi(1+|\rho|)<4$; $\theta\varphi^2(1+|\rho|)\le4$. `SsviSurface.conditions()` checks them
numerically; the synthetic market's ground-truth surface passes.

For a single maturity SSVI **is** a raw SVI slice, with

$$
a=\tfrac{\theta(1-\rho^2)}{2},\quad b=\tfrac{\theta\varphi}{2},\quad m=-\tfrac{\rho}{\varphi},\quad\sigma=\tfrac{\sqrt{1-\rho^2}}{\varphi},
$$

(a test confirms this mapping to `1e-9`, and a Hypothesis test confirms that every $(\theta,\rho,u)$
with $\varphi=u\varphi_{max}(\theta,\rho)$ gives $g\ge-10^{-9}$ on $[-3,3]$).

## How a slice is fitted

1. **Weights**: $\text{vega}/(\sigma T)$ normalised to a maximum of 1 and floored at 0.05. A price
   error of $\delta P$ moves total variance by $2\sigma T\,\delta P/\text{vega}$, so this
   down-weights quotes whose price carries little volatility information (tick-size noise on cheap
   wings).
2. **Warm path**: polish the previous parameters (at most 40 evaluations); accept if the result is
   arbitrage-free and fits within one percent of the mean variance or within twice the previous
   residual.
3. **Cold path**: otherwise screen four starting points (40 evaluations each), polish the best.
4. **Repair**: if butterfly, Lee or calendar ordering is violated, refit with penalty residuals
   (margin 0.02 on $g$) and keep the result only if it is better and does not destroy the fit.
5. **Fallback**: if violations remain, fit the three-parameter SSVI slice, which is butterfly-free
   by construction.

Calendar consistency is enforced by fitting expiries in increasing maturity with a penalty on
$w(k)<w_{prev}(k)$, and between slices total variance is interpolated linearly in maturity at fixed
$k$ (which preserves ordering: if two slices are ordered, every blend is).

## What was measured

On a small noisy synthetic chain (3 expiries, strikes every 2, 16 consecutive updates; truth is the
SSVI surface that generated the quotes):

| | arbitrage-free updates | mean max error vs truth |
|---|---|---|
| raw SVI with penalties only | 7 of 16 (44%) | 0.23 vol points |
| plus SSVI fallback | **16 of 16** | **0.26 vol points** |

The failures were one-sided data: when only one wing has liquid quotes, raw SVI can place its
minimum just past the data with a steep slope, which has negative density. Cost: cold start
about 7 ms per slice, warm start 0.26 ms for a clean refit (see [benchmarks](../benchmarks.md)); in the
running application the *surface* stage of a five-expiry update measured about 170 ms on the first
update and about 20 ms afterwards (`volsurface demo`).

## Limits

* A fitted surface is checked for **static** arbitrage on the quoted range only; beyond it a slice is
  an extrapolation. Calendar consistency is checked on the overlap of neighbouring slices' data
  ranges.
* Vertical interpolation is linear in total variance (no smoothing across strikes beyond SVI
  itself); outside the fitted maturities total variance is scaled in proportion to maturity.
* Fits reflect the quotes you give them; thin or stale quotes give a poor surface, and the
  diagnostics (RMSE, point count, `g_min`) are shown next to the smile for that reason.
