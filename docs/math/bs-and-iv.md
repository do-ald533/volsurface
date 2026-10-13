# Black-Scholes and implied volatility

Code: `pricing/black_scholes.py`, `pricing/implied_vol.py`, `pricing/bs_kernels.py`.

## Model

European option, spot $S$, strike $K$, maturity $T$ (years), continuously compounded rate $r$,
continuous dividend yield $q$, volatility $\sigma$:

$$
d_1=\frac{\ln(S/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt T},\qquad d_2=d_1-\sigma\sqrt T,
$$

$$
C = S e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2),\qquad P = Ke^{-rT}\Phi(-d_2)-Se^{-qT}\Phi(-d_1).
$$

The code writes both with a sign $s=\pm1$ (call/put): $V = s\,[Se^{-qT}\Phi(s d_1)-Ke^{-rT}\Phi(s d_2)]$.

### Conventions

| Greek | Definition used here |
|---|---|
| delta | $\partial V/\partial S$ |
| gamma | $\partial^2 V/\partial S^2$ |
| vega | $\partial V/\partial\sigma$, **per 1.00 of volatility** (not per vol point) |
| theta | $\partial V/\partial t=-\partial V/\partial T$, **per year** |
| rho | $\partial V/\partial r$ |

If $\sigma\sqrt T<10^{-12}$ the value is the discounted forward intrinsic value
$\max(s(Se^{-qT}-Ke^{-rT}),0)$ and the Greeks are `nan` (their limits are distributions, not numbers).

The original `black_scholes.py` seed (call/put with a string flag, no dividend, errors swallowed by
`try/except`) became this module: same closed form, with `q`, Greeks, vectorisation, explicit input
validation (`InvalidInputError`) and defined limits.

## Implied volatility

Given a price, find $\sigma$ with $V(\sigma)=\text{price}$. $V$ is strictly increasing in $\sigma$
wherever vega is positive, so a unique root exists exactly when the price lies strictly inside the
no-arbitrage bounds

$$
\max\bigl(s(Se^{-qT}-Ke^{-rT}),0\bigr) < \text{price} < \begin{cases}Se^{-qT}&\text{call}\\ Ke^{-rT}&\text{put}\end{cases}
$$

and the root lies in $[10^{-6},10]$. Outside that, the solver returns `nan` instead of a number
that would not reprice.

### Algorithms

1. **`implied_vol_bs`** (used by the pipeline): safeguarded Newton-Raphson. Every iterate keeps a
   bracket $[lo,hi]$ around the root; the Newton step $\sigma-(V-\text{price})/\text{vega}$ is taken
   only if it falls inside the bracket and vega is not tiny, otherwise the interval is bisected.
   It therefore converges for every attainable price. It is compiled with Numba, one option per
   loop iteration in parallel, and each option stops as soon as it converges.
2. **`implied_vol_bs_reference`**: the same algorithm with whole-array NumPy operations. It is the
   readable specification the compiled solver is tested against (agreement on the whole 5,000-option
   random chain within `1e-4` relative, and both reprice to `2e-10`).
3. **`newton_iterations`**: the textbook algorithm, one option, every iterate recorded. It raises
   `ConvergenceError` where plain Newton fails (vanishing vega), which is the reason the safeguarded
   version exists. The *Newton-Raphson IV* tab of the Pricing Lab animates it.

The tutorial notebook that inspired the Newton explorer scaled vega by 100 because its library
reports vega per vol point; here vega is per 1.00, so no scaling is needed.

### Validation

* put-call parity and "price is within the no-arbitrage bounds" as Hypothesis properties over
  $S\in[1,1000]$, moneyness $0.5$ to $2$, $T\in[0.01,5]$, $r\in[-2\%,15\%]$, $q\in[0,8\%]$,
  $\sigma\in[3\%,250\%]$;
* re-pricing at the recovered volatility reproduces the input price whenever it is identifiable;
* price and all five Greeks agree with QuantLib's analytic European engine (`oracle` tests).

### Performance

10,000 options: **0.57 ms** compiled versus **13.5 ms** for the NumPy specification (23x), see
[benchmarks](../benchmarks.md).
