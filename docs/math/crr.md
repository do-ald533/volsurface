# American options on a CRR tree

Code: `pricing/crr_kernels.py` (Numba), `pricing/american.py` (validated front end),
`pricing/crr_reference.py` (plain NumPy, dtype selectable).

## The tree

With $n$ steps, $\Delta t=T/n$ and $\Delta x=\sigma\sqrt{\Delta t}$, node $j$ at step $i$ is
$S\,e^{(2j-i)\Delta x}$. The up-probability comes from the log-drift,

$$
p_u=\tfrac12+\tfrac12\,\frac{(r-q-\tfrac12\sigma^2)\Delta t}{\Delta x},\qquad p_d=1-p_u,
$$

values roll back as $V=\max\bigl(e^{-r\Delta t}(p_uV_{up}+p_dV_{down}),\ \text{intrinsic}\bigr)$
(the `max` is dropped for a European option). If $p_u\notin[0,1]$ the tree is invalid and the
result is `nan`.

**Gamma** is a finite difference of the three node values at step 2:

$$
\Gamma\approx\frac{\dfrac{V_2-V_1}{S_{uu}-S}-\dfrac{V_1-V_0}{S-S_{dd}}}{\tfrac12(S_{uu}-S_{dd})}.
$$

This is the construction used by the GEX notebook this project re-implements (MIT licensed,
see [REFERENCES](../../REFERENCES.md)). The notebook states that it matches QuantLib's
`BinomialVanillaEngine("crr")`; that was **checked, not assumed**: in `tests/unit/pricing/test_american.py`
price agrees to relative `1e-9` and gamma to `1e-6` for American and European puts and calls with
dividends (in practice they agree to about ten digits).

## Implied volatility on the tree

Two solvers, selected by `method=`:

* `"bisection"`: the notebook's solver, 40 halvings of $[1\%,500\%]$.
* `"brent"` (default): Brent's method on the same bracket and the same root, with an absolute
  volatility tolerance of `1e-10`. The first version used Illinois false position; it was slower
  than bisection on deep out-of-the-money options, where the price is flat in $\sigma$, so it was
  replaced.

A price at or below the tree value at 1% (or at/above the value at 500%) has no unique
volatility and gives `nan` for both solvers.

For 1,500 options: bisection 357 ms, Brent 93 ms (3.9x), agreement better than `1e-7` in volatility.

## Single precision

The original notebook runs in `float32` on a GPU. `crr_reference(..., dtype=np.float32)` lets
the loss be measured (200 steps, 60 random options, spot 100):

| | median | 95th percentile | maximum |
|---|---|---|---|
| price, relative error | 6e-6 | 1.5e-5 | 1.7e-5 |
| price, absolute error | 3.9e-5 | 1.8e-4 | 2.1e-4 |
| gamma, relative error | 1.8e-5 | 1.3e-4 | 9.3e-3 |

Prices survive single precision comfortably (a few hundredths of a cent); gamma has a heavy tail
because it is a difference of nearly equal quantities. That is why the CPU backend is float64 and
why a future GPU backend should be validated against it.

## Early exercise

An American call on a non-dividend-paying stock is never exercised early, so its tree value equals
the European value (asserted to `1e-12`); an American put is worth at least the European put and at
least intrinsic value, and strictly more for some strikes (also asserted).
