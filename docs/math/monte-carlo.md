# Monte Carlo pricing

Code: `pricing/montecarlo/` (`kernels.py` Numba, `models.py` path models, `engine.py` statistics).
Lab panel: *Monte Carlo* tab. Background: risk-neutral valuation, $V_0=e^{-rT}\mathbb E^{\mathbb Q}[\text{payoff}]$,
estimated by averaging discounted payoffs of simulated paths (law of large numbers).

## Models

| Model | Dynamics | Sampling |
|---|---|---|
| `GbmModel` | $dS=(r-q)S\,dt+\sigma S\,dW$ | exact terminal draw for terminal payoffs; stepped for the Asian payoff |
| `HestonModel` | $dS=(r-q)S\,dt+\sqrt v\,S\,dW^1$, $dv=\kappa(\theta-v)dt+\xi\sqrt v\,dW^2$, $d\langle W^1,W^2\rangle=\rho\,dt$ | full-truncation Euler on $\ln S$ ($v^+=\max(v,0)$ in drift and diffusion) |
| `TwoHestonModel` | two such assets, price shocks correlated by $\rho_{assets}$ | same scheme per asset |

`HestonModel.feller_satisfied` reports $2\kappa\theta\ge\xi^2$; the lab shows a warning when it fails,
because full truncation is then doing real work. Payoffs: call, put, best-of call, worst-of call,
arithmetic-average (Asian) call.

## Random numbers

Draws are a hash of `(seed, path, draw index)` (SplitMix64, Box-Muller for normals), not a stateful
stream. A path therefore sees the same numbers whichever thread simulates it, so results are
**bit-identical for any thread count** (`test_results_do_not_depend_on_the_number_of_threads` runs
the same price with 8 threads and with 1). The generator's moments, pair correlation and
neighbouring-path correlation are tested.

## Statistics

`MonteCarloEngine.price` discounts the samples and reports the mean, the standard error
$s/\sqrt n$ and a convergence history (running mean and standard error at 64 checkpoints computed
from cumulative sums). With antithetic variates a sample is the average of a path and its mirror,
so samples remain independent and the standard error stays valid; for a call this cut the standard
error by more than 15% in the tests.

## Validation

* GBM call and put agree with Black-Scholes within 3.5 standard errors (400,000 paths).
* Heston with $\xi=0$ and $v_0=\theta$ (constant variance) reduces to Black-Scholes.
* Standard error shrinks as $1/\sqrt n$ (ratio 4.0 for a 16x increase in paths).
* Negative $\rho$ makes out-of-the-money puts dearer than $\rho=0$; best-of dominates a single leg
  and worst-of is dominated by it; higher asset correlation lowers the best-of price; identical
  perfectly correlated assets make best-of equal to the single call; an Asian call is cheaper than
  the European call.

## Performance

1,000,000 GBM paths: about 28 ms. 200,000 Heston paths x 100 steps: about 300 ms (see
[benchmarks](../benchmarks.md)).

## Provenance

The Heston scheme and the rainbow (best-of) example follow the standard mathematics presented in
the Quant Guild lecture notebook. That notebook has no licence file, so **no code was copied**; the
kernels, the counter-based generator, antithetic variates and the standard-error/convergence
machinery are new. See [REFERENCES](../../REFERENCES.md).
