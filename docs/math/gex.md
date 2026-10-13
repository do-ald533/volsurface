# Gamma exposure (GEX), the flip level and alerts

Code: `analytics/gex.py`, `analytics/alerts.py`, `pricing/quote_filter.py` (`gex_quote_spec`).

## Definition

For each listed option $i$ with open interest $OI_i$, contract multiplier $M$ and right sign
$s_i=+1$ (call) or $-1$ (put), the contribution at underlying price $x$ is

$$
\text{GEX}_i(x)=s_i\,OI_i\,M\cdot 0.01\cdot10^{-9}\cdot\Gamma_i(x)\,x^2,
$$

in **billions of dollars per 1% move of the underlying**; $\Gamma_i$ is the American tree gamma at
$x$ (see [crr](crr.md)) with the option's own implied volatility. The profile
$\text{GEX}(x)=\sum_i\text{GEX}_i(x)$ is evaluated on 81 points from 90% to 110% of spot.

**Convention, not fact.** Counting calls positive and puts negative is the convention of the
notebook this was ported from; it corresponds to assuming dealers are long calls and short puts.
Providers differ, and open interest says nothing about who holds the positions. The output is
an indicator under that assumption.

## Flip level

The flip is the price on the grid nearest to spot where $\text{GEX}$ changes sign, found by linear
interpolation between the two grid points around the crossing (`nan` if there is none). Positive
GEX at spot is read as "moves tend to be damped", negative as "moves tend to be amplified", with the
same caveat.

## Quote selection

Only options with a two-sided quote, time value above $0.05$, expiring within 120 days and with
positive known open interest contribute; the IV surface uses the same rules without the open
interest requirement. All of these thresholds are `PipelineSettings` fields.

## Alerts (rules on consecutive updates)

| Rule | Fires when |
|---|---|
| `GexSignChange` | the sign of GEX at spot differs from the previous update |
| `NearFlip` | spot enters the band of 0.5% around the flip (also at first sight if already inside) |
| `IvJump` | at-the-money volatility (nearest strike, nearest expiry at least a day out) moves by more than 2 vol points |

`AlertEngine` skips underlyings whose GEX never exceeds 0.05 bn$ per 1% anywhere on the grid, because
tiny gamma makes sign changes meaningless.

## Where open interest comes from

* **Synthetic and replay**: carried in the data. The synthetic market shapes open interest (calls
  peaking above, puts below) around the *initial* spot and keeps it at those strikes while spot
  moves, so a scripted spot path produces genuine sign changes (`data/samples/flip`).
* **Interactive Brokers**: generic tick `101` fills `callOpenInterest` and `putOpenInterest`
  (ib_async documentation); open interest is published once a day by the exchanges, so it is
  constant within a session.

## Performance

The profile needs the gamma of every option at 81 spot levels: 1,500 options x 81 levels x a
tree. On this CPU (8 threads) that is about 0.69 s with 200 steps; the pipeline uses 100 steps
for GEX by default. The upstream notebook's README reports 85,814 options per snapshot at 0.69 s with
Numba CUDA (the author's figure, not reproduced here); this application scans one underlying at a
time, which the CPU backend handles.
