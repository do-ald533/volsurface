# Benchmarks

Measured with `pytest-benchmark` (`uv run pytest -m perf tests/perf --benchmark-only`) and
`uv run volsurface bench`. Nothing here is estimated; rerun them on your hardware.

**Machine**: Intel Core i7-11370H (4 cores / 8 threads, 3.3 GHz), Linux, CPython 3.13.13,
NumPy 2.4.6, SciPy 1.18.1, Numba 0.68.0 (8 threads, default threading layer), 2026-10-13.
Numba's on-disk cache was warm; the first run after an edit also pays compilation.

| Operation | Time (mean) | Notes |
|---|---|---|
| Black-Scholes price, 10,000 options (NumPy) | 0.41 ms | `bs_price`, vectorised |
| Black-Scholes implied vol, 10,000 options, **Numba** | **0.8 ms** (min 0.57) | `implied_vol_bs` |
| Black-Scholes implied vol, 10,000 options, NumPy specification | 14.2 ms (min 13.5) | `implied_vol_bs_reference`; 23x slower at the minimum |
| CRR price + gamma, 1,500 options, 200 steps | 9.1 ms | `american_price_gamma` |
| CRR implied vol, 1,500 options, **Brent** | 92.9 ms | `american_implied_vol` |
| CRR implied vol, 1,500 options, bisection (the notebook's method) | 357 ms | `method="bisection"`; 3.8x slower |
| CRR gamma on an 81-level grid, 1,500 options, 200 steps | 689 ms | GEX profile; the pipeline uses 100 steps |
| SVI slice fit (40 points), cold start | 7.2 ms | four screened starts + polish |
| SVI slice fit (40 points), warm start | 0.27 ms | previous parameters accepted after a short polish; 27x faster |
| Monte Carlo GBM call, 1,000,000 paths | 31 ms | antithetic, counter-based RNG |
| Monte Carlo Heston call, 200,000 paths x 100 steps | 301 ms | full-truncation Euler |

## End-to-end (the running application)

`uv run volsurface demo --ticks 2` on the synthetic chain (528 options, five fitted expiries):

| Stage | first update | second update |
|---|---|---|
| filter | 0.2 ms | 0.1 ms |
| implied vol (CRR, 200 steps) | 30 ms | 27 ms |
| surface (five SVI fits) | 172 ms | 21 ms |
| GEX (100-step tree, 81 levels) | 34 ms | 33 ms |
| total | 237 ms | 81 ms |

The pipeline therefore sustains well over 5 updates per second on one underlying; the window
renders at a fixed 5 Hz by default (`ui_hz`).

## Reading the numbers

* Compiling is the one-off cost the GUI hides: it shows the window first, then warms the kernels.
* Parallel batches are serialised by a lock (Numba's default threading layer cannot run two
  parallel regions at once), so these figures are for one batch at a time.
* The upstream GEX notebook's GPU figures (85,814 options per snapshot) are not reproduced here
  because this machine has no working NVIDIA driver.
