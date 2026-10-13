# ADR 0004: SVI slices with a provably safe SSVI fallback

Status: accepted (2026-10-13)

## Context
Quotes are noisy and often cover one wing only. A raw 5-parameter SVI fit can reproduce the data and
still have negative implied density, and the paper gives no explicit parameter conditions for raw
SVI.

## Decision
Fit raw SVI with vega-based weights; check butterfly (`g(k) >= 0`), Lee's wing bound and calendar
ordering numerically; repair with penalty residuals; if violations remain, fit the SSVI
three-parameter slice whose parameters are constrained by the theorem's sufficient conditions.

## Consequences
* Measured on a noisy synthetic chain: arbitrage-free updates rose from 7/16 to 16/16 at a cost of
  about 0.04 vol points of accuracy.
* A fallback slice is less flexible than raw SVI; the diagnostics show RMSE per expiry.
* Global SSVI calibration across maturities remains future work.
