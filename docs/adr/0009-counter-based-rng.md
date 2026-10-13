# ADR 0009: Counter-based random numbers for Monte Carlo

Status: accepted (2026-10-13)

## Context
`prange` with a stateful generator makes results depend on how paths are split across threads.

## Decision
Each draw is a hash of `(seed, path, draw index)` (SplitMix64) with Box-Muller normals.

## Consequences
* Results are bit-identical for any thread count (tested with 1 and 8 threads), so Monte Carlo tests
  can assert exact equality and a recording of a run can be reproduced.
* SplitMix64 is a well-known statistical-quality hash, not a cryptographic generator; moments and
  correlations are tested but this is not a replacement for a full test battery.
