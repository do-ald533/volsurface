# ADR 0007: Implied-volatility solver choices

Status: accepted (2026-10-13)

## Context
Black-Scholes IV for thousands of quotes per update and American IV on a tree are the hot paths.

## Decision
* Black-Scholes: safeguarded Newton (bracket plus bisection fallback), compiled with Numba and
  parallel over options, with the NumPy version kept as the executable specification.
* Tree: Brent's method on the notebook's bracket and root. Illinois false position was tried first
  and rejected after measurement (slower than bisection on flat prices).

## Consequences
* 23x (Black-Scholes) and 3.9x (tree) faster than the straightforward versions, with tests that the
  fast and reference solvers agree.
