# ADR 0001: Hexagonal architecture with enforced layering

Status: accepted (2026-10-13)

## Context
The project mixes fast-changing edges (a broker API, a GUI toolkit) with numerics that should stay
simple, portable and testable. The starting point was one script that mixed IBKR callbacks,
matplotlib state and pricing.

## Decision
Keep a pure core (`domain`, `pricing`, `surface`, `analytics`) and put frameworks in adapters
(`marketdata`, `ui`). Ports are structural `Protocol`s in `domain/ports.py`. The rule is executable:
`import-linter` contracts fail CI when the core imports Qt, ib_async or an outer layer, or when the
UI imports market-data adapters directly.

## Consequences
* The core runs, and is tested, without a display, a network or a broker.
* Adding a source or a backend does not touch existing code.
* More modules and indirection than a script; mitigated by [architecture.md](../architecture.md).
* One pragmatic exception: `app.sources` re-exports `SourceSpec` so the UI does not import the
  adapters package.
