# ADR 0005: Numba CPU first, GPU as an optional plugin

Status: accepted (2026-10-13)

## Context
The upstream GEX notebook uses Numba CUDA in float32. `numba-cuda` is in maintenance mode per its
PyPI page, and the development machine has no working NVIDIA driver.

## Decision
`PricingBackend` is a strategy interface; the default `numba-cpu` backend is float64, multi-threaded
and tested in CI. Other backends register through the `volsurface.pricing_backends` entry-point
group, so a CUDA plugin needs no change to this code base.

## Consequences
* One underlying (about 1,500 options) is comfortably real-time on a laptop CPU
  ([benchmarks](../benchmarks.md)).
* Single-precision error is quantified ([crr.md](../math/crr.md)) so a GPU backend can be validated.
