# ADR 0006: Synthetic and replay modes are first-class

Status: accepted (2026-10-13)

## Context
A portfolio project that needs a brokerage account and paid data to show anything cannot be
reviewed. Real quotes also cannot be published (vendor redistribution terms).

## Decision
The synthetic market (an SSVI surface priced with the CRR tree) and recorded replays are full
providers behind the same port as IBKR. `EventRecorder` and `ReplayProvider` make the event log the
recording. Samples in `data/samples` are synthetic; `data/recordings/` is git-ignored.

## Consequences
* `uv run volsurface gui` works offline in seconds; replay is deterministic and golden-tested.
* The synthetic generator carries ground truth, so fit quality is measured, not eyeballed.
