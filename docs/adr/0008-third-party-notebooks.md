# ADR 0008: Reference notebooks are not redistributed

Status: accepted (2026-10-13)

## Context
The repository started with three notebooks by other authors. Licences were checked via the GitHub API
on 2026-10-13: the GEX notebook's repository is MIT; the Monte Carlo lecture repository has no licence
file (and the notebook carries a copyright line); the Newton-method notebook's licence is unknown.

## Decision
Third-party notebooks live in the git-ignored `references/` folder and are not committed. Their
mathematics is re-implemented, with tests and independent oracles, and credited in
`REFERENCES.md`. MIT-licensed ideas are noted with their notice.

## Consequences
* The public repository contains only original code and synthetic data.
