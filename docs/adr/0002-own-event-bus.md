# ADR 0002: An in-repo asyncio event bus instead of a library

Status: accepted (2026-10-13)

## Context
The application is event-driven (providers push chains, several consumers react). Libraries were
checked on PyPI on 2026-10-13: `psygnal` 0.16.1 (typed Qt-style signals, threading, no asyncio per
its page), `blinker` 1.9.0 (last released Nov 2024, async not stated), `pyee` 14.0.0 (Node-style
emitter, typing not stated), `bubus` 1.5.6 (pydantic async bus, one maintainer, a release yanked).

## Decision
Write `AsyncEventBus` (under 350 lines, no dependencies) behind `EventPublisher` /
`EventSubscriber` protocols. It provides what the application needs and the libraries do not
combine: per-subscriber mailboxes, `QUEUED` and conflating `LATEST` delivery, error isolation with
`HandlerFailed`, `drain()` for deterministic tests, and subscription statistics.

## Consequences
* No dependency risk; behaviour is specified by tests, including Hypothesis properties for ordering
  and conflation.
* The code is ours to maintain. Because callers depend on the protocols, a library bus could be
  swapped in later.
