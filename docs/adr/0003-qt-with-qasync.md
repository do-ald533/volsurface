# ADR 0003: PySide6, pyqtgraph and one event loop via qasync

Status: accepted (2026-10-13)

## Context
A desktop application that streams data, plots a surface and stays responsive, written in Python.

## Decision
PySide6 (official Qt for Python, LGPL) for the shell and models/views, pyqtgraph for real-time
plots and its OpenGL surface item, `qasync` to run asyncio inside the Qt loop so `ib_async` and
the GUI share one thread. CPU-heavy work runs on a worker thread and Numba kernels release the GIL.
`ui/qt_bridge.py` is the only place the bus meets Qt.

## Consequences
* No cross-thread signal plumbing for I/O.
* The 3D view needs a desktop OpenGL context; on platforms without one (Qt's offscreen platform
  used by CI) the heatmap tab carries the same data and the 3D tab says why it is unavailable.
* `qasync` 0.28.0's README caps Python below 3.14, so the project pins Python 3.13.
