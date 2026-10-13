"""Process-wide lock for Numba ``parallel=True`` regions.

Numba's default ``workqueue`` threading layer aborts the process if two threads launch parallel
regions at the same time. Every batch kernel in this package takes this lock around its launch, so
callers can use the pricing functions from any thread; each batch still spreads over all cores.
"""

import threading

PARALLEL_LOCK = threading.Lock()
