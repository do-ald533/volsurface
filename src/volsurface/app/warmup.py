"""Trigger every Numba compilation up front so the first real update is not slow.

Kernels are cached on disk after the first run, so this is quick on later starts; on a fresh
install it can take a few seconds, which is why the GUI does it after showing the window.
"""

import numpy as np

from volsurface.pricing.backends import PricingBackend
from volsurface.pricing.implied_vol import implied_vol_bs
from volsurface.pricing.montecarlo import (
    GbmModel,
    HestonModel,
    MonteCarloEngine,
    Payoff,
    PayoffKind,
)
from volsurface.surface.svi import SviParams, total_variance
from volsurface.surface.svi_fit import fit_svi_slice


def warm_up_numerics(backend: PricingBackend) -> None:
    """Run each compiled routine once on tiny inputs."""
    backend.warmup()
    implied_vol_bs(np.array([5.0]), 100.0, 100.0, 0.25, 0.03, 0.0, True)
    k = np.linspace(-0.2, 0.2, 9)
    fit_svi_slice(k, total_variance(k, SviParams(0.01, 0.1, -0.3, 0.0, 0.1)))
    payoff = Payoff(PayoffKind.CALL, 100.0)
    for model in (
        GbmModel(100.0, 0.03, 0.0, 0.2),
        HestonModel(100.0, 0.04, 0.03, 0.0, 1.5, 0.04, 0.3, -0.7),
    ):
        MonteCarloEngine(model, steps=4).price(payoff, 0.25, 64)
