"""Grid machinery: parallel == serial, peak bracketing, mode counting, zoom convergence.

The per-cell logL is the already-parity-tested wrapper, so these check only the
plumbing (reshape, zoom box, stop logic) with cheap analytic functions.
"""

import numpy as np
from loky import get_reusable_executor

from jmfgas.inference.grid import evaluate_grid, _zoom_box, adaptive_grid
from jmfgas.inference.mcmc import init_worker


class Gauss2D:
    """Picklable 2-D log-Gaussian (loky needs the callable picklable)."""

    def __init__(self, mu, sig):
        self.mu = tuple(mu)
        self.sig = tuple(sig)

    def __call__(self, theta):
        x, y = theta
        return -0.5 * (((x - self.mu[0]) / self.sig[0]) ** 2
                       + ((y - self.mu[1]) / self.sig[1]) ** 2)


def test_evaluate_grid_parallel_equals_serial():
    f = Gauss2D((0.4, 1.3), (0.2, 0.3))
    ax0 = np.linspace(0, 1, 7)
    ax1 = np.linspace(0.5, 2.0, 9)
    ex = get_reusable_executor(max_workers=2, initializer=init_worker)
    try:
        G = evaluate_grid(f, ax0, ax1, ex)
    finally:
        ex.shutdown(wait=True, kill_workers=True)
    serial = np.array([[f((x, y)) for y in ax1] for x in ax0])
    assert G.shape == (7, 9)
    assert np.allclose(G, serial)


def test_zoom_box_brackets_single_peak():
    ax0 = np.linspace(0, 1, 11)
    ax1 = np.linspace(0, 1, 11)
    X, Y = np.meshgrid(ax0, ax1, indexing="ij")
    logL = -50 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)
    lo0, hi0, lo1, hi1, nc = _zoom_box(ax0, ax1, logL, dlogL=6.0)
    assert nc == 1
    assert lo0 <= 0.5 <= hi0 and lo1 <= 0.5 <= hi1
    assert (hi0 - lo0) < (ax0[-1] - ax0[0])      # genuinely zoomed in


def test_zoom_box_counts_two_modes():
    ax0 = np.linspace(0, 1, 21)
    ax1 = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(ax0, ax1, indexing="ij")
    g = lambda cx, cy: np.exp(-200 * ((X - cx) ** 2 + (Y - cy) ** 2))
    logL = np.log(1.0 * g(0.2, 0.2) + 0.9 * g(0.8, 0.8) + 1e-300)
    _, _, _, _, nc = _zoom_box(ax0, ax1, logL, dlogL=6.0)
    assert nc == 2


def test_adaptive_grid_zooms_and_finds_peak():
    f = Gauss2D((0.4, 1.3), (0.04, 0.05))
    res = adaptive_grid(f, [(0.0, 1.0), (0.5, 2.0)], n=9, max_levels=6,
                        target_spacing=0.01, dlogL=6.0, max_workers=2,
                        log=lambda *_: None)
    assert res["stop_reason"] in ("precision", "converged")
    assert res["n_components"] == 1
    assert abs(res["peak"][0] - 0.4) < 0.05    # zoomed well in from the width-1 prior
    assert abs(res["peak"][1] - 1.3) < 0.05
    assert len(res["levels"]) >= 2             # at least one real zoom happened
