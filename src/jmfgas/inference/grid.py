"""Coarse-to-fine log-likelihood grid for 2-D posteriors.

Evaluate a picklable log-probability over a grid in parallel, then zoom into the
high-likelihood region around the peak and refine, until the cell spacing is
small enough or a level cap is hit. Connected-component labelling of the
high-L mask gives a cheap multimodality check.
"""

import numpy as np
from scipy import ndimage
from loky import get_reusable_executor

from .mcmc import init_worker


def evaluate_grid(log_prob, ax0, ax1, executor):
    """log_prob over the outer product ax0 x ax1; shape (len(ax0), len(ax1))."""
    cells = [(float(x), float(y)) for x in ax0 for y in ax1]
    vals = list(executor.map(log_prob, cells))
    return np.asarray(vals, dtype=float).reshape(len(ax0), len(ax1))


def _zoom_box(ax0, ax1, logL, dlogL):
    """Padded bounding box of the high-L component touching the peak.

    Returns (lo0, hi0, lo1, hi1, n_components).
    """
    finite = np.where(np.isfinite(logL), logL, -np.inf)
    peak = np.unravel_index(np.argmax(finite), finite.shape)
    above = finite >= (finite[peak] - dlogL)
    labels, n_comp = ndimage.label(above, structure=np.ones((3, 3), int))
    comp = labels == labels[peak]
    ii, jj = np.where(comp)
    i0 = max(int(ii.min()) - 1, 0)
    i1 = min(int(ii.max()) + 1, len(ax0) - 1)
    j0 = max(int(jj.min()) - 1, 0)
    j1 = min(int(jj.max()) + 1, len(ax1) - 1)
    return ax0[i0], ax0[i1], ax1[j0], ax1[j1], int(n_comp)


def adaptive_grid(log_prob, bounds, n=12, max_levels=5, target_spacing=0.01,
                  dlogL=6.0, max_workers=12, log=print):
    """Zoom a logL grid into the peak until fine enough or out of levels.

    bounds = [(lo0, hi0), (lo1, hi1)]. A fresh worker pool is spun up per level
    (clean state on long HPC jobs). Stops on "precision" (spacing below target),
    "converged" (the zoom box stopped shrinking, i.e. the high-L region is fully
    captured), or "max_levels". Returns all levels, the stop reason, the peak,
    and the connected-component count.
    """
    if n < 3:
        raise ValueError(f"adaptive_grid needs n >= 3 cells per axis, got {n}")
    (lo0, hi0), (lo1, hi1) = bounds
    levels = []
    stop = "max_levels"
    for lvl in range(max_levels):
        ax0 = np.linspace(lo0, hi0, n)
        ax1 = np.linspace(lo1, hi1, n)
        ex = get_reusable_executor(max_workers=max_workers, initializer=init_worker)
        try:
            logL = evaluate_grid(log_prob, ax0, ax1, ex)
        finally:
            ex.shutdown(wait=True, kill_workers=True)   # fresh pool next level
        if not np.isfinite(logL).any():
            raise ValueError("whole grid is outside the prior; check bounds")
        d0 = (hi0 - lo0) / (n - 1)
        d1 = (hi1 - lo1) / (n - 1)
        lo0n, hi0n, lo1n, hi1n, n_comp = _zoom_box(ax0, ax1, logL, dlogL)
        levels.append(dict(ax0=ax0, ax1=ax1, logL=logL,
                           bounds=((lo0, hi0), (lo1, hi1)),
                           spacing=(d0, d1), n_components=n_comp))
        log(f"level {lvl}: {n}x{n} over [{lo0:.4g},{hi0:.4g}]x[{lo1:.4g},{hi1:.4g}] "
            f"spacing=({d0:.3g},{d1:.3g}) peak_logL={np.nanmax(logL):.4g} "
            f"components={n_comp}")
        if d0 < target_spacing and d1 < target_spacing:
            stop = "precision"
            break
        if (hi0n - lo0n) > 0.9 * (hi0 - lo0) and (hi1n - lo1n) > 0.9 * (hi1 - lo1):
            stop = "converged"   # box no longer shrinking; high-L region resolved
            break
        lo0, hi0, lo1, hi1 = lo0n, hi0n, lo1n, hi1n

    last = levels[-1]
    peak = np.unravel_index(np.nanargmax(last["logL"]), last["logL"].shape)
    coarse_comp = levels[0]["n_components"]   # modes show in the full-prior scan, not the zoomed level
    return dict(levels=levels, stop_reason=stop,
                n_components=coarse_comp,
                multimodal=coarse_comp > 1,
                peak=(float(last["ax0"][peak[0]]), float(last["ax1"][peak[1]])),
                peak_logL=float(np.nanmax(last["logL"])))
