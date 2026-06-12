"""Corner plots: one corner.corner style for both MCMC chains and grid posteriors."""

import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy.stats import gaussian_kde


def chain_corner(flat, labels, title=None):
    """corner.corner with median truths plus mode (red) and 16/84 (red dashed) lines."""
    ndim = flat.shape[1]
    medians, modes, p16, p84 = [], [], [], []
    for i in range(ndim):
        s = flat[:, i]
        lo, mid, hi = np.percentile(s, [16, 50, 84])
        medians.append(mid); p16.append(lo); p84.append(hi)
        grid = np.linspace(s.min(), s.max(), 1000)
        modes.append(grid[np.argmax(gaussian_kde(s)(grid))])
    fig = corner.corner(flat, labels=labels, levels=(1 - np.exp(-0.5), 1 - np.exp(-2.0)),
                        show_titles=True, truths=medians, truth_color="blue",
                        title_kwargs={"fontsize": 12}, smooth=1.5, title_fmt=".3f")
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = axes[i, i]
        ax.axvline(modes[i], color="red", lw=1.5)
        ax.axvline(p16[i], color="red", ls="--", lw=1.2)
        ax.axvline(p84[i], color="red", ls="--", lw=1.2)
        ax.axvline(medians[i], color="blue", lw=1.5)
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    return fig


def _marginal_stats(p, x):
    """Median, 16th, 84th percentile and mode of a 1-D probability profile on grid x."""
    c = np.cumsum(p) / p.sum()
    lo, mid, hi = np.interp([0.16, 0.5, 0.84], c, x)
    return mid, lo, hi, x[np.argmax(p)]


def sample_grid_posterior(ax0, ax1, logL, n=60000, seed=0):
    """Draw samples from the grid posterior P ∝ exp(logL), jittered within each cell, so
    the grid corner can be rendered by the same corner.corner path as the chains."""
    ax0 = np.asarray(ax0); ax1 = np.asarray(ax1)
    P = np.exp(np.nan_to_num(logL - np.nanmax(logL), nan=-np.inf)).ravel()
    P = P / P.sum()
    rng = np.random.default_rng(seed)
    i, j = np.unravel_index(rng.choice(P.size, size=n, p=P), np.asarray(logL).shape)
    d0 = ax0[1] - ax0[0] if ax0.size > 1 else 0.0
    d1 = ax1[1] - ax1[0] if ax1.size > 1 else 0.0
    s0 = ax0[i] + rng.uniform(-0.5, 0.5, n) * d0
    s1 = ax1[j] + rng.uniform(-0.5, 0.5, n) * d1
    return np.column_stack([s0, s1])


def grid_logL_map(ax0, ax1, logL, labels, peak=None):
    """Log-compressed ΔlogL heatmap over one grid level (+ 1/2/3σ contours, peak marker).
    The log scale keeps the high-L ridge visible despite the huge dynamic range of a
    coarse scan, where the peak-dominated posterior corner would hide it."""
    ax0 = np.asarray(ax0); ax1 = np.asarray(ax1)
    Z = np.asarray(logL) - np.nanmax(logL)               # <= 0, zero at the peak
    disp = -np.log10(1.0 - Z)                            # compress the (huge) dynamic range
    fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=200, facecolor="w")
    im = ax.imshow(disp.T, origin="lower", aspect="auto",
                   extent=[ax0[0], ax0[-1], ax1[0], ax1[-1]], cmap="viridis")
    ax.contour(ax0, ax1, Z.T, levels=[-5.92, -3.09, -1.15], colors="w", linewidths=0.8)
    if peak is not None:
        ax.plot(peak[0], peak[1], "r*", ms=13)
    fig.colorbar(im, ax=ax).set_label(r"$-\log_{10}(1+\log L_{\max}-\log L)$")
    ax.set_xlabel(rf"${labels[0]}$", fontsize=12)
    ax.set_ylabel(rf"${labels[1]}$", fontsize=12)
    return fig


def grid_corner(ax0, ax1, logL, labels):
    """Grid posterior rendered like a chain corner: sample P ∝ exp(logL) and pass to
    corner.corner, so grid and MCMC figures share one look (titles, median truths,
    contours). Returns (fig, stats) with stats=(median, 16th, 84th, mode) per parameter
    taken from the exact grid marginals."""
    samples = sample_grid_posterior(ax0, ax1, logL)
    fig = chain_corner(samples, [f"${l}$" for l in labels])
    P = np.exp(np.nan_to_num(logL - np.nanmax(logL), nan=-np.inf))
    P /= P.sum()
    stats = (_marginal_stats(P.sum(axis=1), np.asarray(ax0)),
             _marginal_stats(P.sum(axis=0), np.asarray(ax1)))
    return fig, stats
