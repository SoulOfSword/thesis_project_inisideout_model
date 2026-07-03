"""Corner plots: corner.corner for MCMC chains; a finest-level ΔlogL heatmap with
attached 1-D marginals for grid posteriors."""

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


def _cbar_label(external):
    """L_ref when the colour zero-point is not this plot's own maximum, else L_max."""
    return (r"$-\log_{10}(1+\log L_{\rm ref}-\log L)$" if external
            else r"$-\log_{10}(1+\log L_{\max}-\log L)$")


def _logL_heatmap(ax, ax0, ax1, logL, ref=None, contours=True):
    """Log-compressed ΔlogL heatmap (+ optional 1/2/3σ contours) on `ax`; x=ax0, y=ax1.
    Returns the image handle for a colorbar. `ref` sets the colour zero-point (e.g. a full
    grid's peak) so zoomed levels share one absolute scale; contours stay relative to this
    grid's own max."""
    ax0 = np.asarray(ax0); ax1 = np.asarray(ax1); logL = np.asarray(logL)
    disp = -np.log10(1.0 - (logL - (np.nanmax(logL) if ref is None else ref)))
    im = ax.imshow(disp.T, origin="lower", aspect="auto",
                   extent=[ax0[0], ax0[-1], ax1[0], ax1[-1]], cmap="plasma")
    if contours:
        ax.contour(ax0, ax1, (logL - np.nanmax(logL)).T,
                   levels=[-5.92, -3.09, -1.15], colors="w", linewidths=0.8)
    return im


def grid_logL_map(ax0, ax1, logL, labels, peak=None, ref=None, external_ref=False):
    """Log-compressed ΔlogL heatmap over one grid level (+ 1/2/3σ contours, peak marker).
    The log scale keeps the high-L ridge visible despite a coarse scan's huge dynamic range."""
    fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=200, facecolor="w")
    im = _logL_heatmap(ax, ax0, ax1, logL, ref=ref)
    if peak is not None:
        ax.plot(peak[0], peak[1], "r*", ms=13)
    fig.colorbar(im, ax=ax).set_label(_cbar_label(external_ref))
    ax.set_xlabel(rf"${labels[0]}$", fontsize=12)
    ax.set_ylabel(rf"${labels[1]}$", fontsize=12)
    return fig


def grid_corner(ax0, ax1, logL, labels, peak=None, ref=None, external_ref=False):
    """Finest-level ΔlogL heatmap with the two 1-D marginals attached: the top panel is
    parameter 0 marginalised over parameter 1, the left panel parameter 1 over parameter 0.
    Black lines mark the best fit. Returns (fig, stats) with stats=(median, 16th, 84th, mode)
    per parameter from the exact grid marginals."""
    ax0 = np.asarray(ax0); ax1 = np.asarray(ax1); logL = np.asarray(logL, float)
    vi = np.where(np.isfinite(logL).any(axis=1))[0]      # crop the 2-D panel to the valid bounding
    vj = np.where(np.isfinite(logL).any(axis=0))[0]      # box: invalid edge rows/cols are not shown
    ax0, ax1 = ax0[vi[0]:vi[-1] + 1], ax1[vj[0]:vj[-1] + 1]
    logL = logL[vi[0]:vi[-1] + 1, vj[0]:vj[-1] + 1]
    P = np.exp(np.nan_to_num(logL - np.nanmax(logL), nan=-np.inf))
    P /= P.sum()
    marg0, marg1 = P.sum(axis=1), P.sum(axis=0)          # P(param0) over 1, P(param1) over 0
    stats = (_marginal_stats(marg0, ax0), _marginal_stats(marg1, ax1))
    peak = (stats[0][3], stats[1][3])    # lines/star mark the mode; the titles quote the median

    fig = plt.figure(figsize=(6.3, 5.4), dpi=200, facecolor="w")
    # columns: k-marginal | spacer holding the shared y tick labels | heatmap | colorbar
    gs = fig.add_gridspec(2, 4, width_ratios=(1, 0.35, 4, 0.2), height_ratios=(1, 4),
                          wspace=0.10, hspace=0.06)
    ax_main = fig.add_subplot(gs[1, 2])
    ax_top = fig.add_subplot(gs[0, 2], sharex=ax_main)
    ax_left = fig.add_subplot(gs[1, 0], sharey=ax_main)
    cax = fig.add_subplot(gs[1, 3])

    im = _logL_heatmap(ax_main, ax0, ax1, logL, ref=ref)
    ax_main.axvline(peak[0], color="k", lw=1.2)
    ax_main.axhline(peak[1], color="k", lw=1.2)
    ax_main.plot(peak[0], peak[1], "k*", ms=13)
    ax_main.ticklabel_format(useOffset=False)            # show full tick values, no "+1.0" offset
    ax_main.set_xlabel(rf"${labels[0]}$", fontsize=12)
    ax_main.set_ylabel(rf"${labels[1]}$", fontsize=12)
    fig.colorbar(im, cax=cax).set_label(_cbar_label(external_ref))

    ax_top.fill_between(ax0, marg0, color="0.75")
    ax_top.plot(ax0, marg0, color="k", lw=1.2)
    ax_top.axvline(peak[0], color="k", lw=1.2)
    ax_top.set_ylim(0, float(marg0.max()) * 1.15)
    ax_top.set_yticks([]); ax_top.tick_params(labelbottom=False)

    ax_left.fill_betweenx(ax1, marg1, color="0.75")
    ax_left.plot(marg1, ax1, color="k", lw=1.2)
    ax_left.axhline(peak[1], color="k", lw=1.2)
    ax_left.set_xlim(float(marg1.max()) * 1.15, 0)       # probability grows toward the heatmap
    ax_left.set_xticks([]); ax_left.yaxis.tick_right(); ax_left.tick_params(labelleft=False, labelright=False)

    (m0, l0, h0, _), (m1, l1, h1, _) = stats             # titles quote the median + 16/84 interval
    ax_top.set_title(rf"${labels[0]} = {m0:.3f}_{{-{m0-l0:.3f}}}^{{+{h0-m0:.3f}}}$", fontsize=11)
    ax_left.set_title(rf"${labels[1]} = {m1:.3f}_{{-{m1-l1:.3f}}}^{{+{h1-m1:.3f}}}$", fontsize=11)
    return fig, stats
