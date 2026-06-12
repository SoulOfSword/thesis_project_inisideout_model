"""The final model planes (j-M-f_gas, stellar, gaseous) and the mosaic band helpers."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


_CMAP = cm.jet_r

# f_gas levels at which the model tracks are drawn (one line per level). The set is
# model- AND plane-specific, matched to the reference notebooks. The line colour is
# jet_r(level), so this set drives both which curves appear and what colour they get.
FGAS_LEVELS = {
    "io": {
        "jbar":    (0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
        "stellar": (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
        "gaseous": (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
    },
    "nio": {
        "jbar":    (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95),
        "stellar": (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9),
        "gaseous": (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9),
    },
}
_DEFAULT_LEVELS = FGAS_LEVELS["io"]["jbar"]

# compilation samples: marker + the column names that hold their (logM, logj) per plane
_COMPILATION_MARKERS = {"Dwarfs": "p", "superthin": "s", "HIX": "X", "superspirals": "D"}
_COMPILATION_LABELS = {"Dwarfs": "Dwarfs", "superthin": "superthin",
                       "HIX": "HIX", "superspirals": "superspirals"}


def _fgas_tracks(logx, logy, fgas, levels):
    """For each fixed f_gas level, the (logx, logy) curve across the mass grid.

    logx/logy/fgas are (n_mass, n_jacc); per mass we build f_gas -> logx and
    f_gas -> logy interpolators, then read every level off them. Yields
    (level, x_line, y_line) with the line sorted in x (monotone for plotting).
    """
    n_mass = fgas.shape[0]
    fx, fy = [], []
    for i in range(n_mass):
        m = (np.isfinite(fgas[i]) & np.isfinite(logx[i]) & np.isfinite(logy[i]))
        if m.sum() < 2:
            fx.append(None); fy.append(None); continue
        order = np.argsort(fgas[i, m])
        fg = fgas[i, m][order]
        xs = logx[i, m][order]
        ys = logy[i, m][order]
        uniq, ui = np.unique(fg, return_index=True)
        if len(uniq) < 2:
            fx.append(None); fy.append(None); continue
        fx.append((uniq, xs[ui]))
        fy.append((uniq, ys[ui]))

    for lvl in levels:
        xline = np.array([np.interp(lvl, f[0], f[1], left=np.nan, right=np.nan)
                          if f is not None else np.nan for f in fx])
        yline = np.array([np.interp(lvl, g[0], g[1], left=np.nan, right=np.nan)
                          if g is not None else np.nan for g in fy])
        keep = np.isfinite(xline) & np.isfinite(yline)
        if keep.sum() < 2:
            continue
        o = np.argsort(xline[keep])
        yield lvl, xline[keep][o], yline[keep][o]


def _legend(ax, model_lines):
    """Two stacked legends: the f_gas model tracks and the galaxy-sample markers."""
    models_header = plt.Line2D([0], [0], color="none", label=r"$\mathbf{Models}$")
    galaxies_header = plt.Line2D([0], [0], color="none", label=r"$\mathbf{Galaxies}$")
    gal = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                      markeredgecolor="k", markersize=10, label="MP+21b")]
    gal += [plt.Line2D([0], [0], marker=_COMPILATION_MARKERS[name], color="w",
                       markerfacecolor="none", markeredgecolor="k", markersize=10,
                       label=_COMPILATION_LABELS[name])
            for name in _COMPILATION_MARKERS]
    leg1 = ax.legend(handles=[models_header] + model_lines, loc="upper left",
                     framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=[galaxies_header] + gal, loc="upper left",
              bbox_to_anchor=(0.21, 1.0), framealpha=0.9)


def _draw_plane(ax, logM, logx, logy, fgas, obs_x, obs_y, obs_fgas,
                obs_x_err, obs_y_err, compilation, xcol, ycol,
                xlabel, ylabel, title, levels=_DEFAULT_LEVELS, ylim_bottom=None):
    """Shared body: model tracks coloured by f_gas, observed sample, compilation."""
    model_lines = []
    for lvl, xs, ys in _fgas_tracks(logx, logy, fgas, levels):
        # colour each track by its OWN f_gas level: _fgas_tracks drops levels with
        # <2 points, so a positional colour list would shift every later line down.
        line = ax.plot(xs, ys, lw=3.5, color=_CMAP(lvl), zorder=2,
                       label=fr"$f_{{\rm gas}}={lvl:.2f}$")[0]
        model_lines.append(line)

    sc = ax.scatter(obs_x, obs_y, marker="o", facecolors=_CMAP(obs_fgas),
                    edgecolors="grey", s=70, zorder=9, label="MP+21b")
    ax.errorbar(obs_x, obs_y, xerr=obs_x_err, yerr=obs_y_err, fmt=" ",
                ecolor="grey", capsize=3, alpha=0.3, zorder=0)

    for i, (name, df) in enumerate(compilation.items(), start=3):
        marker = _COMPILATION_MARKERS.get(name)
        if marker is None or xcol not in df or ycol not in df:
            continue
        ax.scatter(df[xcol], df[ycol], marker=marker,
                   facecolors=_CMAP(np.asarray(df["fgas"], float)),
                   edgecolors="grey", s=70, zorder=i)

    _legend(ax, model_lines)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(labelsize=14)
    ax.grid(zorder=1)
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    return sc


def _colorbar(ax):
    """f_gas colorbar (jet_r over 0..1) in a slot to the right of the axes.

    The points are drawn with explicit facecolors, so the colorbar gets its own
    ScalarMappable instead of one of the scatters (which would lose the colormap).
    """
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
    sm = cm.ScalarMappable(cmap=_CMAP, norm=Normalize(0.0, 1.0))
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, cax=cax)
    cbar.set_label(r"$f_{\rm gas}$", rotation=270, fontsize=15, labelpad=15)
    cbar.ax.tick_params(labelsize=14)


def plane_jM_fgas(ax, logM, j_bar, f_gas, obs, compilation, params_label="",
                  levels=_DEFAULT_LEVELS):
    """j_bar - M_bar - f_gas plane: model tracks coloured by f_gas + observed + compilation.

    logM (n_mass,), j_bar/f_gas (n_mass, n_jacc) are the present-day per-mass model grids.
    obs is a dict with log_Mbar, log_jbar, fgas and the *_err arrays. compilation maps a
    sample name to its DataFrame (logMbar/logjbar/fgas columns).
    """
    logj = np.log10(np.where(np.asarray(j_bar) > 0, j_bar, np.nan))
    logMmat = np.broadcast_to(np.asarray(logM)[:, None], logj.shape)
    title = (r"$\log(j_{\rm bar})$ vs $\log(M_{\rm bar})$ vs $f_{\rm gas}$"
             + ("\n" + params_label if params_label else ""))
    sc = _draw_plane(
        ax, logM, logMmat, logj, np.asarray(f_gas),
        obs["log_Mbar"], obs["log_jbar"], obs["fgas"],
        obs["log_Mbar_err"], obs["log_jbar_err"], compilation, "logMbar", "logjbar",
        r"$\log(M_{\rm bar} \, / \, \rm M_{\odot})$",
        r"$\log(j_{\rm bar} \, / \, \rm kpc \, km \, s^{-1})$", title,
        levels=levels, ylim_bottom=0.5)
    _colorbar(ax)
    return sc


def plane_stellar(ax, M_star, j_star, f_gas, obs, compilation, params_label="",
                  levels=_DEFAULT_LEVELS):
    """j_star - M_star - f_gas plane (model tracks coloured by f_gas + observed + compilation)."""
    logMmat = np.log10(np.where(np.asarray(M_star) > 0, M_star, np.nan))
    logj = np.log10(np.where(np.asarray(j_star) > 0, j_star, np.nan))
    title = (r"$\log(j_{\star})$ vs $\log(M_{\star})$ vs $f_{\rm gas}$"
             + ("\n" + params_label if params_label else ""))
    sc = _draw_plane(
        ax, None, logMmat, logj, np.asarray(f_gas),
        obs["log_Mstar"], obs["log_jstar"], obs["fgas"],
        obs["log_Mstar_err"], obs["log_jstar_err"], compilation, "logMstar", "logjstar",
        r"$\log(M_{\star} \, / \, \rm M_{\odot})$",
        r"$\log(j_{\star} \, / \, \rm kpc \, km \, s^{-1})$", title, levels=levels)
    _colorbar(ax)
    return sc


def plane_gaseous(ax, M_gas, j_gas, f_gas, obs, compilation, params_label="",
                  levels=_DEFAULT_LEVELS):
    """j_gas - M_gas - f_gas plane (model tracks coloured by f_gas + observed + compilation)."""
    logMmat = np.log10(np.where(np.asarray(M_gas) > 0, M_gas, np.nan))
    logj = np.log10(np.where(np.asarray(j_gas) > 0, j_gas, np.nan))
    title = (r"$\log(j_{\rm gas})$ vs $\log(M_{\rm gas})$ vs $f_{\rm gas}$"
             + ("\n" + params_label if params_label else ""))
    sc = _draw_plane(
        ax, None, logMmat, logj, np.asarray(f_gas),
        obs["log_Mgas"], obs["log_jgas"], obs["fgas"],
        obs["log_Mgas_err"], obs["log_jgas_err"], compilation, "logMgas", "logjgas",
        r"$\log(M_{\rm gas} \, / \, \rm M_{\odot})$",
        r"$\log(j_{\rm gas} \, / \, \rm kpc \, km \, s^{-1})$", title, levels=levels)
    _colorbar(ax)
    return sc


def prep_curve(jbar, fgas):
    """Filter/sort a (j_bar, f_gas) model curve -> (log10 j_bar, f_gas), unique in f_gas."""
    jbar = np.asarray(jbar, float)
    fgas = np.asarray(fgas, float)
    v = (fgas > 0.01) & (fgas < 0.99) & (jbar > 0) & np.isfinite(jbar) & np.isfinite(fgas)
    f, lj = fgas[v], np.log10(jbar[v])
    o = np.argsort(f)
    f, lj = f[o], lj[o]
    f, ui = np.unique(f, return_index=True)
    return lj[ui], f[ui]


def swept_band(curves):
    """Band = swept region of a bin's model curves. The boundary runs along the lowest-mass
    curve, the locus of high-f ends, the highest-mass curve, then the locus of low-f ends ->
    a smooth closed polygon enclosing every curve in the bin. Returns (x=log j_bar, y=f_gas)."""
    prepped = [prep_curve(j, f) for j, f in curves]
    prepped = [(lj, f) for lj, f in prepped if len(f) >= 2]
    if len(prepped) < 2:
        return None
    order = np.argsort([lj.mean() for lj, f in prepped])      # low-mass -> high-mass
    prepped = [prepped[o] for o in order]
    lj_lo, f_lo = prepped[0]                                  # lowest-mass curve
    lj_hi, f_hi = prepped[-1]                                 # highest-mass curve
    starts = np.array([(lj[0], f[0]) for lj, f in prepped])   # gas-poor (low-f) ends
    ends = np.array([(lj[-1], f[-1]) for lj, f in prepped])   # gas-rich (high-f) ends
    px = np.concatenate([lj_lo, ends[:, 0], lj_hi[::-1], starts[::-1, 0]])
    py = np.concatenate([f_lo,  ends[:, 1], f_hi[::-1], starts[::-1, 1]])
    return px, py
