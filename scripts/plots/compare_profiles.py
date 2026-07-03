"""Comparison figures for the IO vs NIO models (vs observations).

Figures (pick with the positional command, default `all`):
  profiles   : per-galaxy 2x2 radial-profile comparison (Sigma_gas, Sigma_star,
               Sigma_sfr, SFH), IO vs NIO vs observed.
  evolution  : per-galaxy radial-profile time-evolution panels (one fig per model).
  mosaic     : the j_bar-f_gas mosaic (mass-bin panels: model band + observed/compiled data).
  comparison : the assumptions-vs-results model-illustration figures (read from the
               comparison .npz produced by save_comparison_npz.py).

  --galaxies : galaxies for the per-galaxy figures (profiles, evolution).
  --bins     : mass bins (logM_low,logM_high pairs) for the mosaic panels.
"""

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jmfgas.config import load_config
from jmfgas.data import build_converged, sample_frame
from jmfgas.viz import prep_curve, swept_band
from jmfgas.viz.planes import _COMPILATION_MARKERS

DEFAULT_GALAXIES = ["NGC3627", "UGC04278"]
DEFAULT_BINS = [(9.0, 9.5), (9.7, 10.3), (10.8, 11.2)]

# mosaic data markers/colors per sample group (markers match the final planes)
GROUP_MARKER = {"MP+21b": "o", **_COMPILATION_MARKERS}
GROUP_COLOR = {"MP+21b": "magenta", "Dwarfs": "orange", "superthin": "cyan", "HIX": "green",
               "superspirals": "purple", "GLSBs": "brown", "UDGs": "deeppink"}


# ── data loading ─────────────────────────────────────────────────────────────

def _load_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _profile_at_t0(profile, key):
    arr = profile[key]
    return arr[:, -1] if arr.ndim == 2 else arr


def _load_observed_HI(hi_dir, galaxy):
    fp = hi_dir / f"{galaxy}_fullHI.dat"
    if not fp.exists():
        return None
    d = np.genfromtxt(fp, skip_header=1)
    return {"r_kpc": d[:, 0], "Sigma_HI": d[:, 2]}


def _load_observed_SFR(sfr_dir, galaxy):
    fp = sfr_dir / f"{galaxy}_R_Vrot_SigmaGas_SigmaSFR.txt"
    if not fp.exists():
        return None
    d = np.genfromtxt(fp, skip_header=1)
    return {"r_kpc": d[:, 0], "Sigma_gas": d[:, 3] / 1e6, "Sigma_SFR": d[:, 5] * 1e9 / 1e6}


def _load_observed_stars(stars_dir, galaxy):
    fp = stars_dir / f"{galaxy}_fullstars.dat"
    if not fp.exists():
        return None
    d = np.genfromtxt(fp, skip_header=1)
    return {"r_kpc": d[:, 0], "Sigma_star": d[:, 3]}


def _load_galaxy(galaxy, paths):
    p = {"io": None, "nio": None, "obs_HI": None, "obs_SFR": None, "obs_stars": None}
    io_file = paths["io_prof"] / f"{galaxy}_profile.npz"
    if io_file.exists():
        p["io"] = _load_npz(io_file)
    nio_file = paths["nio_prof"] / f"{galaxy}_profile.npz"
    if nio_file.exists():
        p["nio"] = _load_npz(nio_file)
    p["obs_HI"] = _load_observed_HI(paths["hi"], galaxy)
    p["obs_SFR"] = _load_observed_SFR(paths["sfr"], galaxy)
    p["obs_stars"] = _load_observed_stars(paths["stars"], galaxy)
    return p


# ── figure: per-galaxy 2x2 radial-profile comparison ─────────────────────────

def plot_galaxy_comparison(galaxy, p, x_max=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"{galaxy} - Model vs Observed", fontsize=16, fontweight="bold")

    io, nio = p["io"], p["nio"]
    obs_hi, obs_sfr, obs_stars = p["obs_HI"], p["obs_SFR"], p["obs_stars"]

    if x_max is None and io is not None:
        r_kpc = io["r_kpc"]
        final_gas = _profile_at_t0(io, "Sigma_gas")
        valid = np.where(final_gas > 1e-2)[0]
        x_max = r_kpc[valid[-1]] * 1.1 if len(valid) > 0 else 30
    elif x_max is None:
        x_max = 30

    # Gas
    ax = axes[0, 0]
    ax.set_title(r"Gas Surface Density")
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel(r"$\Sigma_{\rm gas}$ (M$_\odot$ / pc$^2$)")
    ax.set_yscale("log")
    if io is not None:
        ax.plot(io["r_kpc"], _profile_at_t0(io, "Sigma_gas"), "-", color="C0", label="IO", lw=2)
    if nio is not None and "Sigma_gas" in nio:
        ax.plot(nio["r_kpc"], _profile_at_t0(nio, "Sigma_gas"), "--", color="C1", label="NIO", lw=2)
    if obs_hi is not None:
        ax.scatter(obs_hi["r_kpc"], 1.4 * obs_hi["Sigma_HI"], color="k", marker="o",
                   s=30, alpha=0.7, label="Obs")
    ax.set_xlim(0, x_max)
    ax.set_ylim(1e-2, 1e3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Stars
    ax = axes[0, 1]
    ax.set_title("Stellar Surface Density")
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel(r"$\Sigma_{\star}$ (M$_\odot$ / pc$^2$)")
    ax.set_yscale("log")
    if io is not None:
        ax.plot(io["r_kpc"], _profile_at_t0(io, "Sigma_star"), "-", color="C0", label="IO", lw=2)
    if nio is not None and "Sigma_star" in nio:
        ax.plot(nio["r_kpc"], _profile_at_t0(nio, "Sigma_star"), "--", color="C1", label="NIO", lw=2)
    if obs_stars is not None:
        ax.scatter(obs_stars["r_kpc"], obs_stars["Sigma_star"], color="k", marker="o",
                   s=30, alpha=0.7, label="Obs")
    ax.set_xlim(0, x_max)
    ax.set_ylim(1e-3, 1e4)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SFR
    ax = axes[1, 0]
    ax.set_title("SFR Surface Density")
    ax.set_xlabel("R (kpc)")
    ax.set_ylabel(r"$\Sigma_{\rm sfr}$ (M$_\odot$ / pc$^2$/Gyr)")
    ax.set_yscale("log")
    if io is not None:
        ax.plot(io["r_kpc"], _profile_at_t0(io, "Sigma_sfr"), "-", color="C0", label="IO", lw=2)
    if nio is not None and "Sigma_sfr" in nio:
        ax.plot(nio["r_kpc"], _profile_at_t0(nio, "Sigma_sfr"), "--", color="C1", label="NIO", lw=2)
    if obs_sfr is not None and obs_sfr["Sigma_SFR"] is not None:
        ax.scatter(obs_sfr["r_kpc"], obs_sfr["Sigma_SFR"], color="k", marker="o",
                   s=30, alpha=0.7, label="Obs")
    ax.set_xlim(0, x_max)
    ax.set_ylim(1e-6, 1e2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # SFH
    ax = axes[1, 1]
    ax.set_title("Star Formation History")
    ax.set_xlabel("Time (Gyr)")
    ax.set_ylabel(r"SFR (M$_\odot$ / Gyr)")
    ax.set_yscale("log")
    if io is not None and "SFH" in io:
        ax.plot(io["times"], io["SFH"], "-", color="C0", label="IO", lw=2)
    if nio is not None and "SFH" in nio:
        ax.plot(nio["times"], nio["SFH"], "--", color="C1", label="NIO", lw=2)
    ax.set_xlim(0, 12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if io is not None:
        fig.text(0.5, 0.03,
                 rf"IO: ${{\log M_{{\rm bar}}}}={float(io.get('log_M_bar', 0)):.2f}, "
                 rf"t_{{\rm acc}}={float(io.get('t_acc', 0)):.2f}$ Gyr",
                 ha="center", fontsize=11)
    if nio is not None:
        fig.text(0.5, 0.01,
                 rf"NIO: ${{\log M_{{\rm bar}}}}={float(nio.get('log_M_bar', 0)):.2f}, "
                 rf"t_{{\rm acc}}={float(nio.get('t_acc', 0)):.2f}$ Gyr",
                 ha="center", fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# ── figure: per-galaxy time-evolution panels ─────────────────────────────────

def plot_time_evolution(galaxy, p, times_to_plot=(3, 6, 9, 12), x_max=None, model_type="io"):
    io = p["io"]
    nio = p.get("nio", None)
    if io is None and nio is None:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    fig.suptitle(f"{galaxy} - Time Evolution ({model_type.upper()})", fontsize=14, fontweight="bold")

    r_kpc, times = io["r_kpc"], io["times"]
    cmap = plt.cm.viridis

    for t_target in times_to_plot:
        t_idx = np.argmin(np.abs(times - t_target))
        color = cmap(t_target / 12.0)
        if model_type == "io":
            axes[0].plot(r_kpc, io["Sigma_gas"][:, t_idx], color=color, label=f"t={times[t_idx]:.1f}")
            axes[1].plot(r_kpc, io["Sigma_star"][:, t_idx], color=color)
            axes[2].plot(r_kpc, io["Sigma_sfr"][:, t_idx], color=color)
        elif model_type == "nio":
            axes[0].plot(r_kpc, nio["Sigma_gas"][:, t_idx], color=color, label=f"t={times[t_idx]:.1f}")
            axes[1].plot(r_kpc, nio["Sigma_star"][:, t_idx], color=color)
            axes[2].plot(r_kpc, nio["Sigma_sfr"][:, t_idx], color=color)
        else:
            raise ValueError("model_type must be 'io' or 'nio'")

    if x_max is None:
        final_gas = io["Sigma_gas"][:, -1]
        valid = np.where(final_gas > 1e-2)[0]
        x_max = r_kpc[valid[-1]] * 1.1 if len(valid) > 0 else r_kpc[-1]

    for ax, title, ylim in zip(axes,
                               [r"$\Sigma_{\rm gas}$", r"$\Sigma_{\star}$", r"$\Sigma_{\rm sfr}$"],
                               [(1e-2, 1e3), (1e-3, 1e4), (1e-6, 1e2)]):
        ax.set_xlabel("R (kpc)")
        ax.set_yscale("log")
        ax.set_xlim(0, x_max)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    plt.tight_layout()
    return fig


# ── figure: j_bar-f_gas mosaic ───────────────────────────────────────────────

_IO_OMEGA = np.array([-1.0, -0.3, 0.1, 1.0 / 3.0, 0.75, 1.0, 2.0, 4.0, 8.0, 10.0])


def _io_grids_live(logM, n, k, sfl):
    """IO present-day f_gas/j_bar over (mass x omega) at (n, k), run live (no saved tables)."""
    import jax.numpy as jnp
    from jmfgas.models.inside_out import build_r_acc_matrix_for_all_M_jax, run_all_masses
    t_acc = jnp.asarray(1.0 / _IO_OMEGA, dtype=jnp.float64)
    r_acc = build_r_acc_matrix_for_all_M_jax(jnp.float64(n), jnp.float64(k))
    out = run_all_masses(jnp.asarray(10.0 ** logM, dtype=jnp.float64), t_acc, r_acc,
                         jnp.asarray(logM, dtype=jnp.float64), star_formation_law=sfl)
    return {"f_gas": np.asarray(out[0]), "j_bar": np.asarray(out[1])}


def _nio_band_live(logM, a, b, n_j, sfl):
    """NIO (j_bar, f_gas) swept over j_acc at fixed mass, omega from the (a, b) power law."""
    import jax.numpy as jnp
    from jmfgas.models import build_r_acc_for_single_M
    from jmfgas.models.non_inside_out import Full_final_definer_Mdep_omega_jax
    r_acc_pc, _ = build_r_acc_for_single_M(logM, n_j=n_j)
    f_gas, j_bar, _, _, _, _ = Full_final_definer_Mdep_omega_jax(
        float(logM), a, b, jnp.asarray(r_acc_pc, dtype=jnp.float64),
        star_formation_law=sfl, at_t0=True)
    return np.asarray(j_bar), np.asarray(f_gas)


def plot_mosaic(bins, paths, n, k, a, b, sfl, n_j, band_step=0.05):
    full = sample_frame("full", paths["data"])             # full sample: all 7 groups, with errors
    full_logM = np.log10(full["Mbar"].to_numpy(float))
    full_logj = np.log10(full["jbar"].to_numpy(float))
    full_fgas = full["fgas"].to_numpy(float)
    full_jbar = full["jbar"].to_numpy(float)
    full_group = full["group"].to_numpy()
    e_logj = full["e_jbar"].to_numpy(float) / full_jbar / np.log(10)
    e_fg = full["e_fgas"].to_numpy(float)
    for lo, hi in bins:
        print(f"  bin [{lo}, {hi}]: {int(((full_logM >= lo) & (full_logM <= hi)).sum())} galaxies (full)")

    logM_grid_io = np.linspace(8, 11.5, 50)
    io_grids = _io_grids_live(logM_grid_io, n, k, sfl)              # live at (n, k); no saved tables
    io_fgrid, io_jgrid = io_grids["f_gas"], io_grids["j_bar"]

    nio_band = {}
    for lo, hi in bins:
        for m in np.round(np.arange(lo, hi + 1e-9, band_step), 4):
            if m not in nio_band:
                nio_band[m] = _nio_band_live(float(m), a, b, n_j, sfl)   # live, power-law omega
    nio_band_masses = np.array(sorted(nio_band.keys()))
    have_nio_band = len(nio_band_masses) > 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150, sharey=True)
    for i, (lo, hi) in enumerate(bins):
        ax = axes[i]
        cen = 0.5 * (lo + hi)

        io_rows = np.where((logM_grid_io >= lo) & (logM_grid_io <= hi))[0]
        r_cen = io_rows[np.argmin(np.abs(logM_grid_io[io_rows] - cen))]
        lj_c, f_c = prep_curve(io_jgrid[r_cen], io_fgrid[r_cen])
        band = swept_band([(io_jgrid[r], io_fgrid[r]) for r in io_rows])
        if band is not None:
            ax.fill(band[0], band[1], color="blue", alpha=0.18, lw=0, zorder=1)
        ax.plot(lj_c, f_c, color="blue", lw=3, label="Inside-out", zorder=3)

        if have_nio_band:
            in_bin = nio_band_masses[(nio_band_masses >= lo) & (nio_band_masses <= hi)]
            if len(in_bin) >= 1:
                m_cen = in_bin[np.argmin(np.abs(in_bin - cen))]
                lj_c, f_c = prep_curve(*nio_band[m_cen])
                band = swept_band([nio_band[m] for m in in_bin])
                if band is not None:
                    ax.fill(band[0], band[1], color="red", alpha=0.18, lw=0, zorder=1)
                ax.plot(lj_c, f_c, color="red", lw=3, ls="--", label="Non-inside-out", zorder=3)
            else:
                print(f"  [warn] no NIO band masses in [{lo}, {hi}]")
        else:
            mlbl = int(round(cen))
            d = pd.read_csv(paths["nio_prof_csv"] / f"jbar_fgas_logM{mlbl}.csv")
            lj_c, f_c = prep_curve(d["j_bar_kpc_km_s"].values, d["f_gas"].values)
            ax.plot(lj_c, f_c, color="red", lw=3, ls="--",
                    label=f"Non-inside-out ($10^{{{mlbl}}}$)", zorder=3)

        in_bin = (full_logM >= lo) & (full_logM <= hi) & (full_jbar > 0) & (full_fgas > 0)
        for grp in GROUP_MARKER:                           # every sample group present in the bin
            m = in_bin & (full_group == grp)
            if not m.any():
                continue
            ax.scatter(full_logj[m], full_fgas[m], c=GROUP_COLOR.get(grp, "gray"), s=55,
                       edgecolors="k", alpha=0.8, zorder=6, marker=GROUP_MARKER[grp], label=grp)
            ax.errorbar(full_logj[m], full_fgas[m], xerr=e_logj[m], yerr=e_fg[m], fmt=" ",
                        ecolor="grey", capsize=2, alpha=0.4, zorder=4)

        ax.set_title(rf"$10^{{{lo:g}}}\!-\!10^{{{hi:g}}}\,M_\odot$", fontsize=15)
        ax.set_xlabel(r"$\log(j_{\rm bar} \, / \, {\rm kpc \, km \, s^{-1}})$", fontsize=16)
        if i == 0:
            ax.set_ylabel(r"$f_{\rm gas}$", fontsize=16)
        ax.grid()
        ax.tick_params(axis="both", labelsize=14)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  loc="lower right" if i == 0 else "upper left", fontsize=10)

    plt.tight_layout()
    return fig


# ── figures: assumptions-vs-results comparison ───────────────────────────────

def _comparison_io_live(logM, n, k, sfl):
    """IO growing/decaying-omega illustration profiles at (n, k), run live -> (pos, neg)."""
    import jax.numpy as jnp
    from jmfgas.models.common import log_M_bar_array_jax, M_times1, C_def_jax
    from jmfgas.models.inside_out import build_r_acc_matrix_for_all_M_jax
    from jmfgas.physics.angmom import j_maxer, j_acc_def
    from jmfgas.models.profiles import radial_profiles_io
    M_bar = 10.0 ** logM
    r_acc = build_r_acc_matrix_for_all_M_jax(jnp.float64(n), jnp.float64(k))
    j_acc_t = np.asarray(j_acc_def(j_maxer(M_bar), M_times1, n=n, con=k))
    dicts = []
    for omega in (1.0 / 2.0, -1.0 / 2.0):
        prof = {kk: (np.asarray(v) if hasattr(v, "shape") else v)
                for kk, v in radial_profiles_io(logM, 1.0 / omega, r_acc,
                                                 log_M_bar_array_jax, sfl_type=sfl).items()}
        prof.update(omega=omega, C=float(C_def_jax(M_bar, 1.0 / omega)),
                    M_bar=M_bar, j_acc_t=j_acc_t, k=k, n=n)
        dicts.append(prof)
    return dicts[0], dicts[1]


def _comparison_nio_live(logM, a, b, sfl):
    """NIO high/low-j_acc illustration profiles with the power-law omega, run live -> (high, low)."""
    import jax.numpy as jnp
    from jmfgas.models.common import C_def_jax
    from jmfgas.physics.angmom import j_maxer
    from jmfgas.models.non_inside_out import omega_Mdep
    from jmfgas.models.profiles import radial_profiles_nio
    M_bar = 10.0 ** logM
    j_max = float(j_maxer(M_bar))
    omega = float(omega_Mdep(jnp.float64(logM), a, b))           # power law (was linear a*(logM-10)+b)
    C = float(C_def_jax(M_bar, 1.0 / omega))
    dicts = []
    for j_val in (j_max, j_max / 10.0):
        prof = dict(radial_profiles_nio(logM, j_val, a, b, sfl_type=sfl))
        prof.update(C=C, j_acc_value=j_val, M_bar=M_bar)         # prof['omega'] is already power-law
        dicts.append(prof)
    return dicts[0], dicts[1]


def plot_nio_2panel(nio_high, nio_low):
    times = nio_high["times"]
    omega = float(nio_high["omega"])
    C_nio = float(nio_high["C"])
    j_high = float(nio_high["j_acc_value"])
    j_low = float(nio_low["j_acc_value"])
    M_dot_acc = C_nio * np.exp(-omega * times)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 3.2), dpi=300)

    ax_left.set_title("Assumptions", fontsize=9)
    ax_left.plot(times, M_dot_acc, "k-", lw=1.5, label=r"$\dot{M}_{\rm acc}(t)$")
    ax_left.set_xlabel("Time (Gyr)", fontsize=8)
    ax_left.set_ylabel(r"$\dot{M}_{\rm acc}$ ($M_\odot\,{\rm Gyr}^{-1}$)", fontsize=8)
    ax_left.set_xlim(0, 12)
    ax_left.set_yscale("log")
    ax_left.tick_params(axis="both", labelsize=7)
    ax_left.grid(True, alpha=0.3)

    ax_j = ax_left.twinx()
    ax_j.axhline(j_high, color="red", ls="--", lw=1.5, label=r"$j_{\rm acc} = j_{\rm max}$")
    ax_j.axhline(j_low, color="blue", ls="--", lw=1.5, label=r"$j_{\rm acc} = j_{\rm max}/10$")
    ax_j.set_ylabel(r"$j_{\rm acc}$ (kpc km s$^{-1}$)", fontsize=8)
    ax_j.set_ylim(0, j_high * 1.3)
    ax_j.tick_params(axis="y", labelsize=7)

    l1, lab1 = ax_left.get_legend_handles_labels()
    l2, lab2 = ax_j.get_legend_handles_labels()
    ax_left.legend(l1 + l2, lab1 + lab2, loc="center left", fontsize=7, framealpha=0.9)

    ax_right.set_title("Results", fontsize=9)
    ax_right.plot(times, nio_high["f_gas_t"], "r-", lw=1.5, label=r"$f_{\rm gas}$ (high $j_{\rm acc}$)")
    ax_right.plot(times, nio_low["f_gas_t"], "b-", lw=1.5, label=r"$f_{\rm gas}$ (low $j_{\rm acc}$)")
    ax_right.set_xlabel("Time (Gyr)", fontsize=8)
    ax_right.set_ylabel(r"$f_{\rm gas}$", fontsize=8)
    ax_right.set_xlim(0.1, 12)
    ax_right.set_ylim(0, 1.05)
    ax_right.tick_params(axis="both", labelsize=7)
    ax_right.grid(True, alpha=0.3)

    ax_jbar = ax_right.twinx()
    ax_jbar.plot(times, nio_high["j_bar_t"], "r--", lw=1.2, label=r"$j_{\rm bar}$ (high $j_{\rm acc}$)")
    ax_jbar.plot(times, nio_low["j_bar_t"], "b--", lw=1.2, label=r"$j_{\rm bar}$ (low $j_{\rm acc}$)")
    ax_jbar.set_ylabel(r"$j_{\rm bar}$ (kpc km s$^{-1}$)", fontsize=8)
    ax_jbar.tick_params(axis="y", labelsize=7)

    l1, lab1 = ax_right.get_legend_handles_labels()
    l2, lab2 = ax_jbar.get_legend_handles_labels()
    ax_right.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=7, framealpha=0.9)

    fig.suptitle(rf"Non-Inside-Out Model ($M_{{\rm bar}} = 10^{{{np.log10(float(nio_high['M_bar'])):.1f}}}"
                 rf"\,M_\odot$, $\omega_{{\rm acc}} = {omega:.2g}\,{{\rm Gyr}}^{{-1}}$)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_io_2panel(io_pos, io_neg):
    times_io = io_pos["times"]
    j_acc_t = io_pos["j_acc_t"]
    omega_p = float(io_pos["omega"])
    omega_n = float(io_neg["omega"])
    C_pos = float(io_pos["C"])
    C_neg = float(io_neg["C"])
    M_dot_pos = C_pos * np.exp(-omega_p * times_io)
    M_dot_neg = C_neg * np.exp(-omega_n * times_io)

    frac_p = Fraction(omega_p).limit_denominator(10)
    frac_n = Fraction(omega_n).limit_denominator(10)
    lbl_p = rf"{frac_p.numerator}/{frac_p.denominator}"
    lbl_n = rf"{frac_n.numerator}/{frac_n.denominator}"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7, 3.2), dpi=300)

    ax_left.set_title("Assumptions", fontsize=9)
    ax_left.plot(times_io, M_dot_pos, "r--", lw=1.2,
                 label=rf"$\dot{{M}}_{{\rm acc}}$ ($\omega={lbl_p}$)")
    ax_left.plot(times_io, M_dot_neg, "b--", lw=1.2,
                 label=rf"$\dot{{M}}_{{\rm acc}}$ ($\omega={lbl_n}$)")
    ax_left.set_xlabel("Time (Gyr)", fontsize=8)
    ax_left.set_ylabel(r"$\dot{M}_{\rm acc}$ ($M_\odot\,{\rm Gyr}^{-1}$)", fontsize=8)
    ax_left.set_xlim(0, 12)
    ax_left.set_yscale("log")
    ax_left.tick_params(axis="both", labelsize=7)
    ax_left.grid(True, alpha=0.3)

    ax_m = ax_left.twinx()
    ax_m.plot(times_io, j_acc_t, "k-", lw=1.5, label=r"$j_{\rm acc}(t)$")
    ax_m.set_ylabel(r"$j_{\rm acc}$ (kpc km s$^{-1}$)", fontsize=8)
    ax_m.tick_params(axis="y", labelsize=7)

    l1, lab1 = ax_left.get_legend_handles_labels()
    l2, lab2 = ax_m.get_legend_handles_labels()
    ax_left.legend(l1 + l2, lab1 + lab2, loc="lower center", fontsize=7, framealpha=0.9)

    ax_right.set_title("Results", fontsize=9)
    ax_right.plot(times_io, io_pos["f_gas_t"], "b-", lw=1.5, label=rf"$f_{{\rm gas}}$ ($\omega={lbl_p}$)")
    ax_right.plot(times_io, io_neg["f_gas_t"], "r-", lw=1.5, label=rf"$f_{{\rm gas}}$ ($\omega={lbl_n}$)")
    ax_right.set_xlabel("Time (Gyr)", fontsize=8)
    ax_right.set_ylabel(r"$f_{\rm gas}$", fontsize=8)
    ax_right.set_xlim(0.1, 12)
    ax_right.set_ylim(0, 1.05)
    ax_right.tick_params(axis="both", labelsize=7)
    ax_right.grid(True, alpha=0.3)

    ax_jbar = ax_right.twinx()
    ax_jbar.plot(times_io, io_pos["j_bar_t"], "b--", lw=1.2, label=rf"$j_{{\rm bar}}$ ($\omega={lbl_p}$)")
    ax_jbar.plot(times_io, io_neg["j_bar_t"], "r--", lw=1.2, label=rf"$j_{{\rm bar}}$ ($\omega={lbl_n}$)")
    ax_jbar.set_ylabel(r"$j_{\rm bar}$ (kpc km s$^{-1}$)", fontsize=8)
    ax_jbar.tick_params(axis="y", labelsize=7)

    l1, lab1 = ax_right.get_legend_handles_labels()
    l2, lab2 = ax_jbar.get_legend_handles_labels()
    ax_right.legend(l1 + l2, lab1 + lab2, loc="lower right", fontsize=7, framealpha=0.9)

    fig.suptitle(rf"Inside-Out Model ($M_{{\rm bar}} = 10^{{{np.log10(float(io_pos['M_bar'])):.1f}}}"
                 rf"\,M_\odot$, $k={float(io_pos['k']):.2g}$, $n={float(io_pos['n']):.2g}$)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_nio_split(nio_high, nio_low):
    times = nio_high["times"]
    omega = float(nio_high["omega"])
    C_nio = float(nio_high["C"])
    j_high = float(nio_high["j_acc_value"])
    j_low = float(nio_low["j_acc_value"])
    M_dot_acc = C_nio * np.exp(-omega * times)

    c_high, c_low = "blue", "red"
    fig, axes = plt.subplots(2, 2, figsize=(10, 5), dpi=300, sharex=True)

    ax = axes[0, 0]
    ax.plot(times, M_dot_acc, "k-", lw=1.5)
    ax.set_ylabel(r"$\dot{M}_{\rm acc}$ ($M_\odot\,{\rm Gyr}^{-1}$)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Assumptions", fontsize=14)
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=12)
    ax.set_xlim(0, 12)

    ax = axes[1, 0]
    ax.axhline(j_high, color=c_high, ls="--", lw=1.5, label=r"$j_{\rm acc}=j_{\rm max}$")
    ax.axhline(j_low, color=c_low, ls="--", lw=1.5, label=r"$j_{\rm acc}=j_{\rm max}/10$")
    ax.set_ylabel(r"$j_{\rm acc}$ (kpc km s$^{-1}$)", fontsize=12)
    ax.set_xlabel("Time (Gyr)", fontsize=12)
    ax.set_ylim(0, j_high * 1.3)
    ax.legend(fontsize=10, framealpha=0.9, loc="center right")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=12)
    ax.set_xlim(0, 12)

    ax = axes[0, 1]
    ax.plot(times, nio_high["f_gas_t"], color=c_high, lw=1.5, label=r"high $j_{\rm acc}$")
    ax.plot(times, nio_low["f_gas_t"], color=c_low, lw=1.5, label=r"low $j_{\rm acc}$")
    ax.set_ylabel(r"$f_{\rm gas}$", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Results", fontsize=14)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper right")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=12)
    ax.set_xlim(0.1, 12)

    ax = axes[1, 1]
    ax.plot(times, nio_high["j_bar_t"], color=c_high, lw=1.5, label=r"high $j_{\rm acc}$")
    ax.plot(times, nio_low["j_bar_t"], color=c_low, lw=1.5, label=r"low $j_{\rm acc}$")
    ax.set_ylabel(r"$j_{\rm bar}$ (kpc km s$^{-1}$)", fontsize=12)
    ax.set_xlabel("Time (Gyr)", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9, loc="center right")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=12)
    ax.set_xlim(0.1, 12)

    fig.suptitle(rf"Non-Inside-Out Model ($M_{{\rm bar}}=10^{{{np.log10(float(nio_high['M_bar'])):.1f}}}"
                 rf"\,M_\odot$, $\omega_{{\rm acc}}={omega:.2g}\,{{\rm Gyr}}^{{-1}}$)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_io_split(io_pos, io_neg):
    times_io = io_pos["times"]
    j_acc_t = io_pos["j_acc_t"]
    omega_p = float(io_pos["omega"])
    omega_n = float(io_neg["omega"])
    C_pos = float(io_pos["C"])
    C_neg = float(io_neg["C"])
    M_dot_pos = C_pos * np.exp(-omega_p * times_io)
    M_dot_neg = C_neg * np.exp(-omega_n * times_io)

    frac_p = Fraction(omega_p).limit_denominator(10)
    frac_n = Fraction(omega_n).limit_denominator(10)
    lbl_p = rf"$\omega={frac_p.numerator}/{frac_p.denominator}$"
    lbl_n = rf"$\omega={frac_n.numerator}/{frac_n.denominator}$"
    c_pos, c_neg = "red", "blue"

    fig, axes = plt.subplots(2, 2, figsize=(10, 5), dpi=300, sharex=True)

    ax = axes[0, 0]
    ax.plot(times_io, M_dot_pos, color=c_pos, lw=1.5, label=lbl_p)
    ax.plot(times_io, M_dot_neg, color=c_neg, lw=1.5, label=lbl_n)
    ax.set_ylabel(r"$\dot{M}_{\rm acc}$ ($M_\odot\,{\rm Gyr}^{-1}$)", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Assumptions", fontsize=14)
    ax.legend(fontsize=10, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=12)
    ax.set_xlim(0, 12)

    ax = axes[1, 0]
    ax.plot(times_io, j_acc_t, "k-", lw=1.5, label=r"$j_{\rm acc}(t)$")
    ax.set_ylabel(r"$j_{\rm acc}$ (kpc km s$^{-1}$)", fontsize=12)
    ax.set_xlabel("Time (Gyr)", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=12)
    ax.set_xlim(0, 12)

    ax = axes[0, 1]
    ax.plot(times_io, io_pos["f_gas_t"], color=c_pos, lw=1.5, label=lbl_p)
    ax.plot(times_io, io_neg["f_gas_t"], color=c_neg, lw=1.5, label=lbl_n)
    ax.set_ylabel(r"$f_{\rm gas}$", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Results", fontsize=14)
    ax.legend(fontsize=10, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=12)
    ax.set_xlim(0.1, 12)

    ax = axes[1, 1]
    ax.plot(times_io, io_pos["j_bar_t"], color=c_pos, lw=1.5, label=lbl_p)
    ax.plot(times_io, io_neg["j_bar_t"], color=c_neg, lw=1.5, label=lbl_n)
    ax.set_ylabel(r"$j_{\rm bar}$ (kpc km s$^{-1}$)", fontsize=12)
    ax.set_xlabel("Time (Gyr)", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=12)
    ax.set_xlim(0.1, 12)

    fig.suptitle(rf"Inside-Out Model ($M_{{\rm bar}}=10^{{{np.log10(float(io_pos['M_bar'])):.1f}}}"
                 rf"\,M_\odot$, $k={float(io_pos['k']):.2g}$, $n={float(io_pos['n']):.2g}$)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_combined(nio_high, nio_low, io_pos, io_neg):
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), dpi=300)

    times_nio = nio_high["times"]
    omega_nio = float(nio_high["omega"])
    C_nio = float(nio_high["C"])
    j_high = float(nio_high["j_acc_value"])
    j_low = float(nio_low["j_acc_value"])
    M_dot_nio = C_nio * np.exp(-omega_nio * times_nio)

    ax = axes[0, 0]
    ax.plot(times_nio, M_dot_nio, "k-", lw=1.5, label=r"$\dot{M}_{\rm acc}(t)$")
    ax.set_ylabel(r"$\dot{M}_{\rm acc}$ ($M_\odot\,{\rm Gyr}^{-1}$)", fontsize=9)
    ax.set_xlim(0, 12)                                  # linear: omega=0.1 spans <1 decade, log ticks look odd
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.44, 0.95, "NIO", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    ax_j = ax.twinx()
    ax_j.axhline(j_high, color="blue", ls="--", lw=1.5, label=r"$j_{\rm acc} = j_{\rm max}$")
    ax_j.axhline(j_low, color="red", ls="--", lw=1.5, label=r"$j_{\rm acc} = j_{\rm max}/10$")
    ax_j.set_ylabel(r"$j_{\rm acc}$ (kpc km s$^{-1}$)", fontsize=9)
    ax_j.set_ylim(0, j_high * 1.3)
    ax_j.tick_params(axis="y", labelsize=8)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax_j.get_legend_handles_labels()
    ax_j.legend(l1 + l2, lab1 + lab2, loc="center left", fontsize=6.5, framealpha=1.0)

    ax = axes[0, 1]
    ax.plot(times_nio, nio_high["f_gas_t"], "b-", lw=1.5, label=r"$f_{\rm gas}$ (high $j_{\rm acc}$)")
    ax.plot(times_nio, nio_low["f_gas_t"], "r-", lw=1.5, label=r"$f_{\rm gas}$ (low $j_{\rm acc}$)")
    ax.set_ylabel(r"$f_{\rm gas}$", fontsize=9)
    ax.set_xlim(0.1, 12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)

    ax_jbar = ax.twinx()
    ax_jbar.plot(times_nio, nio_high["j_bar_t"], "b--", lw=1.2, label=r"$j_{\rm bar}$ (high $j_{\rm acc}$)")
    ax_jbar.plot(times_nio, nio_low["j_bar_t"], "r--", lw=1.2, label=r"$j_{\rm bar}$ (low $j_{\rm acc}$)")
    ax_jbar.set_ylabel(r"$j_{\rm bar}$ (kpc km s$^{-1}$)", fontsize=9)
    ax_jbar.tick_params(axis="y", labelsize=8)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax_jbar.get_legend_handles_labels()
    ax_jbar.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=6.5, framealpha=1.0)

    times_io = io_pos["times"]
    j_acc_t = io_pos["j_acc_t"]
    omega_p = float(io_pos["omega"])
    omega_n = float(io_neg["omega"])
    C_p = float(io_pos["C"])
    C_n = float(io_neg["C"])
    M_dot_pos = C_p * np.exp(-omega_p * times_io)
    M_dot_neg = C_n * np.exp(-omega_n * times_io)

    frac_p = Fraction(omega_p).limit_denominator(10)
    frac_n = Fraction(omega_n).limit_denominator(10)
    lbl_p = rf"{frac_p.numerator}/{frac_p.denominator}"
    lbl_n = rf"{frac_n.numerator}/{frac_n.denominator}"

    ax = axes[1, 0]
    ax.plot(times_io, M_dot_pos, "r--", lw=1.2,
            label=rf"$\dot{{M}}_{{\rm acc}}$ ($\omega={lbl_p}$ Gyr$^{{-1}}$)")
    ax.plot(times_io, M_dot_neg, "b--", lw=1.2,
            label=rf"$\dot{{M}}_{{\rm acc}}$ ($\omega={lbl_n}$ Gyr$^{{-1}}$)")
    ax.set_xlabel("Time (Gyr)", fontsize=8)
    ax.set_ylabel(r"$\dot{M}_{\rm acc}$ ($M_\odot\,{\rm Gyr}^{-1}$)", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlim(0, 12)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.47, 0.95, "IO", transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    ax_m = ax.twinx()
    ax_m.plot(times_io, j_acc_t, "k-", lw=1.5, label=r"$j_{\rm acc}(t)$")
    ax_m.set_ylabel(r"$j_{\rm acc}$ (kpc km s$^{-1}$)", fontsize=9)
    ax_m.set_ylim(0, float(np.max(j_acc_t)) * 1.8)      # headroom so j_acc doesn't overlap M_dot(-omega)
    ax_m.tick_params(axis="y", labelsize=8)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax_m.get_legend_handles_labels()
    ax_m.legend(l1 + l2, lab1 + lab2, loc="center left", fontsize=6.5, framealpha=1.0)

    ax = axes[1, 1]
    ax.plot(times_io, io_pos["f_gas_t"], "r-", lw=1.5,
            label=rf"$f_{{\rm gas}}$ ($\omega={lbl_p}$ Gyr$^{{-1}}$)")
    ax.plot(times_io, io_neg["f_gas_t"], "b-", lw=1.5,
            label=rf"$f_{{\rm gas}}$ ($\omega={lbl_n}$ Gyr$^{{-1}}$)")
    ax.set_xlabel("Time (Gyr)", fontsize=8)
    ax.set_ylabel(r"$f_{\rm gas}$", fontsize=9)
    ax.set_xlim(0.1, 12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)

    ax_jbar = ax.twinx()
    ax_jbar.plot(times_io, io_pos["j_bar_t"], "r--", lw=1.2,
                 label=rf"$j_{{\rm bar}}$ ($\omega={lbl_p}$ Gyr$^{{-1}}$)")
    ax_jbar.plot(times_io, io_neg["j_bar_t"], "b--", lw=1.2,
                 label=rf"$j_{{\rm bar}}$ ($\omega={lbl_n}$ Gyr$^{{-1}}$)")
    ax_jbar.set_ylabel(r"$j_{\rm bar}$ (kpc km s$^{-1}$)", fontsize=9)
    ax_jbar.tick_params(axis="y", labelsize=8)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax_jbar.get_legend_handles_labels()
    ax_jbar.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=6.5, framealpha=1.0)

    axes[0, 0].set_title("Assumptions", fontsize=9)
    axes[0, 1].set_title("Results", fontsize=9)
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", labelbottom=False)
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis="x", labelbottom=False)

    plt.tight_layout(h_pad=0.4)
    return fig


# ── orchestration ────────────────────────────────────────────────────────────

def _paths(config):
    cfg = load_config(config)
    out = ROOT / cfg["paths"]["outputs"]
    data = ROOT / cfg["paths"]["data"]
    return {
        "data": data,
        "io_prof": out / "model_radial_profiles" / "io",
        "nio_prof": out / "model_radial_profiles" / "nio",
        "hi": data / "allHIprofs_mod",
        "sfr": data / "allSFRprofs",
        "stars": data / "allSTARSprofs",
        "grids": data / "data9_JAX_aKSL",
        "nio_band": out / "jbar_fgas_profiles" / "nio" / "band",
        "nio_prof_csv": out / "jbar_fgas_profiles" / "nio",
        "comp_out": out / "radial_profile_comparisons",
        "mosaic_out": out / "jbar_fgas_profiles",
    }


def _save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_profiles] wrote {path}")


def run_profiles(galaxies, paths):
    out = paths["comp_out"]
    out.mkdir(parents=True, exist_ok=True)
    for gal in galaxies:
        p = _load_galaxy(gal, paths)
        if p["io"] is None and p["nio"] is None:
            print(f"[compare_profiles] skip {gal}: no IO/NIO profile npz")
            continue
        _save(plot_galaxy_comparison(gal, p), out / f"{gal}_profile_comparison.png")


def run_evolution(galaxies, paths):
    out = paths["comp_out"]
    out.mkdir(parents=True, exist_ok=True)
    for gal in galaxies:
        p = _load_galaxy(gal, paths)
        if p["io"] is None:
            print(f"[compare_profiles] skip {gal}: no IO profile npz (time-evolution needs it)")
            continue
        fig = plot_time_evolution(gal, p, model_type="io")
        if fig is not None:
            _save(fig, out / f"{gal}_time_evolution_io.png")
        fig = plot_time_evolution(gal, p, model_type="nio")
        if fig is not None:
            _save(fig, out / f"{gal}_time_evolution_nio.png")


def run_mosaic(bins, paths, n, k, a, b, sfl, n_j):
    out = paths["mosaic_out"]
    out.mkdir(parents=True, exist_ok=True)
    fig = plot_mosaic(bins, paths, n, k, a, b, sfl, n_j)
    fig.savefig(out / "jbar_vs_fgas_mosaic_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare_profiles] wrote {out / 'jbar_vs_fgas_mosaic_comparison.pdf'}")


def run_comparison(paths, n, k, a, b, sfl, logM=10.0):
    out = paths["comp_out"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"[compare_profiles] comparison at logM={logM:.1f}: IO (n={n}, k={k}), "
          f"NIO power-law (a={a}, b={b})")
    io_pos, io_neg = _comparison_io_live(logM, n, k, sfl)
    nio_high, nio_low = _comparison_nio_live(logM, a, b, sfl)
    _save(plot_nio_2panel(nio_high, nio_low),
          out / "comparison_nio_assumptions_vs_results.pdf", dpi=300)
    _save(plot_io_2panel(io_pos, io_neg),
          out / "comparison_io_assumptions_vs_results.pdf", dpi=300)
    _save(plot_nio_split(nio_high, nio_low), out / "comparison_nio_split_panels.pdf", dpi=300)
    _save(plot_io_split(io_pos, io_neg), out / "comparison_io_split_panels.pdf", dpi=300)
    _save(plot_combined(nio_high, nio_low, io_pos, io_neg),
          out / "models_illustration.pdf", dpi=300)


def _parse_bins(spec):
    if spec is None:
        return DEFAULT_BINS
    bins = []
    for tok in spec:
        lo, hi = (float(x) for x in tok.split(","))
        bins.append((lo, hi))
    return bins


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figure", nargs="?", default="all",
                   choices=["profiles", "evolution", "mosaic", "comparison", "all"])
    p.add_argument("--galaxies", nargs="+", default=DEFAULT_GALAXIES)
    p.add_argument("--bins", nargs="+", default=None,
                   help="mass bins as logM_low,logM_high pairs (e.g. --bins 9.0,9.5 9.7,10.3)")
    p.add_argument("--io-params", type=float, nargs=2, default=None, metavar=("n", "k"),
                   help="IO (n, k) for the mosaic/comparison (run live)")
    p.add_argument("--nio-params", type=float, nargs=2, default=None, metavar=("a", "b"),
                   help="NIO power-law (a, b) for the mosaic/comparison: omega = a*(Mbar/1e10)**b")
    p.add_argument("--sfl", default=None, help="star-formation law (default: config sfl.default)")
    p.add_argument("--n-j", type=int, default=None, help="NIO j_acc sweep points (default: config)")
    p.add_argument("--comparison-logM", type=float, default=10.0,
                   help="single-galaxy mass for the comparison figures")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = _paths(args.config)
    bins = _parse_bins(args.bins)
    sfl = args.sfl or cfg["sfl"]["default"]
    n_j = args.n_j or cfg["integration"]["n_j"]

    if args.figure in ("profiles", "all"):
        run_profiles(args.galaxies, paths)
    if args.figure in ("evolution", "all"):
        run_evolution(args.galaxies, paths)
    if args.figure in ("mosaic", "all"):
        if args.io_params is None or args.nio_params is None:
            raise SystemExit("mosaic needs --io-params n k and --nio-params a b")
        run_mosaic(bins, paths, args.io_params[0], args.io_params[1],
                   args.nio_params[0], args.nio_params[1], sfl, n_j)
    if args.figure in ("comparison", "all"):
        if args.io_params is None or args.nio_params is None:
            raise SystemExit("comparison needs --io-params n k and --nio-params a b")
        run_comparison(paths, args.io_params[0], args.io_params[1],
                       args.nio_params[0], args.nio_params[1], sfl, args.comparison_logM)


if __name__ == "__main__":
    sys.exit(main())
