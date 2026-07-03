"""Per-galaxy residuals (chi = (model - obs)/sigma) for the IO and NIO models, side by side.

For each galaxy the observed j_bar fixes the model's free accretion parameter (the same
path the likelihood / comparison figures use), the model is run, and chi is plotted against
log M_bar, coloured by observed f_gas. One column per model:

  --io-params  n k : inside-out column. j_bar is inverted to its accretion rate omega at
                     (n, k); the model is run and chi computed for each quantity.
  --nio-params a b : non-inside-out column. omega(M_bar)=a*(Mbar/1e10)^b is fixed by mass and
                     j_acc is set to the observed j_bar; the model is run.

The quantity set is chosen on the CLI:
  fgas : f_gas and j_bar          (the f_gas-likelihood observables; any sample)
  all  : f_gas, j_bar, M_gas, M_star, j_gas, j_star   (needs the decomposed -> converged sample)

The inside-out r_acc grid is extended (residual-plots only) out to the sample's heaviest
galaxy, so the massive end isn't snapped to the 11.5 edge bin (which biased its forward
j_bar low). NIO uses the exact M_bar, so it has no such snap. Saves to outputs/residuals/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jmfgas.config import load_config
from jmfgas.data import sample_frame
from jmfgas.viz.planes import _COMPILATION_MARKERS

# both engines return observables in this order (f_gas, j_bar, j_gas, j_star, M_star, M_gas)
_MODEL_IDX = {"fgas": 0, "jbar": 1, "jgas": 2, "jstar": 3, "Mstar": 4, "Mgas": 5}
_LABEL = {"fgas": r"$f_{\rm gas}$", "jbar": r"$j_{\rm bar}$", "Mgas": r"$M_{\rm gas}$",
          "Mstar": r"$M_\star$", "jgas": r"$j_{\rm gas}$", "jstar": r"$j_\star$"}
_SETS = {"fgas": ["fgas", "jbar"],
         "all": ["fgas", "jbar", "Mgas", "Mstar", "jgas", "jstar"]}
_MARKER = {"MP+21b": "o", **_COMPILATION_MARKERS}     # markers match the final plane plots


def load_sample(sample, data_dir):
    """Observable arrays + each galaxy's sample group, from the built sample frame."""
    df = sample_frame(sample, data_dir)
    obs = {c: df[c].to_numpy(float) for c in df.columns if c not in ("Name", "group")}
    if "logMbar" not in obs and "Mbar" in obs:
        obs["logMbar"] = np.log10(obs["Mbar"])
    group = (df["group"].to_numpy() if "group" in df.columns
             else np.full(len(df), "MP+21b"))
    return obs, group


def extended_mass_grid(logM_sample, mass_grid_cfg, margin=0.05):
    """The config mass grid extended (same spacing) to cover the sample's heaviest galaxy.

    Residual-plots only: removes the nearest-bin r_acc snap at the massive end (which biases
    the IO forward j_bar low for the superspirals). The model's default grid is untouched."""
    lo, hi_cfg, n = mass_grid_cfg["logM_min"], mass_grid_cfg["logM_max"], mass_grid_cfg["n"]
    dlog = (hi_cfg - lo) / (n - 1)
    hi = max(hi_cfg, float(np.max(logM_sample)) + margin)
    n_ext = int(np.ceil((hi - lo) / dlog)) + 1
    return np.linspace(lo, hi, n_ext)


def compute_chi_io(n, k, obs, quantities, t0, logM_grid, sfl, rmax):
    """IO chi per quantity: invert j_bar -> omega at (n, k), run the model, compare.
    Returns (chi dict, count of omega-clipped galaxies). Uses the given (extended) mass grid
    and integration radius rmax."""
    import jax.numpy as jnp
    from jmfgas.physics.angmom import j_maxer
    from jmfgas.models.inside_out import (solve_omega_bisect_autobracket_jax,
                                          all_obs_for_galaxies_jax,
                                          build_r_acc_matrix_for_all_M_jax)

    logM = jnp.asarray(obs["logMbar"], dtype=jnp.float64)
    jbar = jnp.asarray(obs["jbar"], dtype=jnp.float64)
    Mbar = 10.0 ** logM
    grid = jnp.asarray(logM_grid, dtype=jnp.float64)
    r_acc = build_r_acc_matrix_for_all_M_jax(jnp.float64(n), jnp.float64(k), grid)

    j_max = j_maxer(Mbar)
    j_min = j_max / 10.0
    delta_j = jnp.maximum(k * j_max - j_min, 1e-12)
    y_raw = (jbar - j_min) / delta_j
    omega, ok = solve_omega_bisect_autobracket_jax(y_raw, n, t0)
    omega_assigned = jnp.where(ok, omega, jnp.where(y_raw < 0.0, 10.0, -10.0))
    was_clipped = (~ok) | (omega_assigned > 10.0) | (omega_assigned < -10.0)
    omega_clipped = jnp.clip(omega_assigned, -10.0, 10.0)
    omega_safe = jnp.where(jnp.abs(omega_clipped) < 1e-4,
                           jnp.sign(omega_clipped) * 1e-4, omega_clipped)
    t_acc = 1.0 / omega_safe

    mod = all_obs_for_galaxies_jax(Mbar, t_acc, sfl, r_acc, grid, Rmax=rmax, at_t0=True)
    mod = [np.asarray(m) for m in mod]
    chi = {q: (mod[_MODEL_IDX[q]] - obs[q]) / obs["e_" + q] for q in quantities}
    return chi, int(np.sum(np.asarray(was_clipped)))


def compute_chi_nio(a, b, obs, quantities, t0, sfl, rmax):
    """NIO chi per quantity: omega(M_bar)=a*(Mbar/1e10)^b fixed by mass, j_acc set to the
    observed j_bar, run the model, compare. Returns (chi dict, 0 clipped)."""
    import jax.numpy as jnp
    from jmfgas.physics.radius import r_btfr_def
    from jmfgas.models.non_inside_out import run_all_masses_Mdep_omega_jax

    logM = np.asarray(obs["logMbar"], float)
    jbar = np.asarray(obs["jbar"], float)
    Mbar = 10.0 ** logM
    r_acc = np.array([[float(r_btfr_def(mb, jb))] for mb, jb in zip(Mbar, jbar)])  # j_acc = obs j_bar
    out = run_all_masses_Mdep_omega_jax(
        jnp.asarray(logM, dtype=jnp.float64), float(a), float(b),
        jnp.asarray(r_acc, dtype=jnp.float64), star_formation_law=sfl, Rmax=rmax)
    mod = [np.asarray(o)[:, 0] for o in out]
    chi = {q: (mod[_MODEL_IDX[q]] - obs[q]) / obs["e_" + q] for q in quantities}
    return chi, 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("quantities", choices=["fgas", "all"],
                   help="fgas: f_gas+j_bar; all: + the four decomposed observables")
    p.add_argument("--io-params", type=float, nargs=2, metavar=("N", "K"), default=None,
                   help="inside-out (n, k) -> a column")
    p.add_argument("--nio-params", type=float, nargs=2, metavar=("A", "B"), default=None,
                   help="non-inside-out (a, b) for omega=a*(Mbar/1e10)^b -> a column")
    p.add_argument("--sample", default=None,
                   help="sample name (default: full for fgas, converged for all)")
    p.add_argument("--sfl", default=None, help="star-formation law (default: config sfl.default)")
    p.add_argument("--no-extend-racc", action="store_true",
                   help="keep the IO r_acc mass grid at the config cap (don't extend to the sample)")
    p.add_argument("--rmax", type=float, default=300.0,
                   help="integration radius [kpc] for the forward model; the default 100.1 kpc "
                        "truncates the largest disks (superspirals/HIX/GLSB), biasing their j_bar low")
    p.add_argument("--out", type=Path, default=None, help="output directory")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--t0", type=float, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    data_dir = ROOT / cfg["paths"]["data"]
    out_dir = args.out or (ROOT / cfg["paths"]["outputs"] / "residuals")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = args.t0 if args.t0 is not None else cfg["time"]["t0"]
    sfl = args.sfl or cfg["sfl"]["default"]
    quantities = _SETS[args.quantities]

    # columns: inside-out then non-inside-out (whichever params were given)
    columns = []                                    # (model, p0, p1, tag)
    if args.io_params is not None:
        columns.append(("io", *args.io_params, f"io_n{args.io_params[0]:.2f}_k{args.io_params[1]:.2f}"))
    if args.nio_params is not None:
        columns.append(("nio", *args.nio_params, f"nio_a{args.nio_params[0]:.2f}_b{args.nio_params[1]:.2f}"))
    if not columns:
        raise SystemExit("pass --io-params N K and/or --nio-params A B")

    sample = args.sample or ("converged" if args.quantities == "all" else "full")
    obs, group = load_sample(sample, data_dir)
    missing = [q for q in quantities if q not in obs or ("e_" + q) not in obs]
    if missing:
        raise SystemExit(f"sample {sample!r} lacks columns for {missing}; "
                         "use --sample converged for 'all' (it carries M_gas/M_star/j_gas/j_star)")

    # extended IO r_acc grid (residual-plots only); NIO uses exact M_bar so needs no grid
    if args.no_extend_racc:
        logM_grid = np.linspace(cfg["mass_grid"]["logM_min"], cfg["mass_grid"]["logM_max"],
                                cfg["mass_grid"]["n"])
    else:
        logM_grid = extended_mass_grid(obs["logMbar"], cfg["mass_grid"])
        print(f"[residuals] IO r_acc grid extended to logM={logM_grid[-1]:.2f} "
              f"({len(logM_grid)} bins; sample max {obs['logMbar'].max():.2f})")

    logM, fgas = obs["logMbar"], obs["fgas"]
    N = len(logM)
    groups = [g for g in _MARKER if g in set(group)]       # present groups, in marker-map order
    sm = plt.cm.ScalarMappable(cmap="jet_r", norm=plt.Normalize(0.0, 1.0)); sm.set_array([])
    nq, nm = len(quantities), len(columns)
    fig, axes = plt.subplots(nq, nm, figsize=(7.5 * nm, 3.3 * nq), dpi=150, squeeze=False)

    for col, (model, p0, p1, tag) in enumerate(columns):
        if model == "io":
            chi, nclip = compute_chi_io(p0, p1, obs, quantities, t0, logM_grid, sfl, args.rmax)
            header = f"IO  n={p0:.2f}, k={p1:.2f}  (clipped: {nclip}/{N})"
        else:
            chi, nclip = compute_chi_nio(p0, p1, obs, quantities, t0, sfl, args.rmax)
            header = rf"NIO  $\omega={p0:g}(M/10^{{10}})^{{{p1:g}}}$,  $j_{{\rm acc}}=j_{{\rm bar,obs}}$"
        print(f"[{tag}] clipped={nclip}/{N}")
        for row, q in enumerate(quantities):
            ax = axes[row][col]
            c = chi[q]
            valid = np.isfinite(c)
            chi2 = float(np.nansum(c[valid] ** 2))
            print(f"    {q:5s}  chi2={chi2:8.1f}  chi2/N={chi2 / N:6.2f}")
            for g in groups:                               # one scatter per group -> plane marker
                m = valid & (group == g)
                if m.any():
                    ax.scatter(logM[m], c[m], c=fgas[m], cmap="jet_r", marker=_MARKER[g],
                               s=50, alpha=0.7, edgecolors="k", linewidths=0.5,
                               vmin=0.0, vmax=1.0)
            ax.axhline(0, color="k", ls="--", lw=1)
            ax.set_ylabel(r"$({\rm mod}-{\rm obs})/\sigma$", fontsize=12)
            ax.set_title(_LABEL[q], fontsize=12)
            ax.grid(alpha=0.3)
            fig.colorbar(sm, ax=ax, label=r"$f_{\rm gas,obs}$")
            ax.text(0.05, 0.95, rf"$\chi^2={chi2:.0f}$", transform=ax.transAxes,
                    ha="left", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="gray", alpha=0.8))
        axes[0][col].text(0.5, 1.20, header, transform=axes[0][col].transAxes, ha="center",
                          fontsize=13, fontweight="bold")

    handles = [plt.Line2D([0], [0], marker=_MARKER[g], color="w", markerfacecolor="none",
                          markeredgecolor="k", markersize=9, label=g) for g in groups]
    axes[-1][0].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\log(M_{\rm bar}/M_\odot)$", fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    name = f"residuals_{args.quantities}_{'__vs__'.join(t for *_, t in columns)}_{sample}.pdf"
    fig.savefig(out_dir / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
