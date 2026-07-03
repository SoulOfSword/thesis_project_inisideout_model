"""omega_acc vs M_bar diagnostics.

--model io  : per-galaxy omega from inverting j_bar at the IO (n, k), binned per mass.
--model nio : per-mass-bin best-fit omega from the NIO 4obs scan (engine; run on a node).
--model both: binned IO omega (inverted j_bar, full sample) + the NIO power-law line,
              for hand-picked (--io-params n k) and (--nio-params a b). Light.

The io/nio modes overlay the NIO law omega = a(logM - 10) + b for the grid and/or MCMC (a, b).
The io run also prints the omega spread, which sets the accretion-rate grid for the planes.
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
from jmfgas.inference.build import obs_table
from jmfgas.inference.omega import omega_per_galaxy

_MASS_BINS = np.arange(8, 12.5, 0.5)            # covers the full sample (up to logM ~ 11.8)
_NIO_COLS = ("logMbar", "jbar", "Mgas", "e_Mgas", "Mstar", "e_Mstar",
             "jgas", "e_jgas", "jstar", "e_jstar")


def _init_worker():
    import jax
    jax.config.update("jax_enable_x64", True)


def _scan_bin(task):
    """logL vs omega for one mass bin (NIO, a=0, b=omega)."""
    obs_bin, omega_grid, sfl = task
    from jmfgas.inference.likelihoods import logL_4obs_all_galaxies
    return np.array([logL_4obs_all_galaxies(0.0, float(om), *obs_bin, sfl) for om in omega_grid])


def _nio_lines(args):
    """[(label, a, b, linestyle)] for omega = a(logM-10)+b from the grid and/or MCMC."""
    out = []
    if args.nio_source in ("grid", "both") and args.nio_grid.exists():
        a, b = np.load(args.nio_grid, allow_pickle=True)["peak"]
        out.append(("grid", float(a), float(b), "-"))
    if args.nio_source in ("mcmc", "both") and args.nio_chain.exists():
        import emcee
        ch = emcee.backends.HDFBackend(str(args.nio_chain), read_only=True).get_chain(
            discard=args.burn_in, flat=True)
        a, b = np.median(ch, axis=0)
        out.append(("MCMC", float(a), float(b), "--"))
    return out


def _draw_nio(ax, lines):
    x = np.linspace(_MASS_BINS[0], _MASS_BINS[-1], 50)
    for label, a, b, ls in lines:
        ax.plot(x, a * (x - 10.0) + b, color="r", ls=ls, lw=2,
                label=rf"NIO {label}: $\omega={a:.2f}(\log M-10)+{b:.2f}$")


def io_diagnostic(args, data_dir, lines):
    n, k = args.io_params or (float(v) for v in np.load(args.io_grid, allow_pickle=True)["peak"])
    n, k = float(n), float(k)
    t = obs_table(args.sample, data_dir)
    logM = t["logMbar"]
    omega_raw, _, _ = omega_per_galaxy(logM, t["jbar"], n, k)
    omega = np.clip(omega_raw, None, 10.0)                      # the model's omega ceiling
    cen = 0.5 * (_MASS_BINS[:-1] + _MASS_BINS[1:])
    med = np.full(len(cen), np.nan); lo = med.copy(); hi = med.copy()
    for i in range(len(cen)):
        v = omega[(logM >= _MASS_BINS[i]) & (logM < _MASS_BINS[i + 1])]
        if len(v):
            med[i], lo[i], hi[i] = np.percentile(v, [50, 16, 84])
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200, facecolor="w")
    ax.errorbar(cen, med, xerr=0.25, yerr=[med - lo, hi - med], fmt="-o", color="b",
                capsize=2, alpha=0.8, label=rf"IO binned ($n={n:.2f}$, $k={k:.2f}$)")
    _draw_nio(ax, lines)
    print(f"IO (n={n:.3f}, k={k:.3f}): omega over {len(logM)} galaxies "
          f"[{omega_raw.min():.2f}, {omega_raw.max():.2f}]  (1/99 pct "
          f"[{np.percentile(omega_raw, 1):.2f}, {np.percentile(omega_raw, 99):.2f}])")
    return fig


def nio_diagnostic(args, data_dir, lines):
    cen, best, lo, hi = _nio_scan(args, data_dir)
    v = np.isfinite(best)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200, facecolor="w")
    ax.errorbar(cen[v], best[v], xerr=0.25, yerr=[best[v] - lo[v], hi[v] - best[v]],
                fmt="o", color="steelblue", capsize=3, alpha=0.85, label="NIO per-bin fit")
    _draw_nio(ax, lines)
    return fig


def _io_binned(logM, jbar, n, k):
    """Per-galaxy IO omega (inverted j_bar, clipped to +-10) -> (cen, median, 16, 84) per bin."""
    omega = np.clip(omega_per_galaxy(logM, jbar, n, k)[0], -10.0, 10.0)
    cen = 0.5 * (_MASS_BINS[:-1] + _MASS_BINS[1:])
    med = np.full(len(cen), np.nan); lo = med.copy(); hi = med.copy()
    for i in range(len(cen)):
        v = omega[(logM >= _MASS_BINS[i]) & (logM < _MASS_BINS[i + 1])]
        if len(v):
            med[i], lo[i], hi[i] = np.percentile(v, [50, 16, 84])
    return cen, med, lo, hi


def _nio_scan(args, data_dir):
    """NIO per-mass-bin best-fit omega (a=0) from the 4obs scan -> (cen, best, lo, hi).
    Heavy: loky workers + the engine, so run on a node."""
    from loky import get_reusable_executor
    t = obs_table(args.sample, data_dir)
    logM = t["logMbar"]
    omega_grid = np.linspace(args.omega_min, args.omega_max, args.scan_n)
    n_bins = len(_MASS_BINS) - 1
    masks = [(logM >= _MASS_BINS[i]) & (logM < _MASS_BINS[i + 1]) for i in range(n_bins)]
    tasks = [(tuple(np.asarray(t[c])[m] for c in _NIO_COLS), omega_grid, "cutoff_ksl")
             for m in masks]
    ex = get_reusable_executor(max_workers=min(n_bins, args.max_workers), initializer=_init_worker)
    rows = list(ex.map(_scan_bin, tasks))
    ex.shutdown(wait=True, kill_workers=True)
    cen = 0.5 * (_MASS_BINS[:-1] + _MASS_BINS[1:])
    best = np.full(n_bins, np.nan); lo = best.copy(); hi = best.copy()
    for i, row in enumerate(rows):
        if masks[i].sum() == 0 or not np.isfinite(row).any():
            continue
        jm = int(np.nanargmax(row)); best[i] = omega_grid[jm]
        thr = row[jm] - 0.5                                    # 1 sigma from delta logL = 0.5
        l = jm
        while l > 0 and row[l] >= thr:
            l -= 1
        r = jm
        while r < len(omega_grid) - 1 and row[r] >= thr:
            r += 1
        lo[i], hi[i] = omega_grid[l], omega_grid[r]
        print(f"  logM [{_MASS_BINS[i]:.1f},{_MASS_BINS[i+1]:.1f})  N={int(masks[i].sum()):2d}  "
              f"omega={best[i]:+.2f}  [{lo[i]:+.2f}, {hi[i]:+.2f}]")
    return cen, best, lo, hi


def both_diagnostic(args, data_dir):
    """Binned IO omega (inverted j_bar, full sample) + the NIO analytical power-law line."""
    import jax.numpy as jnp
    from jmfgas.models.non_inside_out import omega_Mdep
    if args.io_params is None or args.nio_params is None:
        raise SystemExit("--model both needs --io-params n k and --nio-params a b")
    n, k = (float(v) for v in args.io_params)
    a, b = (float(v) for v in args.nio_params)
    t = obs_table(args.sample, data_dir)
    io_cen, io_med, io_lo, io_hi = _io_binned(np.asarray(t["logMbar"]), np.asarray(t["jbar"]), n, k)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300, facecolor="w")
    vi = np.isfinite(io_med)
    ax.errorbar(io_cen[vi], io_med[vi], xerr=0.25,
                yerr=[io_med[vi] - io_lo[vi], io_hi[vi] - io_med[vi]], fmt="o",
                capsize=2, alpha=0.85, color="royalblue", label=rf"IO binned ($n={n:.2f}$, $k={k:.2f}$)")
    xl = np.linspace(_MASS_BINS[0], _MASS_BINS[-1], 100)
    ax.plot(xl, np.asarray(omega_Mdep(jnp.asarray(xl), a, b)), color="r", lw=2.5, zorder=5,
            label=rf"NIO law: $\omega={a:g}\,(M_{{\rm bar}}/10^{{10}})^{{{b:g}}}$")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["io", "nio", "both"], required=True)
    p.add_argument("--sample", default="mcmc-obs")
    p.add_argument("--io-grid", type=Path, default=ROOT / "outputs/grids/grid_io_4obs_grid-obs.npz")
    p.add_argument("--io-params", type=float, nargs=2, default=None, metavar=("n", "k"))
    p.add_argument("--nio-params", type=float, nargs=2, default=None, metavar=("a", "b"),
                   help="NIO power-law omega = a*(Mbar/1e10)**b, for --model both")
    p.add_argument("--nio-grid", type=Path, default=ROOT / "outputs/grids/grid_nio_4obs_grid-obs.npz")
    p.add_argument("--nio-chain", type=Path,
                   default=ROOT / "outputs/mcmc_chains/chain_nio_4obs_mcmc-obs_20260606.h5")
    p.add_argument("--nio-source", choices=["grid", "mcmc", "both"], default="both")
    p.add_argument("--burn-in", type=int, default=50)
    p.add_argument("--omega-min", type=float, default=-5.0)
    p.add_argument("--omega-max", type=float, default=15.0)
    p.add_argument("--scan-n", type=int, default=200)
    p.add_argument("--max-workers", type=int, default=7)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    data_dir = ROOT / cfg["paths"]["data"]
    if args.model == "both":
        fig = both_diagnostic(args, data_dir)
    else:
        lines = _nio_lines(args)
        fig = io_diagnostic(args, data_dir, lines) if args.model == "io" \
            else nio_diagnostic(args, data_dir, lines)

    ax = fig.axes[0]
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel(r"$\log(M_{\rm bar}\,/\,\rm M_\odot)$", fontsize=14)
    ax.set_ylabel(r"$\omega_{\rm acc}$ (Gyr$^{-1}$)", fontsize=14)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3); fig.tight_layout()
    out = args.out or (ROOT / cfg["paths"]["figures"] / f"omega_vs_mass_{args.model}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
