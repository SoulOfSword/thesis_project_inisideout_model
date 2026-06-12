"""Accretion-frequency (omega) histogram for the inside-out model at a given (n, k).

Inverts each observed galaxy's j_bar for its omega, prints solve statistics and the
per-mass-bin medians, and draws the omega distribution (plain + stacked by mass bin).

  python scripts/plots/omega_histogram.py --grid outputs/grids/grid_io_4obs_grid-obs.npz
  python scripts/plots/omega_histogram.py --params 0.51 2.17
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

_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--grid", type=Path, help="grid .npz; uses its peak (n, k)")
    src.add_argument("--params", type=float, nargs=2, metavar=("n", "k"))
    p.add_argument("--sample", default="mcmc-obs")
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    t0 = args.t0 if args.t0 is not None else cfg["time"]["t0"]
    data_dir = ROOT / cfg["paths"]["data"]
    if args.grid is not None:
        n, k = (float(v) for v in np.load(args.grid, allow_pickle=True)["peak"])
    else:
        n, k = args.params

    t = obs_table(args.sample, data_dir)
    logM, jbar = t["logMbar"], t["jbar"]
    omega, ok, y = omega_per_galaxy(logM, jbar, n, k, t0=t0)

    unsolved = ~ok
    if unsolved.any():
        print("Unsolved galaxies (pinned to omega caps):")
        for idx in np.where(unsolved)[0]:
            print(f"  galaxy {idx}: y={y[idx]:.4f}, logMbar={logM[idx]:.2f}, omega={omega[idx]:.1f}")
    mask = ok & np.isfinite(omega)
    print(f"\n(n, k)               : ({n:.4f}, {k:.4f})")
    print(f"N galaxies           : {len(y)}")
    print(f"Solved               : {mask.sum()}  ({mask.mean():.3f})")
    print(f"Unsolved             : {unsolved.sum()}  ({unsolved.mean():.3f})")
    if mask.any():
        print(f"omega range (solved) : [{omega[mask].min():.4g}, {omega[mask].max():.4g}]")
        print(f"omega < 0            : {(omega[mask] < 0).sum()}  ({(omega[mask] < 0).mean():.3f})")
        print(f"omega > 10           : {(omega[mask] > 10).sum()}  ({(omega[mask] > 10).mean():.3f})")
    print(f"y range              : [{np.nanmin(y):.4g}, {np.nanmax(y):.4g}]")

    mass_bins = np.arange(8, 12, 0.5)
    omega_capped = np.clip(omega, None, 10.0)
    by_mass, labels = [], []
    print("\nomega by mass bin (capped at 10):")
    for i in range(len(mass_bins) - 1):
        sel = (logM >= mass_bins[i]) & (logM < mass_bins[i + 1])
        vals = omega_capped[sel]
        by_mass.append(vals)
        lab = rf"$10^{{{mass_bins[i]}}}$-$10^{{{mass_bins[i+1]}}}$"
        if len(vals):
            print(f"  {mass_bins[i]:.1f}-{mass_bins[i+1]:.1f}  N={len(vals):3d}  "
                  f"median={np.median(vals):6.2f}  std={np.std(vals):.2f}")
            labels.append(lab + rf" (N={len(vals)}, med={np.median(vals):.2f})")
        else:
            labels.append(lab + " (N=0)")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 8), dpi=200, facecolor="w")
    ax0.hist(omega, bins=70, color="0.6", edgecolor="white", lw=0.3)
    ax0.set_xlabel(r"$\omega_{\rm acc}$ (Gyr$^{-1}$)"); ax0.set_ylabel("count")
    ax0.set_title(rf"$\omega$ solutions ($n={n:.3f}$, $k={k:.3f}$)")
    ax0.grid(alpha=0.3)

    edges = np.linspace(omega_capped.min() - 0.5, omega_capped.max() + 0.5, 50)
    ax1.hist(by_mass, bins=edges, stacked=True, color=_COLORS[:len(by_mass)],
             label=labels, edgecolor="white", lw=0.5)
    ax1.set_xlabel(r"$\omega_{\rm acc}$ (Gyr$^{-1}$)"); ax1.set_ylabel("count")
    ax1.set_title("by mass bin")
    ax1.legend(fontsize=8, loc="upper right"); ax1.grid(alpha=0.3)
    fig.tight_layout()

    out = args.out or (ROOT / cfg["paths"]["figures"] / "omega_hist_io.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
