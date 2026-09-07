"""j_max vs M_bar diagnostic (the j_max analogue of omega_vs_mass).

j_max = k * j_MP(M_bar) is the angular-momentum ceiling each model assigns to a galaxy. The two
models fill in k differently, so this figure shows the mirror image of omega_vs_mass:

  variable accretion : k is fixed (the accretion k, default 2), so j_max = k * j_MP(M_bar) is a
                       single smooth line -- every galaxy shares the same ceiling at a given mass.
  variable torques   : k is inverted per galaxy from its observed j_bar (at omega = omega_Mdep),
                       so j_max = k * j_MP scatters galaxy-to-galaxy -- shown as the per-mass-bin
                       median with 16-84% scatter.

(In omega_vs_mass it is the other way round: the accretion model carries the scatter and the torques
model is the fixed mass law.)
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
from jmfgas.physics.angmom import j_maxer
from jmfgas.models import invert_k_jax

_MASS_BINS = np.arange(8, 12.5, 0.5)            # covers the full sample (up to logM ~ 11.8)


def _torques_jmax(logM, jbar, a, b, n=1.0):
    """Per-galaxy ceiling j_max = k * j_MP, with k inverted at omega = omega_Mdep(M_bar)."""
    import jax.numpy as jnp
    k = np.asarray(invert_k_jax(jnp.asarray(logM, dtype=jnp.float64),
                                jnp.asarray(jbar, dtype=jnp.float64), a, b, n=n))
    return k * np.asarray(j_maxer(10.0 ** np.asarray(logM))), k


def _binned(logM, values):
    """(centre, median, 16, 84) of `values` in each mass bin."""
    cen = 0.5 * (_MASS_BINS[:-1] + _MASS_BINS[1:])
    med = np.full(len(cen), np.nan); lo = med.copy(); hi = med.copy()
    for i in range(len(cen)):
        v = values[(logM >= _MASS_BINS[i]) & (logM < _MASS_BINS[i + 1])]
        if len(v):
            med[i], lo[i], hi[i] = np.percentile(v, [50, 16, 84])
    return cen, med, lo, hi


def build_figure(args, data_dir):
    n, k_acc = (float(v) for v in args.accretion_params)
    a, b = (float(v) for v in args.spin_params)
    t = obs_table(args.sample, data_dir)
    logM = np.asarray(t["logMbar"]); jbar = np.asarray(t["jbar"])

    jmax_torq, k_torq = _torques_jmax(logM, jbar, a, b, n)
    cen, med, lo, hi = _binned(logM, jmax_torq)

    xl = np.linspace(_MASS_BINS[0], _MASS_BINS[-1], 200)
    jmax_acc_line = k_acc * np.asarray(j_maxer(10.0 ** xl))

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300, facecolor="w")
    vi = np.isfinite(med)
    ax.errorbar(cen[vi], med[vi], xerr=0.25, yerr=[med[vi] - lo[vi], hi[vi] - med[vi]],
                fmt="o", color="royalblue", capsize=3, alpha=0.85,
                label=f"variable torques "+"\n"+rf"($a={a:.1f}$, $b={b:.1f}$)")
    ax.plot(xl, jmax_acc_line, color="r", lw=2.5, zorder=5,
            label=rf"variable accretion: "+"\n"+rf"$j_{{\max}}={k_acc:g}\,j_{{\rm MP}}(M_{{\rm bar}})$")
    ax.plot(xl, np.asarray(j_maxer(10.0 ** xl)), color="gray", ls=":", lw=1.5,
            label=r"$j_{\rm MP}(M_{\rm bar})$")

    ax.set_yscale("log")
    ax.set_xlabel(r"$\log(M_{\rm bar}\,/\,\rm M_\odot)$", fontsize=14)
    ax.set_ylabel(r"$j_{\max}$ (kpc km s$^{-1}$)", fontsize=14)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3); fig.tight_layout()

    print(f"variable torques: k over {len(logM)} galaxies in "
          f"[{k_torq.min():.2f}, {k_torq.max():.2f}] (median {np.median(k_torq):.2f})")
    print(f"{'logM bin':>13} {'N':>4}   {'torques j_max median [16,84]':>34}   {'acc 2*j_MP':>11}")
    for i in range(len(cen)):
        m = (logM >= _MASS_BINS[i]) & (logM < _MASS_BINS[i + 1])
        if not m.sum():
            continue
        acc = k_acc * float(j_maxer(10.0 ** cen[i]))
        print(f"  [{_MASS_BINS[i]:.1f},{_MASS_BINS[i+1]:.1f}) {int(m.sum()):>4}   "
              f"{med[i]:>10.0f} [{lo[i]:.0f}, {hi[i]:.0f}]".ljust(50) + f"{acc:>11.0f}")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", default="mcmc-obs")
    p.add_argument("--accretion-params", type=float, nargs=2, default=[1.0, 2.0], metavar=("n", "k"),
                   help="variable-accretion (n, k); the line is j_max = k * j_MP")
    p.add_argument("--spin-params", type=float, nargs=2, default=[0.1, 0.5], metavar=("a", "b"),
                   help="variable-torques omega = a*(Mbar/1e10)**b, for the per-galaxy k inversion")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    data_dir = ROOT / cfg["paths"]["data"]
    fig = build_figure(args, data_dir)

    out = args.out or (ROOT / cfg["paths"]["figures"] / "jmax_vs_mass.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
