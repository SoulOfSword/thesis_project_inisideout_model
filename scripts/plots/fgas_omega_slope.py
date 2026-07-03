"""Does f_gas still constrain omega where j_bar's F(omega) has gone flat?

The inside-out f_gas likelihood inverts each galaxy's j_bar to its omega through y = F(omega)
(F = j_bar's normalised, mass-independent shape), then grades (n, k) with f_gas. Where F is
flat -- the extreme-omega tails -- j_bar pins omega only weakly and the inversion is
degenerate. A joint (soft-j_bar) fit can break that degeneracy ONLY where f_gas itself still
varies with omega. This evaluates, at fixed (n, k), the two observables' omega-sensitivity:

  F(omega)      and  |dF/domega|       (j_bar; one mass-independent curve)
  f_gas(omega)  and  |df_gas/domega|   (one curve per representative mass; same model path
                                        as logL_jax: fgas_and_jbar_for_galaxies_jax/cutoff_ksl)

Both F and f_gas live in [0,1], so the raw slopes are directly comparable (bottom-left).
Bottom-right turns each slope into an omega-resolution sigma/|slope| using the sample-median
errors -- the quantity that decides whether f_gas out-constrains j_bar in the flat tail
(lower = omega pinned tighter). The grey band marks where |dF/domega| has dropped below 10%
of its peak (j_bar effectively flat); the test is whether f_gas keeps slope inside it. The
sample's inverted omega are drawn as a rug on the top-left. Saves to outputs/figures/.

  python scripts/plots/fgas_omega_slope.py --from outputs/grids/grid_io_fgas_full.npz
  python scripts/plots/fgas_omega_slope.py --params 0.35 1.05 --logM 8.5 9.5 10.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jmfgas.config import load_config
from jmfgas.data import sample_frame
from jmfgas.physics.angmom import j_maxer
from jmfgas.models.inside_out import (F_omega_jax, fgas_and_jbar_for_galaxies_jax,
                                      build_r_acc_matrix_for_all_M_jax)
from jmfgas.models.common import log_M_bar_array_jax


def model_nk(source, params, burn_in):
    """(n, k) from a grid .npz (peak) / chain .h5 (median), or directly from --params."""
    if source is not None:
        source = Path(source)
        m = next((t for t in source.stem.split("_") if t in ("io", "nio")), None)
        if source.suffix == ".npz":
            d = np.load(source, allow_pickle=True)
            m = str(d["model"]) if "model" in d.files else m
            n, k = (float(v) for v in d["peak"])
        else:
            import emcee
            flat = emcee.backends.HDFBackend(str(source), read_only=True).get_chain(
                discard=burn_in, flat=True)
            n, k = (float(v) for v in np.median(flat, axis=0))
        if m != "io":
            raise SystemExit(f"{source.name} is {m!r}; this needs io")
        return n, k
    if params is not None:
        return params[0], params[1]
    raise SystemExit("pass --params N K or --from <grid.npz|chain.h5>")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="source", type=Path,
                   default=ROOT / "outputs" / "grids" / "grid_io_fgas_full.npz")
    p.add_argument("--params", type=float, nargs=2, metavar=("N", "K"), default=None)
    p.add_argument("--sample", default="full")
    p.add_argument("--logM", type=float, nargs="+", default=None,
                   help="representative log10(M_bar) curves (default: 10/50/90th sample pct)")
    p.add_argument("--npts", type=int, default=600, help="omega-grid points (even => skips 0)")
    p.add_argument("--burn-in", type=int, default=50)
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    t0 = args.t0 if args.t0 is not None else cfg["time"]["t0"]
    source = None if args.params is not None else args.source
    n, k = model_nk(source, args.params, args.burn_in)

    # sample: galaxy omega rug (j_bar inverted) + representative masses + median errors
    df = sample_frame(args.sample, ROOT / cfg["paths"]["data"])
    Mbar = df["Mbar"].to_numpy(float)
    jbar = df["jbar"].to_numpy(float)
    e_jbar = df["e_jbar"].to_numpy(float)
    e_fgas = df["e_fgas"].to_numpy(float)
    jmax = j_maxer(Mbar); jmin = jmax / 10.0
    delta = k * jmax - jmin
    y = (jbar - jmin) / delta
    dy = np.abs(e_jbar / delta)
    sig_f = float(np.median(e_fgas))            # representative f_gas error (0-1 space)
    dy_rep = float(np.median(dy))               # representative j_bar error in y (0-1 space)
    masses = args.logM or list(np.percentile(np.log10(Mbar), [10.0, 50.0, 90.0]))

    # omega grid and the two observables' shapes
    w = np.linspace(-10.0, 10.0, args.npts)
    F = np.asarray(F_omega_jax(jnp.asarray(w), n, t0))
    dF = np.gradient(F, w)
    with np.errstate(divide="ignore"):
        t_acc = np.where(np.abs(w) < 1e-9, np.inf, 1.0 / w)   # omega=0 -> flat history
    r_acc = build_r_acc_matrix_for_all_M_jax(jnp.float64(n), jnp.float64(k))
    fgas_curves, dfgas = [], []
    for lm in masses:
        Mb = jnp.full(w.shape, 10.0**lm, dtype=jnp.float64)
        fg, _ = fgas_and_jbar_for_galaxies_jax(
            Mb, jnp.asarray(t_acc), "cutoff_ksl", r_acc, log_M_bar_array_jax, at_t0=True)
        fg = np.asarray(fg)
        fgas_curves.append(fg); dfgas.append(np.gradient(fg, w))

    # galaxy omega rug (invert on the reversed, decreasing F); flat band from |dF|
    wgal = np.clip(np.interp(y, F[::-1], w[::-1]), -10.0, 10.0)
    band = np.abs(dF) < 0.1 * np.nanmax(np.abs(dF))
    res_j = dy_rep / np.abs(dF)                  # omega-resolution from j_bar (mass-indep here)

    cols = plt.cm.plasma(np.linspace(0.0, 0.85, len(masses)))
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), dpi=150, facecolor="w", sharex=True)

    ax[0, 0].plot(w, F, color="navy", lw=2.5)
    ax[0, 0].plot(wgal, np.full_like(wgal, 0.02), "|", ms=12, color="crimson",
                  alpha=0.5, label=f"{len(wgal)} galaxies")
    ax[0, 0].set_ylabel(r"$F(\omega)=y$", fontsize=14); ax[0, 0].set_ylim(-0.02, 1.05)
    ax[0, 0].set_title(r"$j_{\rm bar}$ observable (mass-independent)", fontsize=13)
    ax[0, 0].legend(fontsize=11, loc="upper right")

    for fg, lm, c in zip(fgas_curves, masses, cols):
        ax[0, 1].plot(w, fg, color=c, lw=2, label=rf"$\log M={lm:.2f}$")
    ax[0, 1].set_ylabel(r"$f_{\rm gas}(\omega)$", fontsize=14); ax[0, 1].set_ylim(-0.02, 1.05)
    ax[0, 1].set_title(r"$f_{\rm gas}$ observable", fontsize=13)
    ax[0, 1].legend(fontsize=11, loc="upper right")

    ax[1, 0].plot(w, np.abs(dF), color="navy", lw=2.5, label=r"$|dF/d\omega|$  ($j_{\rm bar}$)")
    for dfg, lm, c in zip(dfgas, masses, cols):
        ax[1, 0].plot(w, np.abs(dfg), color=c, lw=2, label=rf"$|df_{{\rm gas}}/d\omega|$, $\log M={lm:.2f}$")
    ax[1, 0].set_yscale("log"); ax[1, 0].set_ylim(1e-5, 2.0)
    ax[1, 0].set_ylabel(r"$|d/d\omega|$  (both in $[0,1]$)", fontsize=14)
    ax[1, 0].set_title("raw $\\omega$-sensitivity", fontsize=13)
    ax[1, 0].legend(fontsize=9, loc="lower center")

    ax[1, 1].plot(w, res_j, color="navy", lw=2.5, label=r"$\delta\omega$ from $j_{\rm bar}$")
    for dfg, lm, c in zip(dfgas, masses, cols):
        ax[1, 1].plot(w, sig_f / np.abs(dfg), color=c, lw=2,
                      label=rf"$\delta\omega$ from $f_{{\rm gas}}$, $\log M={lm:.2f}$")
    ax[1, 1].set_yscale("log"); ax[1, 1].set_ylim(1e-2, 1e3)
    ax[1, 1].set_ylabel(r"$\delta\omega=\sigma/|{\rm slope}|$ [Gyr$^{-1}$] (lower=tighter)", fontsize=13)
    ax[1, 1].set_title(r"$\sigma$-weighted $\omega$-resolution", fontsize=13)
    ax[1, 1].legend(fontsize=9, loc="upper center")

    for a in ax.ravel():
        a.set_xlim(-10, 10); a.grid(alpha=0.3); a.tick_params(labelsize=12)
        a.fill_between(w, *a.get_ylim(), where=band, color="0.85", alpha=0.5, zorder=0)
    for a in ax[1, :]:
        a.set_xlabel(r"$\omega_{\rm acc}$ [Gyr$^{-1}$]", fontsize=14)

    n_in_band = int(np.sum(np.interp(wgal, w, np.abs(dF)) < 0.1 * np.nanmax(np.abs(dF))))
    fig.suptitle(rf"$n={n:.3f}$, $k={k:.3f}$ — does $f_{{\rm gas}}$ keep slope in the flat-$j_{{\rm bar}}$ band? "
                 rf"({n_in_band}/{len(wgal)} galaxies fall in the band)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or (ROOT / cfg["paths"]["figures"] / "fgas_omega_slope.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")

    # printed verdict: at tail omegas, which observable pins omega tighter (median mass)?
    im = len(masses) // 2; dfg_med = np.abs(dfgas[im])
    print(f"n={n:.4f} k={k:.4f}  sig_f(med)={sig_f:.4f}  dy(med)={dy_rep:.4f}  "
          f"median-mass logM={masses[im]:.2f}")
    print(f"{'omega':>7} {'|dF|':>10} {'dw_jbar':>10} {'|dfgas|':>10} {'dw_fgas':>10} {'fgas/jbar':>10}")
    for wq in (-9.0, -7.0, -5.0, -3.0, 3.0, 5.0, 7.0, 9.0):
        i = int(np.argmin(np.abs(w - wq)))
        rj = dy_rep / abs(dF[i]); rf = sig_f / dfg_med[i]
        print(f"{w[i]:7.2f} {abs(dF[i]):10.2e} {rj:10.2e} {dfg_med[i]:10.2e} {rf:10.2e} {rf/rj:10.3f}")
    print(f"(ratio<1 => f_gas pins omega tighter than j_bar there; {n_in_band}/{len(wgal)} "
          f"galaxies in the flat band)\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
