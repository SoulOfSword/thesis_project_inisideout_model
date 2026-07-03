"""Compare the two F(omega) implementations and visualise the inversion degeneracy.

F(omega) = int_0^1 x^n e^{-omega t0 x} dx / int_0^1 e^{-omega t0 x} dx is the monotone map
inverted to assign each galaxy its accretion rate from y = (j_bar - j_min)/(k j_max - j_min).
This draws a 2x2 mosaic over omega in [-10, 10]:

  top row    : the current model F (fixed 256-pt Simpson quadrature)
  bottom row : the analytic A/B Taylor-series F (machine-precise for both signs)
  left column : F(omega) = y          right column : dF/domega

Observed galaxies are placed at (omega*, y*) on each F curve. The vertical bar is the
j_bar uncertainty mapped into y; the horizontal bar is the omega interval it implies
(via F^-1) -- long where F is flat (the degenerate regimes), short where F is steep.
"""

import argparse
import sys
from functools import partial
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
from jmfgas.models.inside_out import F_omega_jax as F_current


# ---- candidate analytic implementation (A(z)=int_0^1 x^n e^{-zx}dx, B(z)=int_0^1 e^{-zx}dx) ----
@partial(jax.jit, static_argnames=("K",))
def A_integral(z, n, K=256):
    z = jnp.asarray(z, dtype=jnp.float64)
    n = jnp.asarray(n, dtype=jnp.float64)
    k = jnp.arange(1, K, dtype=jnp.float64)
    ones = jnp.ones(jnp.shape(z) + (1,))
    zp = jnp.where(z >= 0.0, z, 0.0)                       # z >= 0: e^{-z} Sigma z^k/(n+1)_{k+1}
    cp = jnp.cumprod(zp[..., None] / (n + 1.0 + k), axis=-1)
    A_pos = jnp.exp(-zp) * jnp.sum(jnp.concatenate([ones, cp], axis=-1) / (n + 1.0), axis=-1)
    s = jnp.where(z < 0.0, -z, 0.0)                        # z < 0: Sigma s^k/(k!(n+1+k))
    cn = jnp.cumprod((s[..., None] / k) * (n + k) / (n + 1.0 + k), axis=-1)
    A_neg = jnp.sum(jnp.concatenate([ones, cn], axis=-1) / (n + 1.0), axis=-1)
    return jnp.where(z >= 0.0, A_pos, A_neg)


@jax.jit
def B_integral(z):
    z = jnp.asarray(z, dtype=jnp.float64)
    safe = jnp.where(z == 0.0, 1.0, z)
    return jnp.where(z == 0.0, jnp.ones_like(z), -jnp.expm1(-z) / safe)


@partial(jax.jit, static_argnames=("K",))
def F_analytic(omega, n, t0, K=256):
    z = jnp.asarray(omega, dtype=jnp.float64) * t0
    return A_integral(z, n, K) / B_integral(z)


def model_nk(source, params, burn_in):
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
    p.add_argument("--ngal", type=int, default=15, help="galaxies to draw (evenly spaced in y)")
    p.add_argument("--burn-in", type=int, default=50)
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    t0 = args.t0 if args.t0 is not None else cfg["time"]["t0"]
    source = None if args.params is not None else args.source
    n, k = model_nk(source, args.params, args.burn_in)

    # galaxies -> y = (j_bar - j_min)/(k j_max - j_min) and its uncertainty dy
    df = sample_frame(args.sample, ROOT / cfg["paths"]["data"])
    Mbar = df["Mbar"].to_numpy(float)
    jbar = df["jbar"].to_numpy(float)
    e_jbar = df["e_jbar"].to_numpy(float)
    jmax = j_maxer(Mbar); jmin = jmax / 10.0
    delta = k * jmax - jmin
    y = (jbar - jmin) / delta
    dy = np.abs(e_jbar / delta)
    good = np.isfinite(y) & np.isfinite(dy)
    order = np.argsort(y[good])
    pick = np.unique(np.linspace(0, order.size - 1, args.ngal).round().astype(int))
    sel = np.where(good)[0][order[pick]]
    ys, dys = y[sel], dy[sel]

    w = np.linspace(-10.0, 10.0, 600)
    F_cur = np.asarray(F_current(jnp.asarray(w), n, t0))
    F_ana = np.asarray(F_analytic(jnp.asarray(w), n, t0))

    def invert(Fc, yq):                                    # F decreasing -> interp on the reverse
        return np.interp(yq, Fc[::-1], w[::-1])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150, facecolor="w")
    rows = [(F_cur, "current (Simpson, N=256)"), (F_ana, "analytic A/B Taylor")]
    for r, (Fc, name) in enumerate(rows):
        dFc = np.gradient(Fc, w)
        ws = invert(Fc, ys)
        w_hi = invert(Fc, ys - dys)                        # lower y -> higher omega
        w_lo = invert(Fc, ys + dys)
        axF, axD = axes[r][0], axes[r][1]
        axF.plot(w, Fc, color="navy", lw=2.5, zorder=1)
        axF.errorbar(ws, ys, xerr=[ws - w_lo, w_hi - ws], yerr=dys, fmt="o", ms=5,
                     color="crimson", ecolor="0.5", elinewidth=1.2, capsize=2, alpha=0.85, zorder=3)
        axF.set_xlim(-10, 10)
        axF.set_xlabel(r"$\omega_{\rm acc}$ [Gyr$^{-1}$]", fontsize=14)
        axF.set_ylabel(r"$F(\omega)\;=\;y$", fontsize=14)
        axF.set_title(f"$F(\\omega)$ ({name})", fontsize=13)
        axF.grid(alpha=0.3); axF.tick_params(labelsize=12)

        axD.plot(w, np.abs(dFc), color="navy", lw=2.5, zorder=1)
        axD.scatter(ws, np.abs(np.interp(ws, w, dFc)), color="crimson", s=25, zorder=3)
        axD.axhline(0, color="k", lw=0.8, ls="--")
        axD.set_xlim(-10, 10)
        axD.set_yscale("log")
        axD.set_xlabel(r"$\omega_{\rm acc}$ [Gyr$^{-1}$]", fontsize=14)
        axD.set_ylabel(r"$\vert dF/d\omega \vert$", fontsize=14)
        axD.set_title(f"$dF/d\\omega$ ({name})", fontsize=13)
        axD.grid(alpha=0.3); axD.tick_params(labelsize=12)

    fig.suptitle(rf"$n={n:.3f}$, $k={k:.3f}$",# — {len(sel)} galaxies "
                 #r"(vertical bar: $\delta j$ in $y$; horizontal: implied $\delta\omega$)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or (ROOT / cfg["paths"]["figures"] / f"F_omega_compare_{n:.1f}_{k:.1f}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
