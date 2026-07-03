"""Single-galaxy radial surface-density profiles (gas, stars, SFR) for the inside-out model.

Given (n, k), a galaxy mass and one or more accretion rates omega, build the present-day
radial profiles with the model engine and plot log10(Sigma) vs radius for each. Useful for
inspecting intermediate results in the extreme-omega regimes (where the omega inversion or
the Euler-Gamma extension behind r_acc can misbehave): pass a few omega spanning the caps,
e.g. --omega -10 -1 1 10. Per-omega present-day M_gas / M_star / f_gas / j_bar are printed.
Radius is shown to 25 kpc. Besides the present-day figure, one time-evolution figure per
quantity is written (a panel per omega, profiles at 3/6/9/12 Gyr). Saves to outputs/profiles/.

  python scripts/plots/radial_profiles.py --params 0.5 2.0 --logM 10.0 --omega -10 -1 1 10
  python scripts/plots/radial_profiles.py --from outputs/grids/grid_io_fgas_full.npz --logM 9.0 --omega 8
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

_PANELS = [("Sigma_gas", r"$\Sigma_{\rm gas}$ [$M_\odot\,{\rm pc}^{-2}$]"),
           ("Sigma_star", r"$\Sigma_\star$ [$M_\odot\,{\rm pc}^{-2}$]"),
           ("Sigma_sfr", r"$\Sigma_{\rm SFR}$ [$M_\odot\,{\rm pc}^{-2}\,{\rm Gyr}^{-1}$]")]
_EVOL_TIMES = (3.0, 6.0, 9.0, 12.0)               # Gyr; time slices for the evolution figures
_RMAX_KPC = 25.0


def _style(ax, ylabel):
    """Shared axis styling: radius limit, larger labels/ticks, grid."""
    ax.set_xlim(0.0, _RMAX_KPC)
    ax.set_xlabel(r"$R$ [kpc]", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.3)


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
            raise SystemExit(f"{source.name} is {m!r}; this script supports io only")
        return n, k
    if params is not None:
        return params[0], params[1]
    raise SystemExit("pass --params N K or --from <grid.npz|chain.h5>")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logM", type=float, required=True, help="log10(M_bar/Msun) of the galaxy")
    p.add_argument("--omega", type=float, nargs="+", required=True,
                   help="accretion rate(s) omega [Gyr^-1]; one curve per value (0 = flat history)")
    p.add_argument("--params", type=float, nargs=2, metavar=("N", "K"), default=None)
    p.add_argument("--from", dest="source", type=Path, default=None,
                   help="grid .npz / chain .h5 to read (n, k) from instead of --params")
    p.add_argument("--burn-in", type=int, default=50)
    p.add_argument("--sfl", default=None, help="star-formation law (default: config sfl.default)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    sfl = args.sfl or cfg["sfl"]["default"]
    out_dir = args.out or (ROOT / cfg["paths"]["outputs"] / "profiles")
    out_dir.mkdir(parents=True, exist_ok=True)
    n, k = model_nk(args.source, args.params, args.burn_in)

    import jax.numpy as jnp
    from jmfgas.models.profiles import radial_profiles_io
    from jmfgas.models.inside_out import build_r_acc_matrix_for_all_M_jax
    from jmfgas.models.common import log_M_bar_array_jax
    r_acc = build_r_acc_matrix_for_all_M_jax(jnp.float64(n), jnp.float64(k))

    profs = []                                            # one full time-resolved profile per omega
    print(f"n={n:.4f} k={k:.4f} logM={args.logM:.3f} sfl={sfl}")
    print(f"{'omega':>8} {'M_gas':>11} {'M_star':>11} {'f_gas':>7} {'j_bar':>9}")
    for omega in args.omega:
        t_acc = np.inf if omega == 0.0 else 1.0 / omega   # omega=0 -> flat accretion history (t_acc->inf)
        prof = radial_profiles_io(jnp.float64(args.logM), jnp.float64(t_acc),
                                  r_acc, log_M_bar_array_jax, sfl_type=sfl)
        print(f"{omega:8.3g} {float(prof['M_gas_t'][-1]):11.3e} "
              f"{float(prof['M_star_t'][-1]):11.3e} {float(prof['f_gas_t'][-1]):7.3f} "
              f"{float(prof['j_bar_t'][-1]):9.4f}")
        profs.append({q: np.asarray(prof[q]) for q, _ in _PANELS})
    R = np.asarray(prof["r_kpc"]); times = np.asarray(prof["times"])
    rmask = R <= _RMAX_KPC
    base = f"profiles_io_logM{args.logM:.2f}_n{n:.2f}_k{k:.2f}_omega{'_'.join(f'{o:g}' for o in args.omega)}"

    def logS(arr):                                        # log10 of Sigma over R <= 25 kpc (Sigma>0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(np.where(arr > 0, arr, np.nan))[rmask]

    # present-day profiles: one panel per quantity, one curve per omega
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150, facecolor="w")
    ocolors = plt.cm.plasma(np.linspace(0.0, 0.9, len(args.omega)))
    for ax, (q, lbl) in zip(axes, _PANELS):
        for omega, pr, color in zip(args.omega, profs, ocolors):
            ax.plot(R[rmask], logS(pr[q][:, -1]), color=color, lw=2, label=rf"$\omega={omega:g}$")
        _style(ax, r"$\log_{10}\,$" + lbl)
    axes[0].legend(fontsize=11, title=rf"$\log M_{{\rm bar}}={args.logM:.2f}$,"
                                      rf" $n={n:.2f}$, $k={k:.2f}$")
    fig.tight_layout()
    fig.savefig(out_dir / f"{base}.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_dir / (base + '.pdf')}")

    # time evolution: one figure per quantity, one panel per omega, one curve per time
    tcolors = plt.cm.plasma(np.linspace(0.0, 0.85, len(_EVOL_TIMES)))
    its = [int(np.argmin(np.abs(times - t))) for t in _EVOL_TIMES]
    for q, lbl in _PANELS:
        fig, axes = plt.subplots(1, len(args.omega), figsize=(5.0 * len(args.omega), 4.8),
                                 dpi=150, facecolor="w", squeeze=False)
        for ax, omega, pr in zip(axes[0], args.omega, profs):
            for t, it, color in zip(_EVOL_TIMES, its, tcolors):
                ax.plot(R[rmask], logS(pr[q][:, it]), color=color, lw=2, label=rf"${t:g}$ Gyr")
            ax.set_title(rf"$\omega={omega:g}$", fontsize=14)
            _style(ax, r"$\log_{10}\,$" + lbl)
        axes[0][0].legend(fontsize=11, title="time")
        fig.tight_layout()
        fig.savefig(out_dir / f"{base}_evolution_{q}.pdf", bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out_dir / (base + '_evolution_' + q + '.pdf')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
