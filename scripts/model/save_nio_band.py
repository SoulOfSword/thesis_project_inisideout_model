"""Regenerate the non-inside-out jbar-fgas profile tables.

For each mass the model is swept over a j_acc grid (j_max/10 .. j_max, n_j points)
and the present-day observables are written as CSV with columns
j_acc_kpc_km_s, j_bar_kpc_km_s, f_gas, j_gas_kpc_km_s, j_star_kpc_km_s.

Two sets are produced:
  nio/jbar_fgas_logM{9,10,11}.csv        integer panel masses
  nio/band/jbar_fgas_logM{M:.4f}.csv     fine mass grid covering the panel bins

--params a b : omega(logM) = a*(logM-10) + b  (default from config mcmc.nio.init)
--sfl        : star-formation law (default from config)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax.numpy as jnp

from jmfgas.config import load_config
from jmfgas.models.non_inside_out import (build_r_acc_for_single_M,
                                          Full_final_definer_Mdep_omega_jax)

DEFAULT_BAND_BINS = [(9.0, 9.5), (9.7, 10.3), (10.8, 11.2)]


def model_row(logM, a, b, n_j, sfl):
    """j_acc-swept present-day observables for one mass; returns a DataFrame."""
    r_acc_pc, j_acc = build_r_acc_for_single_M(logM, n_j=n_j)
    r_acc_jax = jnp.array(r_acc_pc, dtype=jnp.float64)
    f_gas, j_bar, j_gas, j_star, _, _ = Full_final_definer_Mdep_omega_jax(
        float(logM), a, b, r_acc_jax, star_formation_law=sfl, at_t0=True)
    return pd.DataFrame({
        "j_acc_kpc_km_s": np.asarray(j_acc),
        "j_bar_kpc_km_s": np.asarray(j_bar),
        "f_gas": np.asarray(f_gas),
        "j_gas_kpc_km_s": np.asarray(j_gas),
        "j_star_kpc_km_s": np.asarray(j_star),
    })


def parse_bins(spec):
    out = []
    for chunk in spec.split(","):
        lo, hi = chunk.split(":")
        out.append((float(lo), float(hi)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg = load_config()
    nio_init = cfg["mcmc"]["nio"]["init"]
    p.add_argument("--params", type=float, nargs=2, metavar=("a", "b"),
                   default=nio_init, help="omega(logM) coefficients a b")
    p.add_argument("--sfl", default=cfg["sfl"]["default"])
    p.add_argument("--n-j", type=int, default=cfg["integration"]["n_j"],
                   help="j_acc grid points per mass")
    p.add_argument("--masses", type=float, nargs="+", default=[9, 10, 11],
                   help="integer panel masses")
    p.add_argument("--band-bins", type=parse_bins, default=DEFAULT_BAND_BINS,
                   help='fine-grid bins, e.g. "9.0:9.5,9.7:10.3,10.8:11.2"')
    p.add_argument("--band-step", type=float, default=0.05)
    p.add_argument("--no-band", action="store_true", help="skip the fine band grid")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / cfg["paths"]["outputs"] / "jbar_fgas_profiles" / "nio")
    args = p.parse_args()

    a, b = args.params
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for logM in args.masses:
        df = model_row(logM, a, b, args.n_j, args.sfl)
        lbl = int(logM) if float(logM).is_integer() else logM
        out = args.out_dir / f"jbar_fgas_logM{lbl}.csv"
        df.to_csv(out, index=False)
        print(f"[save_nio_band] wrote {out}  rows={len(df)}")

    if args.no_band:
        return

    band_dir = args.out_dir / "band"
    band_dir.mkdir(parents=True, exist_ok=True)
    band_masses = np.round(np.unique(np.concatenate(
        [np.arange(lo, hi + 1e-9, args.band_step) for lo, hi in args.band_bins])), 4)
    for logM_b in band_masses:
        df = model_row(float(logM_b), a, b, args.n_j, args.sfl)
        out = band_dir / f"jbar_fgas_logM{float(logM_b):.4f}.csv"
        df.to_csv(out, index=False)
    print(f"[save_nio_band] wrote {len(band_masses)} band profiles -> {band_dir}")


if __name__ == "__main__":
    sys.exit(main())
