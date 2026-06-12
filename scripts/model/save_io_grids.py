"""Regenerate the present-day inside-out model grids.

For 50 baryonic masses (config mass_grid) and a fixed set of accretion rates,
writes the final_*_cutoff_ksl.txt tables: f_gas, j_bar, j_gas, j_star, M_star,
M_gas, each of shape (n_masses, n_omega).

--params n k : inside-out growth parameters (default from config mcmc.io.init)
--sfl        : star-formation law (default from config)
--omega      : accretion-rate grid omega [Gyr^-1]; t_acc = 1/omega per column
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax.numpy as jnp

from jmfgas.config import load_config
from jmfgas.models.common import T0, log_M_bar_array
from jmfgas.models.inside_out import build_r_acc_matrix_for_all_M_jax, run_all_masses

# default accretion-rate grid (one column per value); t_acc = 1 / omega
DEFAULT_OMEGA = [-1.0, -0.3, 0.1, 1.0 / 3.0, 0.75, 1.0, 2.0, 4.0, 8.0, 10.0]

# observable -> output filename stem (each saved as <stem>_<sfl>.txt)
GRID_FILES = {
    "f_gas": "final_f_gas",
    "j_bar": "final_j_bar",
    "j_gas": "final_j_gas",
    "j_star": "final_j_star",
    "M_star": "final_Mstar_grid",
    "M_gas": "final_Mgas_grid",
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg = load_config()
    io_init = cfg["mcmc"]["io"]["init"]
    p.add_argument("--params", type=float, nargs=2, metavar=("n", "k"),
                   default=io_init, help="inside-out growth params n k")
    p.add_argument("--sfl", default=cfg["sfl"]["default"])
    p.add_argument("--t0", type=float, default=cfg["time"]["t0"])
    p.add_argument("--omega", type=float, nargs="+", default=DEFAULT_OMEGA,
                   help="accretion-rate grid [Gyr^-1]; t_acc = 1/omega")
    p.add_argument("--masses", type=int, default=None,
                   help="use only the first N masses (smoke test)")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / cfg["paths"]["data"] / "data9_JAX_aKSL")
    args = p.parse_args()

    if abs(args.t0 - T0) > 1e-9:
        raise SystemExit(
            f"--t0 {args.t0} differs from the engine time grid (t0={T0} from config); "
            "change config/model.yaml time.t0 and re-import to use another value.")

    n, k = args.params
    logM = log_M_bar_array if args.masses is None else log_M_bar_array[:args.masses]
    Mbar_grid = jnp.array(10.0 ** logM, dtype=jnp.float64)
    logM_jax = jnp.array(logM, dtype=jnp.float64)

    t_acc = jnp.array(1.0 / np.asarray(args.omega, dtype=float), dtype=jnp.float64)
    r_acc_matrix = build_r_acc_matrix_for_all_M_jax(n, k)
    if args.masses is not None:
        r_acc_matrix = r_acc_matrix[:args.masses]

    f_gas, j_bar, j_gas, j_star, M_star, M_gas = run_all_masses(
        Mbar_grid, t_acc, r_acc_matrix, logM_jax, star_formation_law=args.sfl)

    grids = {"f_gas": f_gas, "j_bar": j_bar, "j_gas": j_gas,
             "j_star": j_star, "M_star": M_star, "M_gas": M_gas}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key, stem in GRID_FILES.items():
        out = args.out_dir / f"{stem}_{args.sfl}.txt"
        np.savetxt(out, np.asarray(grids[key]))
        print(f"[save_io_grids] wrote {out}  shape={np.asarray(grids[key]).shape}")


if __name__ == "__main__":
    sys.exit(main())
