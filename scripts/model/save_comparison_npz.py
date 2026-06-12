"""Regenerate the assumptions-vs-results comparison profiles.

Time-resolved single-galaxy profiles for the comparison figures:
  io  -> comparison_io_{pos,neg}_omega.npz   (growing vs decaying accretion)
  nio -> comparison_nio_{high,low}_jacc.npz  (high vs low accreted j)

Each npz holds the full profile dict (times, f_gas_t, j_bar_t, ...) plus the
scalars the plotting cells need (omega, C, M_bar; the IO files also carry
j_acc_t, k, n; the NIO files carry j_acc_value). M_dot_acc is derived downstream
from C, omega and times.

--model {io,nio}  --logM  --params (n k) | (a b)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax.numpy as jnp

from jmfgas.config import load_config
from jmfgas.models.common import log_M_bar_array_jax, M_times1, C_def_jax
from jmfgas.models.inside_out import build_r_acc_matrix_for_all_M_jax
from jmfgas.physics.angmom import j_maxer, j_acc_def
from jmfgas.models.profiles import radial_profiles_io, radial_profiles_nio


def _np(d):
    """Profile dict -> numpy (scalars kept as-is)."""
    return {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in d.items()}


def save_io(logM, n, k, sfl, out_dir):
    M_bar = 10.0 ** logM
    r_acc_matrix = build_r_acc_matrix_for_all_M_jax(n, k)
    j_acc_t = np.asarray(j_acc_def(j_maxer(M_bar), M_times1, n=n, con=k))
    written = []
    for label, omega in [("pos_omega", 1.0 / 3.0), ("neg_omega", -1.0 / 3.0)]:
        t_acc = 1.0 / omega
        prof = _np(radial_profiles_io(logM, t_acc, r_acc_matrix,
                                      log_M_bar_array_jax, sfl_type=sfl))
        prof.update(omega=omega, C=float(C_def_jax(M_bar, t_acc)),
                    M_bar=M_bar, j_acc_t=j_acc_t, k=k, n=n)
        out = out_dir / f"comparison_io_{label}.npz"
        np.savez(out, **prof)
        written.append(out)
    return written


def save_nio(logM, a, b, sfl, out_dir):
    M_bar = 10.0 ** logM
    j_max = float(j_maxer(M_bar))
    omega = a * (logM - 10.0) + b
    C = float(C_def_jax(M_bar, 1.0 / omega))
    written = []
    for label, j_val in [("high_jacc", j_max), ("low_jacc", j_max / 10.0)]:
        prof = radial_profiles_nio(logM, j_val, a, b, sfl_type=sfl)  # already numpy
        prof.update(C=C, j_acc_value=j_val, M_bar=M_bar)
        out = out_dir / f"comparison_nio_{label}.npz"
        np.savez(out, **prof)
        written.append(out)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    cfg = load_config()
    p.add_argument("--model", choices=["io", "nio"], required=True)
    p.add_argument("--logM", type=float, default=10.0)
    p.add_argument("--params", type=float, nargs=2, default=None,
                   help="io: n k (default 0.5 1.5); nio: a b (default 2 2)")
    p.add_argument("--sfl", default=cfg["sfl"]["default"])
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    prof_root = ROOT / cfg["paths"]["outputs"] / "model_radial_profiles"
    out_dir = args.out_dir or (prof_root / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "io":
        n, k = args.params if args.params else (0.5, 1.5)
        written = save_io(args.logM, n, k, args.sfl, out_dir)
    else:
        a, b = args.params if args.params else (2.0, 2.0)
        written = save_nio(args.logM, a, b, args.sfl, out_dir)

    for out in written:
        print(f"[save_comparison_npz] wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
