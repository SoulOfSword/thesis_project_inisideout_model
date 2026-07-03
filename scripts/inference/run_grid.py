"""Evaluate a 2-D log-likelihood grid with adaptive zoom and save it.

--model {io,nio} --likelihood {4obs,fgas}

Coarse-to-fine: a coarse grid over the prior, then repeated zoom into the peak's
high-likelihood region until the cell spacing is below the target or the level
cap is hit. The saved .npz holds the finest grid plus every level and the peak.
Use plot_grid.py to render it (CostaCorner). a0 is 1-D; use run_mcmc.py for it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jmfgas.config import load_config
from jmfgas.inference import build_log_prob, adaptive_grid

LABELS = {"io": ["n", "k"], "nio": ["a", "b"]}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["io", "nio"], required=True)
    p.add_argument("--likelihood", choices=["4obs", "fgas"], required=True)
    p.add_argument("--sample", choices=["mcmc-obs", "full-hix", "converged", "MP_full", "full"],
                   default="mcmc-obs")
    p.add_argument("--mass-min", type=float, default=None, help="logMbar lower cut (converged sample)")
    p.add_argument("--mass-max", type=float, default=None, help="logMbar upper cut (converged sample)")
    p.add_argument("--exclude-hix", action="store_true", help="drop HIX galaxies (converged sample)")
    p.add_argument("--bounds", type=float, nargs=4, default=None,
                   metavar=("N_LO", "N_HI", "K_LO", "K_HI"),
                   help="fix the grid box instead of the prior (for a zoomed scan)")
    p.add_argument("--ref-grid", type=Path, default=None,
                   help="another grid .npz; store its peak_logL as ref_logL (shared normalization)")
    p.add_argument("--coarse-n", type=int, default=None)
    p.add_argument("--max-levels", type=int, default=None)
    p.add_argument("--target-spacing", type=float, default=None)
    p.add_argument("--dlogL", type=float, default=None)
    p.add_argument("--max-workers", type=int, default=12)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    g = cfg["grid"]
    data_dir = ROOT / cfg["paths"]["data"]
    mass_range = None
    if args.mass_min is not None or args.mass_max is not None:
        mass_range = (args.mass_min if args.mass_min is not None else -np.inf,
                      args.mass_max if args.mass_max is not None else np.inf)
    try:
        log_prob, ndim, _, bounds = build_log_prob(
            args.model, args.likelihood, args.sample, cfg, data_dir,
            mass_range=mass_range, exclude_hix=args.exclude_hix)
    except ValueError as e:
        raise SystemExit(str(e))
    if ndim != 2:
        raise SystemExit(f"grid inference is 2-D; {args.model}/{args.likelihood} is "
                         f"{ndim}-D. Use run_mcmc.py instead.")
    if args.bounds is not None:                 # fixed zoom box overrides the prior bounds
        bounds = [(args.bounds[0], args.bounds[1]), (args.bounds[2], args.bounds[3])]
    ref_logL = (float(np.load(args.ref_grid, allow_pickle=True)["peak_logL"])
                if args.ref_grid is not None else np.nan)

    pick = lambda v, d: d if v is None else v   # 0.0 is a valid override, don't fall through
    res = adaptive_grid(
        log_prob, bounds,
        n=pick(args.coarse_n, g["coarse_n"]),
        max_levels=pick(args.max_levels, g["max_levels"]),
        target_spacing=pick(args.target_spacing, g["target_spacing"]),
        dlogL=pick(args.dlogL, g["dlogL"]),
        max_workers=args.max_workers)

    labels = LABELS[args.model]
    print(f"stop: {res['stop_reason']}  peak {labels[0]}={res['peak'][0]:.4f} "
          f"{labels[1]}={res['peak'][1]:.4f}  logL={res['peak_logL']:.4f}  "
          f"components={res['n_components']}"
          + ("  [MULTIMODAL]" if res["multimodal"] else ""))

    fine = res["levels"][-1]
    sample_tag = args.sample.replace("mcmc", "grid")    # a grid output shouldn't say "mcmc"
    if mass_range is not None:
        sample_tag += f"_M{args.mass_min}-{args.mass_max}"
    if args.exclude_hix:
        sample_tag += "_noHIX"
    if args.bounds is not None:
        sample_tag += "_zoom"
    out = args.out or (ROOT / cfg["paths"]["grids"]
                       / f"grid_{args.model}_{args.likelihood}_{sample_tag}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    per_level = {}
    for i, lv in enumerate(res["levels"]):
        per_level[f"ax0_{i}"] = lv["ax0"]
        per_level[f"ax1_{i}"] = lv["ax1"]
        per_level[f"logL_{i}"] = lv["logL"]
    np.savez(out,
             ax0=fine["ax0"], ax1=fine["ax1"], logL=fine["logL"],
             labels=np.array(labels), model=args.model, likelihood=args.likelihood,
             sample=args.sample, stop_reason=res["stop_reason"],
             n_components=res["n_components"], peak=np.array(res["peak"]),
             peak_logL=res["peak_logL"], ref_logL=ref_logL, n_levels=len(res["levels"]),
             **per_level)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
