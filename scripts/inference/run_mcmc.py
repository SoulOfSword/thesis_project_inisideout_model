"""Run an MCMC fit and save the chain.

--model {io,nio} --likelihood {4obs,fgas,a0}

io uses parameters (n, k); nio uses (a, b); a0 fixes a=0 (nio, b only).
4obs and a0 use the CONVERGED+HIX 4-observable sample; fgas can use that
(default) or the full baryons+HIX sample (--sample full-hix).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jmfgas.config import load_config
from jmfgas.inference import run_mcmc
from jmfgas.inference.build import build_log_prob


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["io", "nio"], required=True)
    p.add_argument("--likelihood", choices=["4obs", "fgas", "a0"], required=True)
    p.add_argument("--sample", choices=["mcmc-obs", "converged", "MP_full", "full-hix", "full"],
                   default="mcmc-obs")
    p.add_argument("--nwalkers", type=int, default=32)
    p.add_argument("--nsteps", type=int, default=200)
    p.add_argument("--init", choices=["gaussian", "uniform"], default="gaussian",
                   help="gaussian: a small ball around the config init; "
                        "uniform: flat over the whole prior box (walkers maximally spread)")
    p.add_argument("--max-workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    data_dir = ROOT / cfg["paths"]["data"]
    try:
        log_prob, ndim, init, bounds = build_log_prob(
            args.model, args.likelihood, args.sample, cfg, data_dir)
    except ValueError as e:
        raise SystemExit(str(e))

    rng = np.random.default_rng(args.seed)
    if args.init == "uniform":                       # walkers flat across the whole prior box
        pos = np.empty((args.nwalkers, ndim))
        for d in range(ndim):
            lo, hi = bounds[d]
            eps = 1e-3 * (hi - lo)                    # keep off the strict prior edges
            pos[:, d] = rng.uniform(lo + eps, hi - eps, args.nwalkers)
    else:                                            # gaussian ball around the config init
        pos = np.asarray(init) + 0.1 * rng.standard_normal((args.nwalkers, ndim))
        for d in range(ndim):
            lo, hi = bounds[d]
            pos[:, d] = np.clip(pos[:, d], lo + 0.01, hi - 0.01)

    out = args.out or (ROOT / cfg["paths"]["mcmc_chains"]
                       / f"chain_{args.model}_{args.likelihood}_{args.sample}.h5")
    run_mcmc(log_prob, ndim, args.nwalkers, args.nsteps, out,
             init=pos, fresh_start=not args.resume, max_workers=args.max_workers,
             nsteps_is_total=args.resume)
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
