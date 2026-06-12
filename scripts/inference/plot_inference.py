"""Plot an inference result.

--chain CHAIN.h5 : corner.corner (median/mode/16-84) + walker traces (burn-in/thin).
--grid GRID.npz  : one corner per zoom level (sampled posterior) into a <grid>.corner/ folder.
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
from jmfgas.inference import get_sampler_from_backend
from jmfgas.viz import chain_corner, grid_corner, grid_logL_map, trace_plot


def _resolve_labels(args, ndim):
    """Default labels to the model's parameter names (io: n, k; nio: a, b)."""
    if args.labels:
        return args.labels
    model = args.model
    if model is None and args.chain is not None:
        parts = args.chain.stem.split("_")          # chain_<model>_<lik>_<sample>...
        if len(parts) > 1 and parts[1] in ("io", "nio"):
            model = parts[1]
    if model in ("io", "nio"):
        return [f"${n}$" for n in load_config()["mcmc"][model]["param_names"]]
    return [f"$p_{{{i}}}$" for i in range(ndim)]


def plot_chain(args):
    backend = get_sampler_from_backend(args.chain)
    chain = backend.get_chain()
    flat = backend.get_chain(discard=args.burn_in, thin=args.thin, flat=True)
    if flat.shape[0] < 2:
        raise SystemExit(f"empty chain after burn-in={args.burn_in} "
                         f"(chain has {chain.shape[0]} steps); lower --burn-in")
    ndim = flat.shape[1]
    labels = _resolve_labels(args, ndim)
    stem = args.chain.with_suffix("")
    chain_corner(flat, labels, args.title).savefig(f"{stem}_corner.pdf", bbox_inches="tight")
    trace_plot(chain, labels, args.burn_in).savefig(f"{stem}_chains.pdf", bbox_inches="tight")
    for i, lab in enumerate(labels):
        lo, mid, hi = np.percentile(flat[:, i], [16, 50, 84])
        print(f"{lab} = {mid:.4f} (+{hi-mid:.4f} / -{mid-lo:.4f})")
    print(f"wrote {stem}_corner.pdf")
    print(f"wrote {stem}_chains.pdf")


def plot_grid(args):
    d = np.load(args.grid, allow_pickle=True)
    labels = args.labels or [str(l) for l in d["labels"]]
    n_levels = int(d["n_levels"])
    peak = d["peak"] if "peak" in d.files else None
    # paper figure: the sampled-posterior corner of the finest level
    fig, stats = grid_corner(d["ax0"], d["ax1"], d["logL"], labels)
    corner_out = args.grid.with_suffix(".corner.pdf")
    fig.savefig(corner_out, bbox_inches="tight"); plt.close(fig)
    # diagnostic: one ΔlogL map per zoom level (level 0 = coarse prior, last = finest)
    lvl_dir = args.grid.with_name(args.grid.stem + ".levels")
    lvl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_levels):
        f = grid_logL_map(d[f"ax0_{i}"], d[f"ax1_{i}"], d[f"logL_{i}"], labels, peak=peak)
        f.savefig(lvl_dir / f"level{i}.pdf", bbox_inches="tight"); plt.close(f)
    for lab, (mid, lo, hi, mode) in zip(labels, stats):
        print(f"{lab} = {mid:.4f} (+{hi-mid:.4f} / -{mid-lo:.4f})  mode={mode:.4f}")
    print(f"wrote {corner_out}  and {n_levels} level maps -> {lvl_dir}/")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--chain", type=Path)
    src.add_argument("--grid", type=Path)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--model", choices=["io", "nio"], default=None,
                   help="label params by model (default: inferred from the chain filename)")
    p.add_argument("--burn-in", type=int, default=0)
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--title", default=None)
    args = p.parse_args()
    plot_grid(args) if args.grid else plot_chain(args)


if __name__ == "__main__":
    sys.exit(main())
