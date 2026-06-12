"""Build the observed galaxy sample tables.

--sample converged : the 77-galaxy corrected sample -> data/common_sample.csv,
                     plus the CONVERGED+HIX 4-observable table -> data/mcmc_observables.csv
--sample full      : all baryons galaxies (corrected where gas+stars exist, else raw)
                     -> data/full_sample.csv  (add --with-hix to append the HIX galaxies)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jmfgas.config import load_config
from jmfgas.data import build_converged, build_full, build_mcmc_observables


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", choices=["converged", "full"], default="converged")
    p.add_argument("--with-hix", action="store_true",
                   help="append HIX galaxies (full sample only)")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    data_dir = ROOT / load_config(args.config)["paths"]["data"]

    if args.sample == "converged":
        conv = build_converged(data_dir)
        out = data_dir / "common_sample.csv"
        conv.to_csv(out, index=False)
        print(f"converged: {len(conv)} galaxies -> wrote {out}")
        mcmc = build_mcmc_observables(data_dir)
        out2 = data_dir / "mcmc_observables.csv"
        mcmc.to_csv(out2, index=False)
        print(f"converged+HIX: {len(mcmc)} galaxies -> wrote {out2}")
    else:
        full = build_full(data_dir, with_hix=args.with_hix)
        name = "full_sample_hix.csv" if args.with_hix else "full_sample.csv"
        out = data_dir / name
        full.to_csv(out, index=False)
        print(f"full{'+HIX' if args.with_hix else ''}: {len(full)} galaxies -> wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
