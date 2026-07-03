"""Build the observed galaxy samples and cache each to data/sample_<name>.csv.

Samples:
  converged : 77 galaxies in baryons ∩ gas ∩ stars (corrected); also writes
              mcmc-obs (converged + HIX, the 4-observable table)
  MP_full   : the full MP+21 baryonic sample (--with-hix also writes full-hix = MP_full + HIX)
  full      : MP_full + HIX + extra compilations (UDGs/GLSBs/Dwarfs), errors imputed,
              baryonic columns only  (for the f_gas likelihood)
  all       : every sample above

The grid / MCMC scripts read these CSVs by sample name.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jmfgas.config import load_config
from jmfgas.data import sample_frame


def _save(name, data_dir):
    df = sample_frame(name, data_dir)
    out = data_dir / f"sample_{name}.csv"
    df.to_csv(out, index=False)
    print(f"{name:10s}: {len(df):4d} galaxies -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", choices=["converged", "MP_full", "full", "all"], default="all")
    p.add_argument("--with-hix", action="store_true",
                   help="also write full-hix (MP_full + HIX); implied by --sample all")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    data_dir = ROOT / load_config(args.config)["paths"]["data"]
    s = args.sample
    if s in ("converged", "all"):
        _save("converged", data_dir)
        _save("mcmc-obs", data_dir)
    if s in ("MP_full", "all"):
        _save("MP_full", data_dir)
        if args.with_hix or s == "all":
            _save("full-hix", data_dir)
    if s in ("full", "all"):
        _save("full", data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
