"""Draw the three final planes (j_bar-M_bar-f_gas, stellar, gaseous) for one model.

--model accretion  inside-out engine, fixed (n, k), the scatter is a swept accretion rate omega
                   (or the saved tables in data/data9_JAX_aKSL/ when no --params is given)
--model spin       inside-out engine, omega=a*(Mbar/1e10)**b fixed by mass, the scatter is a swept
                   spin parameter k (0.2..2, capping j_acc) at n=1

The model tracks (one line per fixed f_gas level, coloured by f_gas) overlay the
observed CONVERGED sample and the external compilation samples. Three PDFs are
written to the configured figures path: plane_jMfgas_{model}, plane_stellar_{model},
plane_gaseous_{model}.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jmfgas.config import load_config
from jmfgas.data import build_converged
from jmfgas.viz import plane_jM_fgas, plane_stellar, plane_gaseous
from jmfgas.viz.planes import FGAS_LEVELS


def observed_arrays(data_dir):
    """CONVERGED sample as log observables + relative (d ln x = dx/x) errors."""
    df = build_converged(data_dir)
    g = lambda c: df[c].values.astype(float)
    obs = {"fgas": g("fgas")}
    for tag, mcol, jcol in [("bar", "Mbar", "jbar"), ("gas", "Mgas", "jgas"),
                            ("star", "Mstar", "jstar")]:
        obs[f"log_M{tag}"] = np.log10(g(mcol))
        obs[f"log_M{tag}_err"] = g("e_" + mcol) / g(mcol)
        obs[f"log_j{tag}"] = np.log10(g(jcol))
        obs[f"log_j{tag}_err"] = g("e_" + jcol) / g(jcol)
    return obs


def full_baryonic_obs(data_dir):
    """Baryonic-plane points for the full MP+21b (BARY) sample, converged + failed galaxies.
    HIX is excluded here — it is drawn separately as a compilation (crosses)."""
    from jmfgas.data import build_full
    df = build_full(data_dir, with_hix=False)
    g = lambda c: df[c].values.astype(float)
    return {"fgas": g("fgas"),
            "log_Mbar": np.log10(g("Mbar")), "log_Mbar_err": g("e_Mbar") / g("Mbar"),
            "log_jbar": np.log10(g("jbar")), "log_jbar_err": g("e_jbar") / g("jbar")}


def stellar_arrays(data_dir):
    """Stellar-plane points for all baryons ∩ stars galaxies (wider than converged)."""
    from jmfgas.data import build_stellar
    df = build_stellar(data_dir)
    g = lambda c: df[c].values.astype(float)
    return {"fgas": g("fgas"),
            "log_Mstar": np.log10(g("Mstar")), "log_Mstar_err": g("e_Mstar") / g("Mstar"),
            "log_jstar": np.log10(g("jstar")), "log_jstar_err": g("e_jstar") / g("jstar")}


def gaseous_arrays(data_dir):
    """Gaseous-plane points for all baryons ∩ gas galaxies (wider than converged)."""
    from jmfgas.data import build_gaseous
    df = build_gaseous(data_dir)
    g = lambda c: df[c].values.astype(float)
    return {"fgas": g("fgas"),
            "log_Mgas": np.log10(g("Mgas")), "log_Mgas_err": g("e_Mgas") / g("Mgas"),
            "log_jgas": np.log10(g("jgas")), "log_jgas_err": g("e_jgas") / g("jgas")}


def compilation_frames(data_dir):
    """Compilation samples for the plane overlays (one DataFrame per stem). The two Dwarfs
    files (used + not-used, disjoint galaxies) are concatenated into a single 'Dwarfs' series,
    so they share one marker and one legend entry."""
    cdir = data_dir / "compilation_AM_others"
    frames = {p.stem: pd.read_csv(p) for p in sorted(cdir.glob("*.csv"))}
    notused = data_dir / "compilation_AM_others_notused"
    extra = notused / "Dwarfs.csv"
    if "Dwarfs" in frames and extra.exists():
        frames["Dwarfs"] = pd.concat([frames["Dwarfs"], pd.read_csv(extra)], ignore_index=True)
    for name in ("GLSBs", "UDGs"):                # baryonic-only (no stellar/gaseous columns)
        f = notused / f"{name}.csv"
        if f.exists():
            frames[name] = pd.read_csv(f)
    return frames


def io_model_grids(grid_dir):
    """Present-day per-mass model grids from the saved cutoff-KSL text files in grid_dir."""
    load = lambda name: np.nan_to_num(np.loadtxt(grid_dir / f"final_{name}_cutoff_ksl.txt"))
    return {"f_gas": load("f_gas"), "j_bar": load("j_bar"),
            "j_star": load("j_star"), "j_gas": load("j_gas"),
            "M_star": load("Mstar_grid"), "M_gas": load("Mgas_grid")}


# accretion-bias scatter grid: omega swept (Gyr^-1; t_acc = 1/omega) at fixed (n, k)
_ACCRETION_OMEGA = (-1.0, -0.3, 0.1, 1.0 / 3.0, 0.75, 1.0, 2.0, 4.0, 8.0, 10.0)
# spin scatter grid: k swept (caps j_acc at t0) at fixed omega = omega_Mdep(M), n=1
_SPIN_K = np.linspace(0.2, 6.0, 25)

# fiducial parameters per model; anything else gets its values appended to the file name
_FIDUCIAL = {"accretion": (1.0, 2.0), "spin": (0.1, 0.5)}
_PARAM_NAMES = {"accretion": ("n", "k"), "spin": ("a", "b")}


def param_suffix(model, p0, p1):
    """'_n0.5_k2' for non-fiducial parameters, '' for the fiducial ones."""
    fid = _FIDUCIAL.get(model)
    if fid is None or (p0, p1) == fid:
        return ""
    n0, n1 = _PARAM_NAMES[model]
    return f"_{n0}{p0:g}_{n1}{p1:g}"


def accretion_model_grids(logM, n, k, sfl, lambda_ratio=1.0):
    """Inside-out engine over the mass + omega grid at fixed (n, k) — the accretion band.

    lambda_ratio scales j_min for a halo of non-median spin; 1.0 is the median halo."""
    import jax.numpy as jnp
    from jmfgas.models import build_r_acc_matrix_for_all_M_jax, run_all_masses
    t_acc = jnp.asarray(1.0 / np.asarray(_ACCRETION_OMEGA, float), dtype=jnp.float64)
    r_acc = build_r_acc_matrix_for_all_M_jax(jnp.float64(n), jnp.float64(k),
                                             lambda_ratio=jnp.float64(lambda_ratio))
    out = run_all_masses(jnp.asarray(10.0 ** logM, dtype=jnp.float64), t_acc, r_acc,
                         jnp.asarray(logM, dtype=jnp.float64), star_formation_law=sfl)
    keys = ("f_gas", "j_bar", "j_gas", "j_star", "M_star", "M_gas")
    return {key: np.asarray(v) for key, v in zip(keys, out)}


def lambda_ratios(cfg):
    """The five lambda/lambda_median values from the config (median first is NOT assumed)."""
    ls = cfg["lambda_scatter"]
    return np.exp(ls["sigma_ln"] * np.asarray(ls["offsets"], float))


def read_source(path, burn_in):
    """(model, p0, p1, sample) from a grid .npz (its peak) or a chain .h5 (its median)."""
    path = Path(path)
    model = next((tok for tok in path.stem.split("_")
                  if tok in ("accretion", "spin", "io", "nio")), None)
    sample = None
    if path.suffix == ".npz":
        d = np.load(path, allow_pickle=True)
        if "model" in d.files:
            model = str(d["model"])
        if "sample" in d.files:
            sample = str(d["sample"])
        p0, p1 = (float(v) for v in d["peak"])
    else:
        import emcee
        flat = emcee.backends.HDFBackend(str(path), read_only=True).get_chain(
            discard=burn_in, flat=True)
        p0, p1 = (float(v) for v in np.median(flat, axis=0))
    model = {"io": "accretion", "nio": "spin"}.get(model, model)   # legacy grids/chains
    if model not in ("accretion", "spin"):
        raise SystemExit(f"can't tell accretion/spin from {path.name}; pass --model")
    return model, p0, p1, sample


def spin_model_grids(logM, a, b, sfl):
    """Inside-out engine over the mass + k grid at fixed omega=omega_Mdep(M) — the spin-bias band."""
    from jmfgas.models import spin_grids_over_k
    return spin_grids_over_k(logM, a, b, _SPIN_K, sfl)


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[run_final_planes] wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="source", type=Path, default=None,
                   help="grid .npz or chain .h5 to read (model, best-fit params) from")
    p.add_argument("--model", choices=["accretion", "spin"], default=None,
                   help="only needed without --from")
    p.add_argument("--params", type=float, nargs=2, default=None, metavar=("P1", "P2"),
                   help="(n, k) or (a, b), with --model, instead of --from")
    p.add_argument("--sample", default=None,
                   help="observed sample for the baryonic plane (e.g. 'full'); default: converged")
    p.add_argument("--burn-in", type=int, default=50, help="chain burn-in if --from is an .h5")
    p.add_argument("--sfl", default=None, help="star-formation law (default: config sfl.default)")
    p.add_argument("--t0", type=float, default=None, help="present-day time [Gyr]")
    p.add_argument("--out", type=Path, default=None, help="output directory for the PDFs")
    p.add_argument("--planes", nargs="+", default=["baryonic", "stellar", "gaseous"],
                   choices=["baryonic", "stellar", "gaseous"], help="which planes to draw")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    data_dir = ROOT / cfg["paths"]["data"]
    out_dir = args.out or (ROOT / cfg["paths"]["figures"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sfl = args.sfl or cfg["sfl"]["default"]

    mg = cfg["mass_grid"]
    logM = np.linspace(mg["logM_min"], mg["logM_max"], mg["n"])

    obs = observed_arrays(data_dir)                   # converged sample (all three planes)
    comp = compilation_frames(data_dir)

    if args.source is not None:                       # read model + params from grid/chain
        model, p0, p1, sample = read_source(args.source, args.burn_in)
        stem = args.source.stem
    elif args.model is not None:
        model, stem, sample = args.model, args.model, None
        p0, p1 = args.params if args.params is not None else (None, None)
    else:
        raise SystemExit("pass --from <grid.npz|chain.h5>  (or --model with --params)")
    if args.sample is not None:                       # --sample overrides (e.g. full with --model)
        sample = args.sample

    # a full-sample choice -> each plane shows its maximal per-plane set (+ compilations), not just
    # the converged intersection: baryonic all BARY, stellar baryons∩stars, gaseous baryons∩gas
    use_full = sample in ("full", "full-hix", "MP_full")
    obs_by_tag = {
        "baryonic": full_baryonic_obs(data_dir) if use_full else obs,
        "stellar": stellar_arrays(data_dir) if use_full else obs,
        "gaseous": gaseous_arrays(data_dir) if use_full else obs,
    }

    if model == "accretion":
        if p0 is None:
            grids = io_model_grids(data_dir / "data9_JAX_aKSL")   # original saved tables
        else:
            grids = accretion_model_grids(logM, p0, p1, sfl)
            stem += param_suffix(model, p0, p1)
    elif model == "spin":
        a, b = (p0, p1) if p0 is not None else (
            cfg["mcmc"]["spin"]["init"][0], cfg["mcmc"]["spin"]["init"][1])
        grids = spin_model_grids(logM, a, b, sfl)
        stem += param_suffix(model, a, b)
    else:
        raise SystemExit(f"unknown model {model!r} (expected accretion or spin)")

    plane_specs = [
        ("baryonic", plane_jM_fgas, "j_bar", "jbar"),
        ("stellar", plane_stellar, "M_star", "stellar"),
        ("gaseous", plane_gaseous, "M_gas", "gaseous"),
    ]
    j_for = {"baryonic": "j_bar", "stellar": "j_star", "gaseous": "j_gas"}
    for tag, fn, mkey, lkey in plane_specs:
        if tag not in args.planes:
            continue
        levels = FGAS_LEVELS[model][lkey]         # f_gas line set differs by model + plane
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200, facecolor="w")
        if tag == "baryonic":
            fn(ax, logM, grids["j_bar"], grids["f_gas"], obs_by_tag[tag], comp, levels=levels)
        else:
            fn(ax, grids[mkey], grids[j_for[tag]], grids["f_gas"], obs_by_tag[tag], comp,
               levels=levels)
        _save(fig, out_dir / f"plane_{tag}_{stem}.pdf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
