"""Fit the disc scale radius - flat velocity relation from SPARC rotation curves.

Fits each SPARC rotation curve with an exponential-disc (Boissier) form to get
R_v, pairs it with the published V_flat, drops poorly-constrained galaxies, and
fits log R_v = gamma * log(v_flat/100) + delta with the Bayesian line fitter
(orthogonal intrinsic scatter). Writes the relation to
data/rv_vflat_relation.json (used by the model's rv_def).
"""

import argparse
import json
import sys
from pathlib import Path
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jmfgas.config import load_config
from BayesLineFit_mod import BayesLineFit


def tanh(x, vf, rv):
    return vf * np.tanh(x / rv)


def boissier(r, wc, rv):
    return wc * (1.0 - np.exp(-r / rv))


def _clean_rc(df):
    r = pd.to_numeric(df["Rad[kpc]"], errors="coerce").to_numpy()
    v = pd.to_numeric(df["Vobs[km/s]"], errors="coerce").to_numpy()
    ev = pd.to_numeric(df["errV[km/s]"], errors="coerce").to_numpy()
    m = np.isfinite(r) & np.isfinite(v) & np.isfinite(ev) & (r >= 0)
    r, v, ev = r[m], v[m], ev[m]
    if r.size:
        order = np.argsort(r)
        r, v, ev = r[order], v[order], ev[order]
    ev = np.where(ev <= 0, np.nan, ev)
    return r, v, ev


def _initial_guesses(r, v):
    if len(r) == 0 or len(v) == 0:
        return 150.0, 1.0
    vf0 = float(np.nanpercentile(v, 90)) if np.isfinite(np.nanpercentile(v, 90)) else 150.0
    R = float(np.nanmax(r)) if np.isfinite(np.nanmax(r)) and np.nanmax(r) > 0 else 1.0
    return vf0, max(0.2, 0.4 * R)


def _fit_model(model_fn, r, v, ev, p0, bounds):
    use_weight = np.isfinite(ev).sum() >= max(3, len(ev) // 3)
    try_p0 = p0
    for _ in range(3):
        try:
            with catch_warnings():
                simplefilter("ignore")
                if use_weight:
                    popt, pcov = curve_fit(model_fn, r, v, sigma=ev, p0=try_p0,
                                           bounds=bounds, absolute_sigma=True, maxfev=200000)
                else:
                    popt, pcov = curve_fit(model_fn, r, v, p0=try_p0,
                                           bounds=bounds, maxfev=200000)
            perr = np.sqrt(np.diag(pcov))
            if not np.all(np.isfinite(perr)):
                raise RuntimeError("non-finite covariance")
            return popt, perr
        except Exception:
            use_weight = False
            try_p0 = _initial_guesses(r, v)
    return np.array([np.nan, np.nan]), np.array([np.nan, np.nan])


def fit_radii(rotcurve_dir, names_keep):
    """Per-galaxy tanh and Boissier fits -> two {name: (name, vrot, e_vrot, rv, e_rv)} maps."""
    tanh_map, bois_map = {}, {}
    bounds = ([1.0, 1e-3], [1000.0, 100.0])
    for filename in sorted(rotcurve_dir.iterdir()):
        if filename.suffix != ".dat":
            continue
        galaxy = filename.stem.split("_")[0]
        if galaxy not in names_keep:
            continue
        df = pd.read_table(filename, sep="\t",
                           usecols=["Rad[kpc]", "Vobs[km/s]", "errV[km/s]"])
        r, v, ev = _clean_rc(df)
        if len(r) < 5 or len(v) < 5:
            tanh_map[galaxy] = (galaxy, np.nan, np.nan, np.nan, np.nan)
            bois_map[galaxy] = (galaxy, np.nan, np.nan, np.nan, np.nan)
            continue
        p0 = _initial_guesses(r, v)
        for model_fn, store in ((tanh, tanh_map), (boissier, bois_map)):
            (vf, rv), (vf_e, rv_e) = _fit_model(model_fn, r, v, ev, p0=p0, bounds=bounds)
            if not np.all(np.isfinite([vf_e, rv_e])):
                vf_e, rv_e = np.nan, np.nan
            store[galaxy] = (galaxy, vf, vf_e, rv, rv_e)
    return tanh_map, bois_map


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--inc-min", type=float, default=40.0)
    p.add_argument("--inc-max", type=float, default=79.0)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "rv_vflat_relation.json")
    p.add_argument("--figdir", type=Path, default=None,
                   help="directory for the line-fit diagnostics (default: config figures path)")
    p.add_argument("--scatter", choices=["vertical", "orthogonal"], default="vertical",
                   help="intrinsic-scatter direction (vertical suits this directional relation)")
    args = p.parse_args()

    cfg = load_config(args.config)
    data_dir = ROOT / cfg["paths"]["data"]
    fig_dir = args.figdir or (ROOT / cfg["paths"]["figures"])

    sparc = pd.read_csv(data_dir / "SPARC_Lelli2016c.mrt.csv")
    sparc_idx = sparc.drop_duplicates("Name").set_index("Name")
    inc = sparc["Inc"].to_numpy()
    sel = (inc >= args.inc_min) & (inc <= args.inc_max)
    names_finc = list(sparc["Name"].values[sel])   # SPARC catalog order
    names_finc_set = set(names_finc)

    tanh_map, bois_map = fit_radii(data_dir / "SPARC_rotcurves_mod", names_finc_set)

    names = [n for n in names_finc
             if n in sparc_idx.index and n in tanh_map and n in bois_map]
    rows = []
    for n in names:
        vflat = float(sparc_idx.at[n, "Vflat"])
        if not (np.isfinite(vflat) and vflat > 0):
            continue
        _, vt, vt_e, rt, rt_e = tanh_map[n]
        _, vb, vb_e, rb, rb_e = bois_map[n]
        rows.append((n, vflat, float(sparc_idx.at[n, "e_Vflat"]),
                     vt, vt_e, rt, rt_e, vb, vb_e, rb, rb_e))
    tab = pd.DataFrame(rows, columns=["Name", "Vflat", "e_Vflat",
                                      "vrot_t", "e_vrot_t", "Rv_t", "e_Rv_t",
                                      "vrot_b", "e_vrot_b", "Rv_b", "e_Rv_b"])
    tab["Delta_t"] = np.abs(tab["vrot_t"] - tab["Vflat"])
    tab["Delta_b"] = np.abs(tab["vrot_b"] - tab["Vflat"])

    # poorly-constrained galaxies excluded from the line fit
    sus = set(tab[(tab["Rv_t"] > 4) & (tab["vrot_t"] > 150)]["Name"])
    sus |= set(tab[(tab["Rv_b"] > 4) & (tab["vrot_b"] > 150)]["Name"])
    sus |= set(tab[tab["Delta_t"] > 25]["Name"])
    sus |= set(tab[tab["Delta_b"] > 25]["Name"])
    for n in names:
        if not (np.isfinite(sparc_idx.at[n, "Vflat"]) and sparc_idx.at[n, "Vflat"] > 200):
            continue
        rt, rb = tanh_map[n][3], bois_map[n][3]
        if (np.isfinite(rt) and rt > 2) or (np.isfinite(rb) and rb > 2):
            sus.add(n)

    mask = (~tab["Name"].isin(sus)
            & np.isfinite(tab["Rv_b"]) & (tab["Rv_b"] > 0)
            & np.isfinite(tab["e_Rv_b"]) & (tab["e_Rv_b"] > 0)
            & np.isfinite(tab["Vflat"]) & (tab["Vflat"] > 0)
            & np.isfinite(tab["e_Vflat"]) & (tab["e_Vflat"] > 0))
    x = tab.loc[mask, "Vflat"].to_numpy()
    sx = tab.loc[mask, "e_Vflat"].to_numpy()
    y = tab.loc[mask, "Rv_b"].to_numpy()
    sy = tab.loc[mask, "e_Rv_b"].to_numpy()

    # Fit log R_v = gamma*log(v_flat/100) + delta with the Bayesian line fitter;
    # v_flat normalised to 100 km/s, log10 errors propagated.
    lx, lsx = np.log10(x / 100.0), sx / x / np.log(10)
    ly, lsy = np.log10(y), sy / y / np.log(10)
    fig_dir.mkdir(parents=True, exist_ok=True)
    a, b, s, _ = BayesLineFit(
        lx, ly, err_x=lsx, err_y=lsy, orthfit=(args.scatter == "orthogonal"),
        plot_title="rvflat",
        outfile_chain=str(fig_dir / "rv_vflat_chain.dat"),
        outfile_bestfit=str(fig_dir / "rv_vflat_bestfit.txt"),
        outplot_convergence=str(fig_dir / "rv_vflat_convergence"),
        outplot_corner=str(fig_dir / "rv_vflat_corner"),
        outplot_bestfit=str(fig_dir / "rv_vflat_bestfit"),
    )
    gamma, delta = float(a[1]), float(b[1])          # posterior medians

    rel = {
        "form": "R_v[kpc] = 10**delta * (v_flat[km/s]/100)**gamma",
        "gamma": gamma,
        "delta": delta,
        "gamma_err_up": float(a[2]),
        "gamma_err_dw": float(a[3]),
        "delta_err_up": float(b[2]),
        "delta_err_dw": float(b[3]),
        "scatter": float(s[1]),
        "fit_method": f"BayesLineFit {args.scatter} scatter, Boissier radii, v_flat/100",
        "n_galaxies": int(mask.sum()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rel, f, indent=2)
    print(f"R_v = 10**{delta:.4f} * (v_flat/100)**{gamma:.4f}  (N={rel['n_galaxies']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
