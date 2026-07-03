"""Fit the two star-formation laws used by the model from resolved gas+SFR profiles.

Assembles atomic+molecular gas and SFR surface-density profiles for a set of
nearby galaxies, maps them onto the rotation-curve radii to attach the local
orbital frequency Omega, and fits two laws with the Bayesian line fitter
(orthogonal intrinsic scatter), both in log10 space:

  volumetric (Omega-corrected):  log(Sigma_SFR/Omega) = n*log(Sigma_gas) + log(alpha)
  surface-density power law:      log(Sigma_SFR)       = n*log(Sigma_gas) + log(alpha)

The fitted (alpha, n) for each law, the canonical low-slope law, and the cutoff
where the steep and canonical laws cross are written to data/sfl_relations.json
(read by physics/sfl.py).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jmfgas.config import load_config
from BayesLineFit_mod import BayesLineFit

LN10 = np.log(10.0)
OMEGA_CONV = 1.022e-9          # (km/s)/kpc -> 1/yr
DWARFS = ["DDO47", "DDO50", "DDO52", "DDO87", "DDO101",
          "DDO126", "DDO133", "DDO168", "NGC2366", "WLM"]
OLD_KSL = (0.1625, 1.4)        # canonical low-slope law (alpha, n); fixed, not fitted

_MAIN_COLS = ["gal", "r", "hi", "hi_err", "h2", "h2_err", "sfrd", "sfrd_err"]
_RC_COLS = ["R[kpc]", "Vrot[km/s]", "e_Vrot[km/s]", "Sigma_gas[Msun/kpc2]",
            "e_Sigma_gas[Msun/kpc2]", "Sigma_SFR[Msun/yr/kpc2]", "e_Sigma_SFR[Msun/yr/kpc2]"]


def _log_interp(R, S, clamp=False):
    """log-space linear interpolator S(R) (S>0); NaN (or edge-held) outside the range."""
    R = np.asarray(R, float); S = np.asarray(S, float)
    m = np.isfinite(R) & np.isfinite(S) & (S > 0)
    if m.sum() < 2:
        return None
    o = np.argsort(R[m])
    logS = np.log10(S[m][o])
    fill = (logS[0], logS[-1]) if clamp else (np.nan, np.nan)
    f = interp1d(R[m][o], logS, kind="linear", bounds_error=False, fill_value=fill)
    return lambda Rn: 10.0 ** f(Rn)


def load_profiles(data_dir):
    """Per-radius atomic+molecular gas and SFR surface densities (all galaxies)."""
    df = pd.read_csv(data_dir / "all_gals_Leroy08forHI_Frank16forH2.txt",
                     sep=r"\s+", header=None, names=_MAIN_COLS, comment="#",
                     engine="python", dtype={"gal": "string"})
    df["gal"] = df["gal"].str.strip()
    return df


def load_rotcurves(data_dir):
    """{galaxy: rotation-curve DataFrame} from the resolved-profile files."""
    rc = {}
    for f in sorted((data_dir / "allSFRprofs").glob("*.txt")):
        gal = f.stem.split("_")[0].strip()
        df = pd.read_table(f, sep=r"\s+", engine="python", usecols=_RC_COLS)
        rc[gal] = df.dropna(subset=["R[kpc]", "Vrot[km/s]"]).sort_values("R[kpc]")
    return rc


def interp_onto_rotcurves(prof, rc):
    """Map each galaxy's gas/SFR profile onto its rotation-curve radii (to attach Omega)."""
    rows = []
    for gal, grp in prof.groupby("gal"):
        gal = str(gal)
        if gal not in rc:
            continue
        R_old = rc[gal]["R[kpc]"].to_numpy(float)
        r = grp["r"].to_numpy(float); o = np.argsort(r)
        sfr = grp["sfrd"].to_numpy(float)[o]
        sfr_err = grp["sfrd_err"].to_numpy(float)[o]
        zero = sfr == 0.0
        sfr[zero] = np.nan; sfr_err[zero] = np.nan
        f_hi = _log_interp(r[o], grp["hi"].to_numpy(float)[o])
        f_hi_e = _log_interp(r[o], grp["hi_err"].to_numpy(float)[o])
        f_h2 = _log_interp(r[o], grp["h2"].to_numpy(float)[o])
        f_h2_e = _log_interp(r[o], grp["h2_err"].to_numpy(float)[o])
        f_sfr = _log_interp(r[o], sfr)
        f_sfr_e = _log_interp(r[o], sfr_err)
        if f_hi is None or f_h2 is None or f_sfr is None:
            continue
        nan = np.full(R_old.shape, np.nan)
        rows.append(pd.DataFrame({
            "gal": gal,
            "Vrot[km/s]": rc[gal]["Vrot[km/s]"].to_numpy(float),
            "e_Vrot[km/s]": rc[gal]["e_Vrot[km/s]"].to_numpy(float),
            "R[kpc]": R_old,
            "hi": f_hi(R_old), "hi_err": f_hi_e(R_old) if f_hi_e is not None else nan,
            "h2": f_h2(R_old), "h2_err": f_h2_e(R_old) if f_h2_e is not None else nan,
            "sfrd": f_sfr(R_old), "sfrd_err": f_sfr_e(R_old) if f_sfr_e is not None else nan,
        }))
    return pd.concat(rows, ignore_index=True)


def dwarf_frame(gal, data_dir):
    """One dwarf's profile (no H2; gas = 1.4 x Sigma_gas for He), columns matching both tables."""
    d = pd.read_csv(data_dir / "allSFRprofs" / f"{gal}_R_Vrot_SigmaGas_SigmaSFR.txt", sep=r"\s+")
    return pd.DataFrame({
        "gal": gal,
        "Vrot[km/s]": d["Vrot[km/s]"].to_numpy(float),
        "e_Vrot[km/s]": d["e_Vrot[km/s]"].to_numpy(float),
        "R[kpc]": d["R[kpc]"].to_numpy(float),
        "hi": 1.4 * d["Sigma_gas[Msun/kpc2]"].to_numpy(float) * 1e-6,
        "hi_err": 1.4 * d["e_Sigma_gas[Msun/kpc2]"].to_numpy(float) * 1e-6,
        "h2": 0.0, "h2_err": 0.0,
        "sfrd": d["Sigma_SFR[Msun/yr/kpc2]"].to_numpy(float),
        "sfrd_err": d["e_Sigma_SFR[Msun/yr/kpc2]"].to_numpy(float),
    })


def _gas_sfr(df):
    """(Sigma_gas, e, Sigma_SFR, e) in Msun/pc^2 and Msun/yr/pc^2. SFR is stored in units of
    1e-4 Msun/yr/kpc^2 for the spirals (scaled here) but Msun/yr/kpc^2 for the dwarfs."""
    s_gas = (df["hi"].to_numpy(float) + df["h2"].to_numpy(float)) * 1e6        # Msun/kpc^2
    s_gas_e = np.sqrt(df["hi_err"].to_numpy(float)**2 + df["h2_err"].to_numpy(float)**2) * 1e6
    is_dwarf = df["gal"].isin(DWARFS).to_numpy()
    s_sfr = df["sfrd"].to_numpy(float).copy(); s_sfr_e = df["sfrd_err"].to_numpy(float).copy()
    s_sfr[~is_dwarf] *= 1e-4; s_sfr_e[~is_dwarf] *= 1e-4                        # Msun/yr/kpc^2
    return s_gas, s_gas_e, s_sfr, s_sfr_e


def boissier_arrays(on_rot):
    """log10 (Sigma_gas, Sigma_SFR/Omega) and their log errors for the Omega-corrected law."""
    s_gas, s_gas_e, s_sfr, s_sfr_e = _gas_sfr(on_rot)
    v = on_rot["Vrot[km/s]"].to_numpy(float); ev = on_rot["e_Vrot[km/s]"].to_numpy(float)
    r = on_rot["R[kpc]"].to_numpy(float)
    omega = (v / r) * OMEGA_CONV; omega_e = (ev / r) * OMEGA_CONV
    m = (np.isfinite(s_gas) & np.isfinite(s_sfr) & np.isfinite(omega)
         & (s_sfr != 0) & (s_sfr_e != 0))
    s_gas, s_gas_e = s_gas[m] * 1e-6, s_gas_e[m] * 1e-6                         # -> Msun/pc^2
    s_sfr, s_sfr_e = s_sfr[m] * 1e-6, s_sfr_e[m] * 1e-6
    omega, omega_e = omega[m], omega_e[m]
    y = s_sfr / omega
    lx, lex = np.log10(s_gas), s_gas_e / s_gas / LN10
    ly = np.log10(y)
    ley = np.sqrt((s_sfr_e / s_sfr)**2 + (omega_e / omega)**2) / LN10
    return lx, ly, lex, ley


def ksl_arrays(prof):
    """log10 (Sigma_gas, Sigma_SFR/Gyr) and their log errors for the surface-density law."""
    s_gas, s_gas_e, s_sfr, s_sfr_e = _gas_sfr(prof)
    m = (np.isfinite(s_gas) & (s_gas != 0) & np.isfinite(s_sfr) & (s_sfr != 0)
         & (s_sfr_e != 0) & np.isfinite(s_gas_e) & (s_gas_e != 0) & np.isfinite(s_sfr_e))
    s_gas, s_gas_e = s_gas[m] * 1e-6, s_gas_e[m] * 1e-6
    s_sfr, s_sfr_e = s_sfr[m] * 1e-6, s_sfr_e[m] * 1e-6
    keep = s_gas >= 0.15                                                       # drop low-density tail
    s_gas, s_gas_e = s_gas[keep], s_gas_e[keep]
    s_sfr, s_sfr_e = s_sfr[keep], s_sfr_e[keep]
    lx, lex = np.log10(s_gas), s_gas_e / s_gas / LN10
    ly, ley = np.log10(s_sfr * 1e9), s_sfr_e / s_sfr / LN10                    # SFR per Gyr
    return lx, ly, lex, ley


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "sfl_relations.json")
    p.add_argument("--figdir", type=Path, default=None,
                   help="directory for the line-fit diagnostics (default: config figures path)")
    p.add_argument("--dry-run", action="store_true",
                   help="assemble the samples and print point counts, without fitting")
    args = p.parse_args()

    cfg = load_config(args.config)
    data_dir = ROOT / cfg["paths"]["data"]
    fig_dir = args.figdir or (ROOT / cfg["paths"]["figures"])

    prof = load_profiles(data_dir)
    rc = load_rotcurves(data_dir)
    on_rot = interp_onto_rotcurves(prof, rc)
    dwarfs = [dwarf_frame(g, data_dir) for g in DWARFS]
    on_rot = pd.concat([on_rot] + dwarfs, ignore_index=True)
    prof = pd.concat([prof] + dwarfs, ignore_index=True)

    bx, by, bex, bey = boissier_arrays(on_rot)
    kx, ky, kex, key_ = ksl_arrays(prof)
    print(f"boissier: {len(bx)} points | ksl: {len(kx)} points")
    if args.dry_run:
        return 0

    fig_dir.mkdir(parents=True, exist_ok=True)
    ab, bb, sb, _ = BayesLineFit(
        bx, by, bex, bey, orthfit=True, plot_title="boissier", outfile_chain=None,
        outfile_bestfit=str(fig_dir / "sfl_boissier_bestfit.txt"), outplot_convergence=None,
        outplot_corner=str(fig_dir / "sfl_boissier_corner"),
        outplot_bestfit=str(fig_dir / "sfl_boissier_bestfit"))
    ak, bk, sk, _ = BayesLineFit(
        kx, ky, kex, key_, orthfit=True, plot_title="kennicutt", outfile_chain=None,
        outfile_bestfit=str(fig_dir / "sfl_ksl_bestfit.txt"), outplot_convergence=None,
        outplot_corner=str(fig_dir / "sfl_ksl_corner"),
        outplot_bestfit=str(fig_dir / "sfl_ksl_bestfit"))

    n_boi, la_boi = float(ab[1]), float(bb[1])      # median slope, median log10(alpha)
    n_ksl, la_ksl = float(ak[1]), float(bk[1])
    a_boi, a_ksl = 10.0**la_boi, 10.0**la_ksl
    a_old, n_old = OLD_KSL
    threshold = (np.log10(a_old) - la_ksl) / (n_ksl - n_old)    # new_ksl x old_ksl crossing

    rel = {
        "form": {
            "boissier": "Sigma_SFR/Omega = alpha * Sigma_gas**n  [Msun/pc^2, yr^-1]",
            "new_ksl": "Sigma_SFR = alpha * Sigma_gas**n  [Msun/pc^2/Gyr]",
            "old_ksl": "canonical low-slope law (fixed, not fitted)",
            "cutoff_ksl": "new_ksl below cutoff_threshold, old_ksl above",
        },
        "boissier": {"alpha": a_boi, "n": n_boi, "n_err_up": float(ab[2]),
                     "n_err_dw": float(ab[3]), "log_alpha": la_boi,
                     "log_alpha_err_up": float(bb[2]), "log_alpha_err_dw": float(bb[3]),
                     "scatter": float(sb[1])},
        "new_ksl": {"alpha": a_ksl, "n": n_ksl, "n_err_up": float(ak[2]),
                    "n_err_dw": float(ak[3]), "log_alpha": la_ksl,
                    "log_alpha_err_up": float(bk[2]), "log_alpha_err_dw": float(bk[3]),
                    "scatter": float(sk[1])},
        "old_ksl": {"alpha": a_old, "n": n_old},
        "cutoff_threshold": float(threshold),
        "fit_method": "BayesLineFit orthogonal scatter, resolved gas+SFR profiles",
        "n_points": {"boissier": int(len(bx)), "ksl": int(len(kx))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rel, f, indent=2)
    print(f"boissier: alpha={a_boi:.3e} n={n_boi:.3f} | "
          f"new_ksl: alpha={a_ksl:.3e} n={n_ksl:.3f} | cutoff_logSigma={threshold:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
