"""Build the observed galaxy samples with the stellar / H2 / helium corrections."""

import re

import numpy as np
import pandas as pd

from ..config import ROOT, load_config

_DATA = ROOT / load_config()["paths"]["data"]

_BARY_COLS = ["Name", "Mbar", "e_Mbar", "jbar", "e_jbar", "fgas", "e_fgas"]


def _normalize_name(name):
    """NGC0253 -> NGC253, 'IC 1954' -> IC1954."""
    name = name.strip().replace(" ", "")
    m = re.match(r"^([A-Za-z]+)0*(\d+.*)", name)
    return (m.group(1).upper() + m.group(2)) if m else name.upper()


def _apply_pavel_stellar(df, data_dir):
    """Pavel+2025 stellar update (M_star, j_star + errors) in place, where a galaxy is listed."""
    P25 = pd.read_csv(data_dir / "pavel2025_stellar.csv")
    for name, row in P25[P25["Name"].isin(df["Name"])].set_index("Name").iterrows():
        mask = df["Name"] == name
        df.loc[mask, "Mstar"] = 10**row["logMstar"]
        df.loc[mask, "e_Mstar"] = 10**row["logMstar"] * np.log(10) * row["e_logMstar"]
        df.loc[mask, "jstar"] = row["jstar"]
        df.loc[mask, "e_jstar"] = row["e_jstar"]
    return df


def _apply_gas_corrections(df, data_dir):
    """H2 (Geesink+2025) + helium x1.4 on df's gas columns in place. Input Mgas/jgas are the HI
    values; output are the corrected total cold gas (HI+H2, x1.4 helium)."""
    M_HI = df["Mgas"].values.astype(float).copy()
    e_M_HI = df["e_Mgas"].values.astype(float).copy()
    j_HI = df["jgas"].values.astype(float).copy()
    e_j_HI = df["e_jgas"].values.astype(float).copy()

    G25 = pd.read_csv(data_dir / "geesink2025_tableA1.csv")
    g25_norm = {_normalize_name(n): i for i, n in enumerate(G25["Name"])}
    M_H2 = np.zeros(len(df)); e_M_H2 = np.zeros(len(df))
    j_H2 = np.zeros(len(df)); e_j_H2 = np.zeros(len(df))
    for i, cn in enumerate([_normalize_name(n) for n in df["Name"]]):
        if cn in g25_norm:
            row = G25.iloc[g25_norm[cn]]
            M_H2[i] = 10**row["log_MH2"]
            e_M_H2[i] = M_H2[i] * np.log(10) * row["log_MH2_err"]
            j_H2[i] = 10**row["log_jH2"]
            e_j_H2[i] = j_H2[i] * np.log(10) * row["log_jH2_err"]

    D = M_HI + M_H2
    df["Mgas"] = 1.4 * D
    df["e_Mgas"] = 1.4 * np.sqrt(e_M_HI**2 + e_M_H2**2)
    jgas_new = (M_HI * j_HI + M_H2 * j_H2) / D
    df["jgas"] = jgas_new
    df["e_jgas"] = (1.0 / D) * np.sqrt(
        ((j_HI - jgas_new) * e_M_HI)**2 + (M_HI * e_j_HI)**2
        + ((j_H2 - jgas_new) * e_M_H2)**2 + (M_H2 * e_j_H2)**2)
    return df


def build_converged(data_dir=_DATA):
    """The 77 galaxies in baryons ∩ gas ∩ stars, with all corrections applied."""
    BARY = pd.read_csv(data_dir / "baryons1.csv")
    GAS = pd.read_csv(data_dir / "gasses.csv")
    STAR = pd.read_csv(data_dir / "stars.csv")

    common = set(BARY["Name"]) & set(GAS["Name"]) & set(STAR["Name"])
    b = BARY[BARY["Name"].isin(common)][["Name", "Mass(Msun)", "e_Mass(Msun)",
                                         "j", "e_j", "fgas", "e_fgas"]].copy()
    b.columns = _BARY_COLS
    g = GAS[GAS["Name"].isin(common)][["Name", "Mass", "e_Mass", "j", "e_j"]].copy()
    g.columns = ["Name", "Mgas", "e_Mgas", "jgas", "e_jgas"]
    s = STAR[STAR["Name"].isin(common)][["Name", "Mass", "e_Mass", "j", "e_j"]].copy()
    s.columns = ["Name", "Mstar", "e_Mstar", "jstar", "e_jstar"]
    df = b.merge(g, on="Name").merge(s, on="Name")

    _apply_pavel_stellar(df, data_dir)      # Pavel+2025 M_star/j_star update where available
    _apply_gas_corrections(df, data_dir)    # H2 (Geesink+2025) + helium x1.4 on the gas

    # recompute Mbar, fgas, jbar + errors
    Mgas = df["Mgas"].values.astype(float); e_Mgas = df["e_Mgas"].values.astype(float)
    Mstar = df["Mstar"].values.astype(float); e_Mstar = df["e_Mstar"].values.astype(float)
    jgas = df["jgas"].values.astype(float); e_jgas = df["e_jgas"].values.astype(float)
    jstar = df["jstar"].values.astype(float); e_jstar = df["e_jstar"].values.astype(float)

    Mbar = Mgas + Mstar
    e_Mbar = np.sqrt(e_Mgas**2 + e_Mstar**2)
    fgas = Mgas / Mbar
    e_fgas = np.sqrt((Mstar * e_Mgas)**2 + (Mgas * e_Mstar)**2) / Mbar**2
    jbar = (Mgas * jgas + Mstar * jstar) / Mbar
    e_jbar = (1.0 / Mbar) * np.sqrt(
        (jgas * e_Mgas)**2 + (Mgas * e_jgas)**2
        + (jstar * e_Mstar)**2 + (Mstar * e_jstar)**2 + (jbar * e_Mbar)**2)

    df["Mbar"] = Mbar; df["e_Mbar"] = e_Mbar
    df["fgas"] = fgas; df["e_fgas"] = e_fgas
    df["jbar"] = jbar; df["e_jbar"] = e_jbar
    return df


def _attach_fgas(df, data_dir):
    """f_gas/e_fgas per galaxy: the recomputed corrected value where a full (gas AND star)
    decomposition exists, else the original baryonic-catalogue value (can't recompute without
    both components)."""
    conv = build_converged(data_dir).set_index("Name")
    bary = pd.read_csv(data_dir / "baryons1.csv").set_index("Name")
    pick = lambda name, col: (float(conv.at[name, col]) if name in conv.index
                              else float(bary.at[name, col]))
    df["fgas"] = [pick(n, "fgas") for n in df["Name"]]
    df["e_fgas"] = [pick(n, "e_fgas") for n in df["Name"]]
    return df


def build_stellar(data_dir=_DATA):
    """All baryons ∩ stars galaxies for the stellar plane (M_star, j_star + f_gas colour).

    M_star/j_star use the Pavel+2025 update where available, else the original stars.csv.
    f_gas is recomputed+corrected where a full decomposition exists, else the catalogue value.
    Wider than the converged sample: keeps galaxies with stars but no gas counterpart."""
    BARY = pd.read_csv(data_dir / "baryons1.csv")
    STAR = pd.read_csv(data_dir / "stars.csv")
    names = set(BARY["Name"]) & set(STAR["Name"])
    s = STAR[STAR["Name"].isin(names)][["Name", "Mass", "e_Mass", "j", "e_j"]].copy()
    s.columns = ["Name", "Mstar", "e_Mstar", "jstar", "e_jstar"]
    _apply_pavel_stellar(s, data_dir)
    _attach_fgas(s, data_dir)
    return s.reset_index(drop=True)


def build_gaseous(data_dir=_DATA):
    """All baryons ∩ gas galaxies for the gaseous plane (M_gas, j_gas + f_gas colour).

    M_gas/j_gas carry the H2 (Geesink) + helium correction where H2 is available, else HI +
    helium. f_gas as in build_stellar. Keeps galaxies with gas but no star counterpart."""
    BARY = pd.read_csv(data_dir / "baryons1.csv")
    GAS = pd.read_csv(data_dir / "gasses.csv")
    names = set(BARY["Name"]) & set(GAS["Name"])
    g = GAS[GAS["Name"].isin(names)][["Name", "Mass", "e_Mass", "j", "e_j"]].copy()
    g.columns = ["Name", "Mgas", "e_Mgas", "jgas", "e_jgas"]
    _apply_gas_corrections(g, data_dir)
    _attach_fgas(g, data_dir)
    return g.reset_index(drop=True)


def build_full(data_dir=_DATA, with_hix=False):
    """All baryons galaxies: those in gas ∩ stars corrected, the rest kept raw."""
    conv = build_converged(data_dir)[_BARY_COLS]
    BARY = pd.read_csv(data_dir / "baryons1.csv")
    extra = [n for n in BARY["Name"] if n not in set(conv["Name"])]
    raw = BARY[BARY["Name"].isin(extra)][["Name", "Mass(Msun)", "e_Mass(Msun)",
                                          "j", "e_j", "fgas", "e_fgas"]].copy()
    raw.columns = _BARY_COLS
    full = pd.concat([conv, raw], ignore_index=True)
    if with_hix:
        conv_full = build_converged(data_dir)
        hix = _hix_arrays(conv_full, data_dir)
        hrows = pd.DataFrame({c: hix[c] for c in _BARY_COLS})
        full = pd.concat([full, hrows], ignore_index=True)
    return full


def _median_per_bin(arr, logMs, bins):
    out = np.full(len(bins) - 1, np.nan)
    for i in range(len(bins) - 1):
        m = (logMs >= bins[i]) & (logMs < bins[i + 1])
        if m.sum() > 0:
            out[i] = np.nanmedian(arr[m])
    valid = np.where(~np.isnan(out))[0]
    centers = 0.5 * (bins[:-1] + bins[1:])
    for i in range(len(out)):
        if np.isnan(out[i]) and len(valid) > 0:
            out[i] = out[valid[np.argmin(np.abs(centers[valid] - centers[i]))]]
    return out


def _assign_err(logM_arr, medians, bins):
    out = np.zeros(len(logM_arr))
    for i, lm in enumerate(logM_arr):
        idx = int(np.clip(np.searchsorted(bins, lm, side="right") - 1, 0, len(medians) - 1))
        out[i] = medians[idx]
    return out


def _hix_arrays(converged, data_dir):
    """HIX observables with errors imputed from CONVERGED medians per 0.5-dex bin."""
    HIX = pd.read_csv(data_dir / "compilation_AM_others" / "HIX.csv")
    bins = np.arange(8, 12, 0.5)
    logM_conv = np.log10(converged["Mbar"].values.astype(float))
    med = {c: _median_per_bin(converged[c].values.astype(float), logM_conv, bins)
           for c in ["e_jbar", "e_Mgas", "e_Mstar", "e_jgas", "e_jstar", "e_fgas"]}
    lm = HIX["logMbar"].to_numpy(float)
    return {
        "Name": HIX["Name"].to_numpy(),
        "logMbar": lm, "Mbar": 10**lm,
        "Mgas": 10**HIX["logMgas"].to_numpy(float), "Mstar": 10**HIX["logMstar"].to_numpy(float),
        "jbar": 10**HIX["logjbar"].to_numpy(float),
        "jgas": HIX["jgas"].to_numpy(float), "jstar": HIX["jstar"].to_numpy(float),
        "fgas": HIX["fgas"].to_numpy(float),
        "e_jbar": _assign_err(lm, med["e_jbar"], bins),
        "e_Mbar": np.sqrt(_assign_err(lm, med["e_Mgas"], bins)**2
                          + _assign_err(lm, med["e_Mstar"], bins)**2),
        "e_Mgas": _assign_err(lm, med["e_Mgas"], bins),
        "e_Mstar": _assign_err(lm, med["e_Mstar"], bins),
        "e_jgas": _assign_err(lm, med["e_jgas"], bins),
        "e_jstar": _assign_err(lm, med["e_jstar"], bins),
        "e_fgas": _assign_err(lm, med["e_fgas"], bins),
    }


def build_mcmc_observables(data_dir=_DATA):
    """CONVERGED + HIX as the 4-observable MCMC table (logMbar / masses / momenta / fgas)."""
    conv = build_converged(data_dir)
    hix = _hix_arrays(conv, data_dir)
    cols = ["jbar", "e_jbar", "Mgas", "e_Mgas", "Mstar", "e_Mstar",
            "jgas", "e_jgas", "jstar", "e_jstar", "fgas", "e_fgas"]
    out = {"logMbar": np.concatenate([np.log10(conv["Mbar"].values.astype(float)),
                                      hix["logMbar"]])}
    for c in cols:
        out[c] = np.concatenate([conv[c].values.astype(float), hix[c]])
    return pd.DataFrame(out)


# ---- the full real sample: MP+21 (BARY) + HIX + extra compilations -------------
# Galaxies without measured errors get a per-mass-bin median RELATIVE error from the
# MP+21 sample (Mbar, jbar and fgas relative errors are imputed independently).
_REL_BINS = np.arange(7.75, 12.25, 0.5)            # mass bins centred on 8.0, 8.5, ...
# every compilation sample beyond MP+21, across both dirs (the two Dwarfs files are
# disjoint galaxy sets). These have only baryonic data, so they enter the f_gas fit only.
_EXTRA_CSVS = [
    "compilation_AM_others/HIX.csv",
    "compilation_AM_others/Dwarfs.csv",
    "compilation_AM_others/superspirals.csv",
    "compilation_AM_others/superthin.csv",
    "compilation_AM_others_notused/Dwarfs.csv",
    "compilation_AM_others_notused/GLSBs.csv",
    "compilation_AM_others_notused/UDGs.csv",
]


def _rel_err_medians(mp_df, bins=_REL_BINS):
    """Per-bin median relative error (|e_x / x|) of the MP+21 sample, for x in Mbar/jbar/fgas."""
    logM = np.log10(mp_df["Mbar"].to_numpy(float))
    return {x: _median_per_bin(np.abs(mp_df[f"e_{x}"].to_numpy(float)
                                      / mp_df[x].to_numpy(float)), logM, bins)
            for x in ("Mbar", "jbar", "fgas")}


def _impute_bary_rows(csv_path, rel, bins=_REL_BINS):
    """Read a (logMbar, logjbar, fgas) compilation and impute its errors from `rel`."""
    df = pd.read_csv(csv_path)
    logM = df["logMbar"].to_numpy(float)
    Mbar = 10.0**logM
    jbar = 10.0**df["logjbar"].to_numpy(float)
    fgas = df["fgas"].to_numpy(float)
    namecol = next((c for c in ("Name", "name", "GALAXY") if c in df.columns), None)
    name = (df[namecol].astype(str).to_numpy() if namecol is not None
            else np.array([f"{csv_path.stem}{i}" for i in range(len(df))]))
    return pd.DataFrame({
        "Name": name, "Mbar": Mbar, "jbar": jbar, "fgas": fgas,
        "e_Mbar": Mbar * _assign_err(logM, rel["Mbar"], bins),
        "e_jbar": jbar * _assign_err(logM, rel["jbar"], bins),
        "e_fgas": fgas * _assign_err(logM, rel["fgas"], bins),
    })[_BARY_COLS]


def build_all(data_dir=_DATA):
    """MP+21 (BARY) + every compilation (HIX, both Dwarfs, superspirals, superthin, GLSBs,
    UDGs), baryonic columns only.

    MP+21 keeps its measured errors; the compilations get per-bin median relative errors
    imputed from MP+21. Only j_bar/f_gas (+ M_bar) are defined for every galaxy, so this
    sample is for the f_gas likelihood / baryonic plane, not the 4-observable fit.
    """
    mp = build_full(data_dir, with_hix=False)
    rel = _rel_err_medians(mp)
    mp = mp[_BARY_COLS].copy()
    mp["group"] = "MP+21b"
    rows = [mp]
    for c in _EXTRA_CSVS:                       # tag each galaxy with its source (for plot markers)
        r = _impute_bary_rows(data_dir / c, rel)
        r["group"] = (data_dir / c).stem        # HIX, Dwarfs, superspirals, superthin, GLSBs, UDGs
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def sample_frame(sample, data_dir=_DATA):
    """Single resolver: sample name -> the built DataFrame (what gets cached to CSV)."""
    if sample == "converged":
        return build_converged(data_dir)
    if sample == "stellar":
        return build_stellar(data_dir)
    if sample == "gaseous":
        return build_gaseous(data_dir)
    if sample == "mcmc-obs":
        return build_mcmc_observables(data_dir)
    if sample == "MP_full":
        return build_full(data_dir, with_hix=False)
    if sample == "full-hix":                       # MP+21 + HIX (absolute errors; legacy)
        return build_full(data_dir, with_hix=True)
    if sample == "full":
        return build_all(data_dir)
    raise ValueError(f"unknown sample {sample!r}")
