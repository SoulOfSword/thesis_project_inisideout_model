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

    # Pavel+2025 stellar update where available
    P25 = pd.read_csv(data_dir / "pavel2025_stellar.csv")
    for name, row in P25[P25["Name"].isin(df["Name"])].set_index("Name").iterrows():
        mask = df["Name"] == name
        df.loc[mask, "Mstar"] = 10**row["logMstar"]
        df.loc[mask, "e_Mstar"] = 10**row["logMstar"] * np.log(10) * row["e_logMstar"]
        df.loc[mask, "jstar"] = row["jstar"]
        df.loc[mask, "e_jstar"] = row["e_jstar"]

    # H2 (Geesink+2025) + helium x1.4 on the gas component
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
