"""Assemble the observable table and the log-probability for a (model, likelihood)."""

from pathlib import Path

import numpy as np
import pandas as pd
import jax.numpy as jnp

from ..config import load_config
from ..data import sample_frame
from ..models.common import log_M_bar_array_jax
from .likelihoods import (LogProbabilityEmcee, LogProbabilityEmcee4Obs,
                          NIOPosterior4Obs, NIOPosteriorA0, NIOPosteriorFgas)


def obs_table(sample, data_dir, mass_range=None, exclude_hix=False):
    """Observable arrays for a sample, from its cached CSV (data/sample_<name>.csv,
    written by build_sample.py) or built on the fly if that's missing. mass_range=(lo, hi)
    on logMbar and exclude_hix need the Name + Mbar columns (converged / full samples)."""
    csv = Path(data_dir) / f"sample_{sample}.csv"
    df = pd.read_csv(csv) if csv.exists() else sample_frame(sample, data_dir)
    if mass_range is not None or exclude_hix:
        if "Mbar" not in df.columns:
            raise ValueError(f"sample {sample!r} has no Mbar/Name columns for mass/HIX cuts")
        logM = np.log10(df["Mbar"].to_numpy(float))
        keep = np.ones(len(df), bool)
        if mass_range is not None:
            keep &= (logM >= mass_range[0]) & (logM < mass_range[1])
        if exclude_hix:
            hix = set(pd.read_csv(data_dir / "compilation_AM_others" / "HIX.csv")["Name"])
            keep &= ~df["Name"].isin(hix).to_numpy()
        df = df[keep].reset_index(drop=True)
    out = {c: df[c].to_numpy(float) for c in df.columns if c not in ("Name", "group")}
    if "logMbar" not in out and "Mbar" in out:
        out["logMbar"] = np.log10(out["Mbar"])
    return out


def build_log_prob(model, likelihood, sample, cfg, data_dir,
                   mass_range=None, exclude_hix=False):
    """Return (log_prob, ndim, init, bounds) for an MCMC or grid run.

    bounds is a list of (lo, hi) per free parameter.
    """
    if likelihood in ("4obs", "a0") and sample in ("full-hix", "MP_full", "full"):
        raise ValueError(f"the {sample!r} sample carries only baryonic columns; use "
                         "--likelihood fgas (4obs/a0 need the converged or mcmc-obs sample)")
    t = obs_table(sample, data_dir, mass_range, exclude_hix)
    jx = lambda c: jnp.asarray(t[c], dtype=jnp.float64)

    if model == "io":
        nb = cfg["mcmc"]["io"]["bounds"]["n"]
        kb = cfg["mcmc"]["io"]["bounds"]["k"]
        init = list(cfg["mcmc"]["io"]["init"])
        if likelihood == "4obs":
            lp = LogProbabilityEmcee4Obs(
                jx("logMbar"), jx("jbar"), jx("Mgas"), jx("e_Mgas"), jx("Mstar"), jx("e_Mstar"),
                jx("jgas"), jx("e_jgas"), jx("jstar"), jx("e_jstar"), jx("e_jbar"),
                log_M_bar_array_jax, nb, kb)
        elif likelihood == "fgas":
            lp = LogProbabilityEmcee(jx("logMbar"), jx("jbar"), jx("fgas"), jx("e_fgas"),
                                     jx("e_jbar"), log_M_bar_array_jax, nb, kb)
        else:
            raise ValueError("io supports likelihood 4obs or fgas")
        return lp, 2, init, [tuple(nb), tuple(kb)]

    ab = cfg["mcmc"]["nio"]["bounds"]["a"]
    bb = cfg["mcmc"]["nio"]["bounds"]["b"]
    init = list(cfg["mcmc"]["nio"]["init"])
    if likelihood == "fgas":                # fgas needs only jbar + fgas, so the full sample is ok
        obs = (t["logMbar"], t["jbar"], t["fgas"], t["e_fgas"])
        return (NIOPosteriorFgas(obs, (ab[0], ab[1], bb[0], bb[1])),
                2, init, [tuple(ab), tuple(bb)])
    obs4 = tuple(t[c] for c in ("logMbar", "jbar", "Mgas", "e_Mgas", "Mstar", "e_Mstar",
                                "jgas", "e_jgas", "jstar", "e_jstar"))
    if likelihood == "4obs":
        return (NIOPosterior4Obs(obs4, "cutoff_ksl", (ab[0], ab[1], bb[0], bb[1])),
                2, init, [tuple(ab), tuple(bb)])
    if likelihood == "a0":
        return (NIOPosteriorA0(obs4, "cutoff_ksl", bb[0], bb[1]),
                1, [init[1]], [tuple(bb)])
    raise ValueError("nio supports likelihood 4obs, fgas or a0")
