"""Data pipeline parity vs the canonical CSVs the notebook writes."""

import numpy as np
import pandas as pd
import pytest

import jmfgas
from jmfgas.data import build_converged, build_full, build_mcmc_observables

DATA = jmfgas.ROOT / "data"


def _compare(df_new, csv_path):
    ref = pd.read_csv(csv_path)
    assert list(df_new.columns) == list(ref.columns)
    assert len(df_new) == len(ref)
    worst = 0.0
    for c in ref.columns:
        if not pd.api.types.is_numeric_dtype(ref[c]):
            assert (np.asarray(df_new[c]) == np.asarray(ref[c])).all(), c
        else:
            a = df_new[c].values.astype(float)
            b = ref[c].values.astype(float)
            worst = max(worst, float(np.nanmax(np.abs(a - b) / (np.abs(b) + 1e-30))))
    return worst


def test_converged_matches_common_sample():
    assert _compare(build_converged(), DATA / "common_sample.csv") < 1e-12


def test_mcmc_observables_matches_csv():
    assert _compare(build_mcmc_observables(), DATA / "mcmc_observables.csv") < 1e-12


def test_full_sample_structure():
    full = build_full()
    assert len(full) == 105
    assert ((full["fgas"] > 0) & (full["fgas"] < 1)).all()
    assert (full["Mbar"] > 0).all()
    assert len(build_full(with_hix=True)) == 117


def test_full_corrected_rows_match_converged():
    full = build_full()
    conv = build_converged()[["Name", "Mbar", "jbar", "fgas"]]
    m = full.merge(conv, on="Name", suffixes=("_f", "_c"))
    for c in ("Mbar", "jbar", "fgas"):
        assert np.allclose(m[f"{c}_f"], m[f"{c}_c"], rtol=0, atol=0)
