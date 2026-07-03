"""Parity against the actual notebook function sources (executed, not re-typed)."""

import json
import re

import numpy as np
import pytest

import jmfgas
from jmfgas.physics import v_btfr_def, j_maxer, j_acc_def, SFL

NB_IO = jmfgas.ROOT / "notebooks" / "model_inside_out.ipynb"
NB_NIO = jmfgas.ROOT / "notebooks" / "model_non_inside_out.ipynb"
M = np.array([1e8, 3e9, 1e10, 5e10, 2e11])


def _cell_sources(path):
    nb = json.loads(path.read_text())
    return ["".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"]


def _extract_def(path, func):
    """Pull a single top-level `def func(...)` block out of the notebook."""
    for src in _cell_sources(path):
        m = re.search(rf"^def {func}\(.*?(?=^\S|\Z)", src, re.DOTALL | re.MULTILINE)
        if m:
            return m.group(0)
    raise LookupError(f"{func} not found in {path.name}")


def _load(path, func, extra_globals=None):
    ns = {"np": np}
    if extra_globals:
        ns.update(extra_globals)
    exec(_extract_def(path, func), ns)
    return ns[func]


def test_v_btfr_matches_nio_notebook():
    nb_v = _load(NB_NIO, "v_btfr_def")          # NIO default Ag = 10**2.52
    assert np.allclose(v_btfr_def(M), nb_v(M))


def test_v_btfr_matches_io_notebook_with_override():
    nb_v = _load(NB_IO, "v_btfr_def")           # IO default Ag = 47
    assert np.allclose(v_btfr_def(M, A_log10=np.log10(47)), nb_v(M))


def test_j_maxer_matches_notebook():
    nb_j = _load(NB_IO, "j_maxer")
    assert np.allclose(j_maxer(M), nb_j(M))


def test_j_acc_def_matches_notebook():
    nb_jacc = _load(NB_IO, "j_acc_def")
    jm = j_maxer(1e10)
    t = np.linspace(0, 12, 25)
    assert np.allclose(j_acc_def(jm, t, n=0.7, con=1.3, lambda_ratio=0.9),
                       nb_jacc(jm, t, n=0.7, con=1.3, lambda_ratio=0.9))


def test_sfl_cutoff_matches_notebook():
    from jmfgas.physics import Rf, threshold_sigma_SFR
    nb_sfl = _load(NB_IO, "SFL", {"Rf": Rf, "threshold_sigma_SFR": threshold_sigma_SFR})
    sigma = np.logspace(-2, 3, 200)
    # the cutoff coefficients are now re-fit from the same data + method, so they
    # reproduce the notebook's hardcoded law only to within the fitter's MCMC scatter
    assert np.allclose(SFL(sigma, "cutoff_ksl", None, 1e10),
                       nb_sfl(sigma, "cutoff_ksl", None, 1e10), rtol=0.05)
