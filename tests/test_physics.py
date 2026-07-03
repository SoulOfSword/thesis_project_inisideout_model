import json

import numpy as np
import pytest
from scipy import optimize

import jmfgas
from jmfgas.physics import (
    v_btfr_def, mbar_from_vflat, j_maxer, j_minner, j_acc_def,
    rv_def, analytical_r, r_btfr_def, r_btfr_def_jax, newton_solve_r_jax,
    SFL, SFL_jax,
)

M = np.array([1e8, 3e9, 1e10, 5e10, 2e11])


def test_v_btfr_matches_relation():
    assert np.allclose(v_btfr_def(M), (M / 10**2.52) ** (1 / 3.58))
    # the inside-out notebook used a different normalisation; the function still supports it
    assert np.allclose(v_btfr_def(M, A_log10=np.log10(47)), (M / 47) ** (1 / 3.58))


def test_mbar_inverse():
    assert np.allclose(mbar_from_vflat(v_btfr_def(M)), M)


def test_j_max_min():
    assert np.allclose(j_maxer(M), (M**0.73) * 10**-4.25)
    assert np.allclose(j_minner(M), j_maxer(M) / 10.0)


def test_j_acc_def():
    jm = j_maxer(1e10)
    t = np.linspace(0, 12, 25)
    for n, con, lam in [(1.0, 1.0, 1.0), (0.5, 1.2, 0.8)]:
        jmin = (jm / 10.0) * lam
        expect = jmin + (con * jm - jmin) * (t / 12.0) ** n
        assert np.allclose(j_acc_def(jm, t, t0=12.0, n=n, con=con, lambda_ratio=lam), expect)


def test_rv_def_uses_fitted_powerlaw():
    rel = json.load(open(jmfgas.ROOT / "data" / "rv_vflat_relation.json"))
    kpc = rv_def(M) / 1000.0
    assert np.allclose(kpc, 10.0**rel["delta"] * (v_btfr_def(M) / 100.0) ** rel["gamma"])
    assert np.all(kpc > 0)  # power law stays positive at all masses


def test_r_btfr_solves_equation():
    jm = j_maxer(1e10)
    j = np.linspace(jm / 10, jm, 10)
    r_pc = r_btfr_def(1e10, j)
    rv_kpc = rv_def(1e10) / 1000.0
    y = j / (2.0 * v_btfr_def(1e10))
    assert np.allclose(analytical_r(r_pc / 1000.0, rv_kpc, y), 0.0, atol=1e-9)


def test_r_btfr_jax_matches_numpy():
    jm = j_maxer(1e10)
    j = np.linspace(jm / 10, jm, 10)
    assert np.allclose(np.asarray(r_btfr_def_jax(1e10, j)), r_btfr_def(1e10, j), rtol=1e-7)


def test_newton_jax_matches_scipy():
    c, y = 2.5, 1.3
    ref = optimize.newton(analytical_r, 1.0, args=(c, y))
    assert np.isclose(float(newton_solve_r_jax(c, y, x0=1.0)), ref, rtol=1e-9)


def test_sfl_cutoff_numpy_matches_jax():
    sigma = np.logspace(-2, 3, 200)
    a = SFL(sigma, "cutoff_ksl", None, 1e10)
    b = np.asarray(SFL_jax(sigma, "cutoff_ksl", None, 1e10))
    assert np.allclose(a, b, rtol=1e-9)


def test_sfl_cutoff_matches_formula():
    from jmfgas.physics import Rf, threshold_sigma_SFR
    rel = json.load(open(jmfgas.ROOT / "data" / "sfl_relations.json"))
    new, old = rel["new_ksl"], rel["old_ksl"]
    sigma = np.logspace(-2, 3, 200)
    mask = np.log10(sigma) < threshold_sigma_SFR
    expect = np.where(mask, (1 - Rf) * new["alpha"] * sigma**new["n"],
                      (1 - Rf) * old["alpha"] * sigma**old["n"])
    assert np.allclose(SFL(sigma, "cutoff_ksl", None, 1e10), expect)
