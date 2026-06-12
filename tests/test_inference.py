"""Likelihood parity vs the notebook + emcee-wrapper sanity (picklable, finite)."""

import json
import pickle
import re

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

import jmfgas
from jmfgas.data import build_mcmc_observables
from jmfgas.models.common import log_M_bar_array_jax
from jmfgas.physics.angmom import j_maxer
import jmfgas.models.inside_out as io
import jmfgas.models.non_inside_out as nio
import jmfgas.inference.likelihoods as L

NB_IO = jmfgas.ROOT / "notebooks" / "model_inside_out.ipynb"
NB_NIO = jmfgas.ROOT / "notebooks" / "model_non_inside_out.ipynb"


def _grab(path, names, cells, glb):
    nb = json.loads(path.read_text())
    src = []
    for i in cells:
        s = "".join(nb["cells"][i]["source"])
        for nm in names:
            m = re.search(rf"(?:^@[^\n]*\n)*^def {re.escape(nm)}\(.*?(?=^\S|\Z)",
                          s, re.DOTALL | re.MULTILINE)
            if m:
                src.append(m.group(0).rstrip())
    ns = dict(glb)
    exec("\n\n".join(src), ns)
    return ns


def test_io_logL_matches_notebook():
    t = build_mcmc_observables()
    J = lambda c: jnp.asarray(t[c].values.astype(float))
    r_pc = io.build_r_acc_matrix_for_all_M_jax(0.5, 1.5)
    ns = _grab(NB_IO, ["logL_jax", "logL_4obs_jax"], [101, 102],
               dict(np=np, jnp=jnp, jax=jax, lax=lax, j_maxer=j_maxer,
                    solve_omega_bisect_autobracket_jax=io.solve_omega_bisect_autobracket_jax,
                    fgas_and_jbar_for_galaxies_jax=io.fgas_and_jbar_for_galaxies_jax,
                    all_obs_for_galaxies_jax=io.all_obs_for_galaxies_jax))
    th = jnp.array([0.5, 1.5])
    a = float(ns["logL_jax"](th, J("logMbar"), J("jbar"), J("fgas"), J("e_fgas"),
                             J("e_jbar"), r_pc, log_M_bar_array_jax, t0=12.0))
    b = float(L.logL_jax(th, J("logMbar"), J("jbar"), J("fgas"), J("e_fgas"),
                         J("e_jbar"), r_pc, log_M_bar_array_jax))
    assert abs(a - b) <= 1e-9 * abs(a)
    args = [J(c) for c in ("logMbar", "jbar", "Mgas", "e_Mgas", "Mstar", "e_Mstar",
                           "jgas", "e_jgas", "jstar", "e_jstar", "e_jbar")]
    a2 = float(ns["logL_4obs_jax"](th, *args, r_pc, log_M_bar_array_jax, t0=12.0))
    b2 = float(L.logL_4obs_jax(th, *args, r_pc, log_M_bar_array_jax))
    assert abs(a2 - b2) <= 1e-9 * abs(a2)


def test_nio_logL_matches_notebook():
    t = build_mcmc_observables()
    A = lambda c: t[c].values.astype(float)[:6]
    ns = _grab(NB_NIO, ["logL_4obs_single_galaxy", "logL_4obs_all_galaxies",
                        "logL_fgas_single_galaxy", "logL_fgas_all_galaxies"], [12, 38],
               dict(np=np, run_single_galaxy_from_jbar=nio.run_single_galaxy_from_jbar,
                    compute_fgas_for_galaxy=L.compute_fgas_for_galaxy))
    obs4 = tuple(A(c) for c in ("logMbar", "jbar", "Mgas", "e_Mgas", "Mstar", "e_Mstar",
                                "jgas", "e_jgas", "jstar", "e_jstar"))
    for a, b in ((1.5, 1.5), (0.0, 2.0)):
        assert abs(ns["logL_4obs_all_galaxies"](a, b, *obs4, "cutoff_ksl")
                   - L.logL_4obs_all_galaxies(a, b, *obs4, "cutoff_ksl")) < 1e-6
        fobs = (A("logMbar"), A("jbar"), A("fgas"), A("e_fgas"))
        assert abs(ns["logL_fgas_all_galaxies"](a, b, *fobs)
                   - L.logL_fgas_all_galaxies(a, b, *fobs)) < 1e-6


def test_wrappers_picklable_and_finite():
    t = build_mcmc_observables()
    J = lambda c: jnp.asarray(t[c].values.astype(float))
    A6 = lambda c: t[c].values.astype(float)[:6]
    obs4_6 = tuple(A6(c) for c in ("logMbar", "jbar", "Mgas", "e_Mgas", "Mstar", "e_Mstar",
                                   "jgas", "e_jgas", "jstar", "e_jstar"))
    iofgas = L.LogProbabilityEmcee(J("logMbar"), J("jbar"), J("fgas"), J("e_fgas"),
                                   J("e_jbar"), log_M_bar_array_jax, (0.1, 2.5), (1.0, 3.0))
    io4 = L.LogProbabilityEmcee4Obs(
        J("logMbar"), J("jbar"), J("Mgas"), J("e_Mgas"), J("Mstar"), J("e_Mstar"),
        J("jgas"), J("e_jgas"), J("jstar"), J("e_jstar"), J("e_jbar"),
        log_M_bar_array_jax, (0.1, 2.5), (1.0, 3.0))
    nio4 = L.NIOPosterior4Obs(obs4_6, "cutoff_ksl", (-1.5, 7.0, -1.5, 5.0))
    nioa0 = L.NIOPosteriorA0(obs4_6, "cutoff_ksl", -1.5, 5.0)
    niofgas = L.NIOPosteriorFgas((A6("logMbar"), A6("jbar"), A6("fgas"), A6("e_fgas")),
                                 (-1.5, 7.0, -1.5, 5.0))
    for w in (iofgas, io4, nio4, nioa0, niofgas):       # loky needs every wrapper picklable
        pickle.loads(pickle.dumps(w))
    # prior rejections (fast, no model eval)
    assert iofgas([5.0, 5.0]) == -np.inf
    assert io4([5.0, 5.0]) == -np.inf
    assert nio4([99.0, 1.5]) == -np.inf
    assert nioa0([99.0]) == -np.inf
    assert niofgas([99.0, 1.5]) == -np.inf
    # finite in-bounds evaluations
    assert np.isfinite(io4([0.5, 1.5]))
    assert np.isfinite(nio4([1.5, 1.5]))
    assert np.isfinite(nioa0([1.5]))
    assert np.isfinite(niofgas([1.5, 1.5]))
