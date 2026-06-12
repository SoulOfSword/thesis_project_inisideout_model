"""NIO model engine parity: my engine vs the notebook engine, same physics + r_acc."""

import json
import re

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax
from functools import partial

import jmfgas
from jmfgas.physics.sfl import SFL_jax
from jmfgas.physics.kinematics import exp_vrot_jax
from jmfgas.physics.angmom import j_maxer
from jmfgas.physics.radius import r_btfr_def
from jmfgas.models import common as C
import jmfgas.models.non_inside_out as nio

NB = jmfgas.ROOT / "notebooks" / "model_non_inside_out.ipynb"
_NAMES = ["omega_Mdep", "tacc_Mdep", "sigma_acc_const_jax", "choose_dt_const_jax",
          "dydt_const_jax", "RungeKutta_const_jax", "compute_row_static_racc_jax",
          "Sigma_definer_static_racc_jax", "C_def_jax", "simpson_uniform_jax",
          "full_from_sigma_jax", "Full_final_definer_Mdep_omega_jax"]


def _notebook_engine():
    nb = json.loads(NB.read_text())
    src = []
    for i in (9, 10, 11):
        s = "".join(nb["cells"][i]["source"])
        for nm in _NAMES:
            m = re.search(rf"(?:^@[^\n]*\n)*^def {re.escape(nm)}\(.*?(?=^\S|\Z)",
                          s, re.DOTALL | re.MULTILINE)
            if m:
                src.append(m.group(0).rstrip())
    ns = dict(np=np, jnp=jnp, jax=jax, lax=lax, partial=partial,
              SFL_jax=SFL_jax, exp_vrot_jax=exp_vrot_jax, j_maxer=j_maxer, r_btfr_def=r_btfr_def,
              M_times1_jax=C.M_times1_jax, n_t=C.N_T, t_end=C.T_END, dt=C.DT)
    exec("\n\n".join(src), ns)
    return ns["Full_final_definer_Mdep_omega_jax"]


def test_nio_engine_matches_notebook():
    nb_full = _notebook_engine()
    worst = 0.0
    for logM in (9.0, 10.0, 10.5, 11.0):
        for a, b in ((0.0, 2.0), (1.5, 1.5), (-0.5, 3.0)):
            r_acc, _ = nio.build_r_acc_for_single_M(logM, n_j=10)
            r_acc_jax = jnp.array(r_acc, dtype=jnp.float64)
            out_nb = nb_full(logM, a, b, r_acc_jax, star_formation_law="cutoff_ksl")
            out_me = nio.Full_final_definer_Mdep_omega_jax(
                logM, a, b, r_acc_jax, star_formation_law="cutoff_ksl")
            for xn, xm in zip(out_nb, out_me):
                xn, xm = np.asarray(xn), np.asarray(xm)
                worst = max(worst, float(np.nanmax(np.abs(xn - xm) / (np.abs(xn) + 1e-30))))
    assert worst < 1e-9, f"worst rel diff {worst:.2e}"


def test_nio_engine_all_sfl_laws():
    """The shared reducer must pass the SFL name through (not hardcode cutoff_ksl)."""
    nb_full = _notebook_engine()
    r_acc, _ = nio.build_r_acc_for_single_M(10.0, n_j=10)
    r_acc_jax = jnp.array(r_acc, dtype=jnp.float64)
    for law in ("cutoff_ksl", "kennicutt_modern", "new_ksl", "old_ksl"):
        out_nb = nb_full(10.0, 1.5, 1.5, r_acc_jax, star_formation_law=law)
        out_me = nio.Full_final_definer_Mdep_omega_jax(
            10.0, 1.5, 1.5, r_acc_jax, star_formation_law=law)
        for xn, xm in zip(out_nb, out_me):
            xn, xm = np.asarray(xn), np.asarray(xm)
            assert np.allclose(xn, xm, rtol=1e-9, equal_nan=True), law
