"""Radial-profile parity for both models vs the notebooks."""

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
from jmfgas.models import common as C
import jmfgas.models.profiles as prof
import jmfgas.models.non_inside_out as nio
import jmfgas.models.inside_out as io

NB_IO = jmfgas.ROOT / "notebooks" / "model_inside_out.ipynb"
NB_NIO = jmfgas.ROOT / "notebooks" / "model_non_inside_out.ipynb"
_BASE = dict(np=np, jnp=jnp, jax=jax, lax=lax, partial=partial,
             SFL_jax=SFL_jax, exp_vrot_jax=exp_vrot_jax,
             M_times1_jax=C.M_times1_jax, n_t=C.N_T, t_end=C.T_END, dt=C.DT)


def _exec(nbpath, names, cells, extra=None):
    nb = json.loads(nbpath.read_text())
    src = []
    for i in cells:
        s = "".join(nb["cells"][i]["source"])
        for nm in names:
            m = re.search(rf"(?:^@[^\n]*\n)*^def {re.escape(nm)}\(.*?(?=^\S|\Z)",
                          s, re.DOTALL | re.MULTILINE)
            if m:
                src.append(m.group(0).rstrip())
    ns = dict(_BASE)
    if extra:
        ns.update(extra)
    exec("\n\n".join(src), ns)
    return ns


def test_nio_profiles_match_notebook():
    names = ["sigma_acc_const_jax", "choose_dt_const_jax", "dydt_const_jax",
             "RungeKutta_const_jax", "compute_row_static_racc_jax",
             "Sigma_definer_static_racc_jax", "C_def_jax", "simpson_uniform_jax",
             "_compute_radial_profiles_nio_jax"]
    ns = _exec(NB_NIO, names, (9, 10, 42))
    worst = 0.0
    for logM in (9.5, 10.5):
        r_acc, _ = nio.build_r_acc_for_single_M(logM, n_j=10)
        r_acc_pc, Mbar = float(r_acc[5]), 10.0**logM
        t_acc = 1.0 / (0.5 * (logM - 10) + 2.0)
        pn = ns["_compute_radial_profiles_nio_jax"](logM, t_acc, r_acc_pc, Mbar, sfl_type="cutoff_ksl")
        pm = prof.radial_profiles_nio_core(logM, t_acc, r_acc_pc, Mbar, sfl_type="cutoff_ksl")
        for k in pn:
            if k in pm:
                xn, xm = np.asarray(pn[k]), np.asarray(pm[k])
                worst = max(worst, float(np.nanmax(np.abs(xn - xm) / (np.abs(xn) + 1e-30))))
    assert worst < 1e-9, f"{worst:.2e}"


def test_io_profiles_match_notebook():
    names = ["interp1d_jax", "get_dt_params_from_r_acc", "sigma_acc_jax", "choose_dt_jax",
             "dydt_jax", "RungeKutta_jax", "compute_row_jax", "Sigma_definer_jax",
             "C_def_jax", "simpson_uniform_jax", "compute_radial_profiles_jax"]
    ns = _exec(NB_IO, names, (23, 46, 47, 90, 134),
               extra={"log_M_bar_array_jax": C.log_M_bar_array_jax})
    r_pc = np.asarray(io.build_r_acc_matrix_for_all_M_jax(0.5, 1.5))
    r_pc_j = jnp.array(r_pc)
    r_kpc_j = r_pc_j / 1000.0
    worst = 0.0
    for logM in (9.5, 10.5):
        pn = ns["compute_radial_profiles_jax"](logM, 3.0, r_kpc_j, C.log_M_bar_array_jax)
        pm = prof.radial_profiles_io(logM, 3.0, r_pc_j, C.log_M_bar_array_jax)
        for k in ("Sigma_gas", "Sigma_sfr", "Sigma_star", "SFH", "M_gas_t", "M_star_t",
                  "f_gas_t", "j_bar_t", "j_gas_t", "j_star_t", "r_kpc", "times"):
            xn, xm = np.asarray(pn[k]), np.asarray(pm[k])
            worst = max(worst, float(np.nanmax(np.abs(xn - xm) / (np.abs(xn) + 1e-30))))
    assert worst < 1e-9, f"{worst:.2e}"
