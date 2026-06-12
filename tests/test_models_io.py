"""IO model engine parity: my engine vs the notebook engine, same physics + r_acc."""

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
import jmfgas.models.inside_out as io

NB = jmfgas.ROOT / "notebooks" / "model_inside_out.ipynb"
_NAMES = ["interp1d_jax", "get_dt_params_from_r_acc", "sigma_acc_jax", "choose_dt_jax",
          "dydt_jax", "RungeKutta_jax", "compute_row_jax", "Sigma_definer_jax",
          "C_def_jax", "simpson_uniform_jax", "full_from_sigma_jax", "Full_final_definer_jax"]


def _notebook_engine():
    nb = json.loads(NB.read_text())
    src = []
    for i in (23, 46, 47, 90):
        s = "".join(nb["cells"][i]["source"])
        for nm in _NAMES:
            m = re.search(rf"(?:^@[^\n]*\n)*^def {re.escape(nm)}\(.*?(?=^\S|\Z)",
                          s, re.DOTALL | re.MULTILINE)
            if m:
                src.append(m.group(0).rstrip())
    ns = dict(np=np, jnp=jnp, jax=jax, lax=lax, partial=partial,
              SFL_jax=SFL_jax, exp_vrot_jax=exp_vrot_jax,
              M_times1_jax=C.M_times1_jax, n_t=C.N_T, t_end=C.T_END, dt=C.DT,
              log_M_bar_array_jax=C.log_M_bar_array_jax)
    exec("\n\n".join(src), ns)
    return ns["Full_final_definer_jax"]


def test_io_engine_matches_notebook():
    nb_full = _notebook_engine()
    worst = 0.0
    for nn, kk in ((0.5, 1.5), (1.0, 2.0)):
        # the notebook engine multiplies r_acc by 1000 (kpc -> pc); mine keeps pc
        r_pc = np.asarray(io.build_r_acc_matrix_for_all_M_jax(nn, kk))
        r_pc_j = jnp.array(r_pc, dtype=jnp.float64)
        r_kpc_j = r_pc_j / 1000.0
        t_acc_arr = jnp.array([3.0, -3.0, 1.0], dtype=jnp.float64)
        for logM in (9.0, 10.0, 11.0):
            Mbar = 10.0**logM
            out_nb = nb_full(Mbar, t_acc_arr, "cutoff_ksl", r_kpc_j, C.log_M_bar_array_jax)
            out_me = io.Full_final_definer_jax(Mbar, t_acc_arr, "cutoff_ksl",
                                               r_pc_j, C.log_M_bar_array_jax)
            for xn, xm in zip(out_nb, out_me):
                xn, xm = np.asarray(xn), np.asarray(xm)
                worst = max(worst, float(np.nanmax(np.abs(xn - xm) / (np.abs(xn) + 1e-30))))
    assert worst < 1e-9, f"worst rel diff {worst:.2e}"
