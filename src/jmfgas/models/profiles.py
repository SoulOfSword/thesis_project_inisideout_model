"""Time-resolved radial profiles for both models, plus npz save/load."""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from ..physics.sfl import SFL_jax
from ..physics.kinematics import exp_vrot_jax
from ..physics.radius import r_btfr_def
from .common import M_times1_jax, C_def_jax, simpson_uniform_jax
from .inside_out import Sigma_definer_jax
from .non_inside_out import Sigma_definer_static_racc_jax, omega_Mdep


def _reduce_profiles(SD_gas, SD_sfr, r_pc, M_bar):
    """Shared radial reduction: masses, gas fraction, specific angular momenta vs time."""
    dt = M_times1_jax[1] - M_times1_jax[0]
    dr = r_pc[1] - r_pc[0]
    SD_star = dt * jnp.cumsum(SD_sfr, axis=-1)
    r_col = r_pc[:, None]
    r2_col = r_col**2

    def integ(y):
        return simpson_uniform_jax(y, dr, axis=0)

    M_gas_t = integ(2.0 * jnp.pi * r_col * SD_gas)
    M_sfr_t = integ(2.0 * jnp.pi * r_col * SD_sfr)
    M_star_t = dt * jnp.cumsum(M_sfr_t)
    M_bar_t = M_gas_t + M_star_t
    f_gas_t = M_gas_t / jnp.where(M_bar_t > 0, M_bar_t, jnp.inf)

    v_rot = exp_vrot_jax(r_pc, M_bar)[:, None]
    nom_gas = integ(2.0 * jnp.pi * r2_col * SD_gas * v_rot)
    nom_sfr = integ(2.0 * jnp.pi * r2_col * SD_sfr * v_rot)
    nom_star = dt * jnp.cumsum(nom_sfr)
    nom_bar = nom_star + nom_gas
    j_bar_t = (nom_bar / jnp.where(M_bar_t > 0, M_bar_t, jnp.inf)) / 1000.0
    j_gas_t = (nom_gas / jnp.where(M_gas_t > 0, M_gas_t, jnp.inf)) / 1000.0
    j_star_t = (nom_star / jnp.where(M_star_t > 0, M_star_t, jnp.inf)) / 1000.0
    return {
        "Sigma_star": SD_star, "SFH": M_sfr_t, "M_gas_t": M_gas_t, "M_star_t": M_star_t,
        "f_gas_t": f_gas_t, "j_bar_t": j_bar_t, "j_gas_t": j_gas_t, "j_star_t": j_star_t,
    }


@partial(jax.jit, static_argnames=("sfl_type",))
def radial_profiles_io(log_M_bar, t_acc, r_acc_matrix_pc, log_M_bar_array_jax,
                       sfl_type="cutoff_ksl"):
    """Time-resolved radial profiles for the inside-out model (r_acc matrix in pc)."""
    M_bar = 10.0**log_M_bar
    Mbar_index = jnp.argmin(jnp.abs(log_M_bar_array_jax - log_M_bar))
    r_pc = jnp.arange(0.0, 1000.0 * 100.1, 120, dtype=jnp.float64)
    C = C_def_jax(M_bar, t_acc)
    SD_gas = Sigma_definer_jax(r_pc, t_acc, M_bar, C, sfl_type,
                               r_acc_matrix_pc, log_M_bar_array_jax)
    SD_sfr = SFL_jax(SD_gas, sfl_type, r_pc[:, None], M_bar)
    out = {"log_M_bar": log_M_bar, "t_acc": t_acc, "r_kpc": r_pc / 1000.0,
           "times": M_times1_jax, "Sigma_gas": SD_gas, "Sigma_sfr": SD_sfr,
           "r_acc_t": r_acc_matrix_pc[Mbar_index] / 1000.0}
    out.update(_reduce_profiles(SD_gas, SD_sfr, r_pc, M_bar))
    return out


@partial(jax.jit, static_argnames=("sfl_type",))
def radial_profiles_nio_core(logM, t_acc, r_acc_pc, M_bar, sfl_type="cutoff_ksl"):
    """Time-resolved radial profiles for the non-inside-out model (constant r_acc, pc)."""
    r_pc = jnp.arange(0.0, 1000.0 * 100.1, 120, dtype=jnp.float64)
    C = C_def_jax(M_bar, t_acc)
    SD_gas = Sigma_definer_static_racc_jax(
        r_pc, t_acc, M_bar, C, sfl_type, jnp.array([r_acc_pc], dtype=jnp.float64))[0]
    SD_sfr = SFL_jax(SD_gas, sfl_type, r_pc[:, None], M_bar)
    out = {"r_kpc": r_pc / 1000.0, "times": M_times1_jax,
           "Sigma_gas": SD_gas, "Sigma_sfr": SD_sfr}
    out.update(_reduce_profiles(SD_gas, SD_sfr, r_pc, M_bar))
    return out


def radial_profiles_nio(logM, j_bar_obs, a, b, sfl_type="cutoff_ksl"):
    """NIO profiles from (logM, observed j_bar, a, b): sets j_acc=j_bar, derives r_acc, t_acc."""
    M_bar = 10.0**logM
    omega = float(omega_Mdep(logM, a, b))
    t_acc = 1.0 / omega if abs(omega) > 1e-4 else 1000.0
    r_acc_pc = float(r_btfr_def(M_bar, j_bar_obs))
    prof = radial_profiles_nio_core(logM, t_acc, r_acc_pc, M_bar, sfl_type=sfl_type)
    meta = {"log_M_bar": logM, "t_acc": t_acc, "omega": omega, "j_acc": j_bar_obs,
            "a": a, "b": b, "r_acc_kpc": r_acc_pc / 1000.0}
    return {**meta, **{k: np.array(v) for k, v in prof.items()}}


def save_radial_profiles(profiles, filename):
    """Write a profile dict to .npz."""
    np.savez(filename, **{k: (np.array(v) if hasattr(v, "shape") else v)
                          for k, v in profiles.items()})


def load_radial_profiles(filename):
    """Load a profile .npz into a dict."""
    data = np.load(filename, allow_pickle=True)
    return {k: data[k] for k in data.files}
