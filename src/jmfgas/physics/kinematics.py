"""Rotation curve and angular frequency (all radii in pc)."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from .btfr import v_btfr_def
from .radius import rv_def


def exp_vrot(r_pc, M_bar):
    """Exponential-disc rotation curve [km/s] at radius r_pc."""
    rv = rv_def(M_bar)
    return v_btfr_def(M_bar) * (1.0 - np.exp(-r_pc / rv))


def exp_vrot_jax(r_pc, M_bar):
    """JAX exponential-disc rotation curve [km/s] at radius r_pc."""
    rv = rv_def(M_bar)
    return v_btfr_def(M_bar) * (1.0 - jnp.exp(-r_pc / rv))


def omega(R_pc, M_bar, rv):
    """Angular frequency v_rot/R [(km/s)/pc]; uses v_rot/rv at R=0."""
    v0 = v_btfr_def(M_bar)
    if hasattr(R_pc, "__len__"):
        out = [v0 / rv if r == 0 else (v0 * (1.0 - np.exp(-r / rv))) / r for r in R_pc]
        return np.array(out)
    return v0 / rv if R_pc == 0 else (v0 * (1.0 - np.exp(-R_pc / rv))) / R_pc


@jax.jit
def omega_kms_per_pc_jax(R_pc, M_bar):
    """JAX angular frequency v_rot/R [(km/s)/pc]."""
    R_pc = jnp.asarray(R_pc)
    v0 = v_btfr_def(M_bar)
    rv = rv_def(M_bar)
    vR = exp_vrot_jax(R_pc, M_bar)
    return jnp.where(R_pc == 0.0, v0 / rv, vR / R_pc)
