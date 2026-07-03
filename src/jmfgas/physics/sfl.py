"""Star formation laws (gas surface density -> SFR surface density)."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from ..config import load_config, load_sfl_relations
from .kinematics import omega, omega_kms_per_pc_jax
from .radius import rv_def

_sfl = load_config()["sfl"]
Rf = _sfl["Rf"]
_rel = load_sfl_relations()
_BOI = (_rel["boissier"]["alpha"], _rel["boissier"]["n"])
_NEW = (_rel["new_ksl"]["alpha"], _rel["new_ksl"]["n"])
_OLD = (_rel["old_ksl"]["alpha"], _rel["old_ksl"]["n"])
threshold_sigma_SFR = _rel["cutoff_threshold"]


def SFL(sigma, sfl_type, R, M_bar, Rf=Rf):
    """SFR surface density for a gas surface density `sigma` under law `sfl_type`."""
    sigma_arr = np.asarray(sigma, dtype=float)
    sigma_pos = np.where(sigma_arr > 0.0, sigma_arr, 0.0)

    if sfl_type == "old_ksl":
        val = (1 - Rf) * _OLD[0] * (sigma_pos**_OLD[1])
    elif sfl_type == "elise_steep":
        val = (1 - Rf) * 2.13e-5 * (sigma_pos**2.47)
    elif sfl_type == "new_ksl":
        val = (1 - Rf) * _NEW[0] * (sigma_pos**_NEW[1])
    elif sfl_type == "boissier":
        val = ((1 - Rf) * _BOI[0] * (sigma_pos**_BOI[1])
               * omega(R, M_bar, rv_def(M_bar)) * 1022.712165)
    elif sfl_type == "kennicutt_modern":
        return (1.0 - Rf) * 6.68e-2 * (sigma**1.54)
    elif sfl_type == "cutoff_ksl":
        with np.errstate(divide="ignore", invalid="ignore"):
            log_sigma = np.where(sigma_pos > 0.0, np.log10(sigma_pos), -np.inf)
        mask = log_sigma < threshold_sigma_SFR
        low_reg = (1 - Rf) * _NEW[0] * (sigma_pos**_NEW[1])
        high_reg = (1 - Rf) * _OLD[0] * (sigma_pos**_OLD[1])
        val = np.where(mask, low_reg, high_reg)
    else:
        raise ValueError(f"Unknown sfl_type: {sfl_type}")

    val = np.where(np.isfinite(val), val, 0.0)
    return float(val) if np.isscalar(sigma) else val


def SFL_jax(sigma, sfl_type, R, M_bar, Rf=Rf):
    """JAX SFR surface density."""
    if sfl_type == "old_ksl":
        return (1.0 - Rf) * _OLD[0] * (sigma**_OLD[1])
    elif sfl_type == "new_ksl":
        return (1.0 - Rf) * _NEW[0] * (sigma**_NEW[1])
    elif sfl_type == "boissier":
        omega_gyr = omega_kms_per_pc_jax(R, M_bar) * 1022.712165
        return (1.0 - Rf) * _BOI[0] * (sigma**_BOI[1]) * omega_gyr
    elif sfl_type == "kennicutt_modern":
        return (1.0 - Rf) * 6.68e-2 * (sigma**1.54)
    elif sfl_type == "cutoff_ksl":
        log_sigma = jnp.log10(jnp.clip(sigma, min=1e-99))
        cond = log_sigma < threshold_sigma_SFR
        low_reg = (1.0 - Rf) * _NEW[0] * (sigma**_NEW[1])
        high_reg = (1.0 - Rf) * _OLD[0] * (sigma**_OLD[1])
        return jnp.where(cond, low_reg, high_reg)
    else:
        raise ValueError(f"Unknown sfl_type '{sfl_type}' in SFL_jax")
