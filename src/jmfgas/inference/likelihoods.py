"""Likelihoods and picklable emcee log-probability wrappers for both models."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import lax

from ..config import load_config
from ..physics.angmom import j_maxer
from ..physics.radius import r_btfr_def
from ..models.inside_out import (solve_omega_bisect_autobracket_jax,
                                 fgas_and_jbar_for_galaxies_jax,
                                 all_obs_for_galaxies_jax,
                                 build_r_acc_matrix_for_all_M_jax)
from ..models.non_inside_out import (run_single_galaxy_from_jbar,
                                     Full_final_definer_Mdep_omega_jax)

_cfg = load_config()
T0 = _cfg["time"]["t0"]


# ============================ inside-out ============================

@jax.jit
def logL_jax(theta, logM_obs, jbar_obs, fgas_obs, sigma_fgas_obs, sigma_j_obs,
             r_acc_matrix, log_M_bar_array, t0=T0, omega_min=-10.0, omega_max=10.0):
    """f_gas chi2 for all galaxies + j_bar chi2 only for galaxies whose omega is clipped."""
    n, k = theta[0], theta[1]

    def invalid():
        return -jnp.inf

    def body():
        Mbar_obs = 10.0**logM_obs
        j_max = j_maxer(Mbar_obs)
        j_min = j_max / 10.0
        delta_j = jnp.maximum(k * j_max - j_min, 1e-12)
        y_raw = (jbar_obs - j_min) / delta_j

        omega, ok = solve_omega_bisect_autobracket_jax(y_raw, n, t0)
        omega_assigned = jnp.where(ok, omega,
                                   jnp.where(y_raw < 0.0, omega_max, omega_min))
        was_clipped = ((~ok) | (ok & (omega_assigned > omega_max))
                       | (ok & (omega_assigned < omega_min)))
        omega_clipped = jnp.clip(omega_assigned, omega_min, omega_max)
        omega_safe = jnp.where(jnp.abs(omega_clipped) < 1e-4,
                               jnp.sign(omega_clipped) * 1e-4, omega_clipped)
        t_acc = 1.0 / omega_safe

        fgas_mod, jbar_mod = fgas_and_jbar_for_galaxies_jax(
            Mbar_obs, t_acc, "cutoff_ksl", r_acc_matrix, log_M_bar_array, at_t0=True)

        sig_f = jnp.maximum(sigma_fgas_obs, 4e-3)
        logL_f = -0.5 * jnp.sum(((fgas_mod - fgas_obs) / sig_f)**2)
        sig_j = jnp.maximum(sigma_j_obs, 1e-6)
        logL_j = -0.5 * jnp.sum(jnp.where(was_clipped, ((jbar_mod - jbar_obs) / sig_j)**2, 0.0))
        return logL_f + logL_j

    return lax.cond((n <= 0.0) | (k <= 0.0), lambda _: invalid(), lambda _: body(), operand=None)


@jax.jit
def logL_4obs_jax(theta, logM_obs, jbar_obs, Mgas_obs, sigma_Mgas, Mstar_obs, sigma_Mstar,
                  jgas_obs, sigma_jgas, jstar_obs, sigma_jstar, sigma_jbar,
                  r_acc_matrix, log_M_bar_array, t0=T0, omega_min=-10.0, omega_max=10.0):
    """chi2 on (Mgas, Mstar, jgas, jstar); j_bar is already constrained by jgas+jstar."""
    n, k = theta[0], theta[1]

    def invalid():
        return -jnp.inf

    def body():
        Mbar_obs = 10.0**logM_obs
        j_max = j_maxer(Mbar_obs)
        j_min = j_max / 10.0
        delta_j = jnp.maximum(k * j_max - j_min, 1e-12)
        y_raw = (jbar_obs - j_min) / delta_j

        omega, ok = solve_omega_bisect_autobracket_jax(y_raw, n, t0)
        omega_assigned = jnp.where(ok, omega,
                                   jnp.where(y_raw < 0.0, omega_max, omega_min))
        omega_clipped = jnp.clip(omega_assigned, omega_min, omega_max)
        omega_safe = jnp.where(jnp.abs(omega_clipped) < 1e-4,
                               jnp.sign(omega_clipped) * 1e-4, omega_clipped)
        t_acc = 1.0 / omega_safe

        fgas_mod, jbar_mod, jgas_mod, jstar_mod, Mstar_mod, Mgas_mod = \
            all_obs_for_galaxies_jax(Mbar_obs, t_acc, "cutoff_ksl",
                                     r_acc_matrix, log_M_bar_array, at_t0=True)

        sig_Mgas = jnp.maximum(sigma_Mgas, 1e-6)
        sig_Mstar = jnp.maximum(sigma_Mstar, 1e-6)
        sig_jgas = jnp.maximum(sigma_jgas, 1e-6)
        sig_jstar = jnp.maximum(sigma_jstar, 1e-6)
        return (-0.5 * jnp.sum(((Mgas_mod - Mgas_obs) / sig_Mgas)**2)
                - 0.5 * jnp.sum(((Mstar_mod - Mstar_obs) / sig_Mstar)**2)
                - 0.5 * jnp.sum(((jgas_mod - jgas_obs) / sig_jgas)**2)
                - 0.5 * jnp.sum(((jstar_mod - jstar_obs) / sig_jstar)**2))

    return lax.cond((n <= 0.0) | (k <= 0.0), lambda _: invalid(), lambda _: body(), operand=None)


class LogProbabilityEmcee:
    """Picklable emcee log-probability for the inside-out f_gas likelihood."""

    def __init__(self, logM_obs, jbar_obs, fgas_obs, sigma_fgas, sigma_jbar,
                 log_M_bar_array, n_bounds, k_bounds):
        self.logM_obs = logM_obs
        self.jbar_obs = jbar_obs
        self.fgas_obs = fgas_obs
        self.sigma_fgas = sigma_fgas
        self.sigma_jbar = sigma_jbar
        self.log_M_bar_array = log_M_bar_array
        self.n_bounds = tuple(n_bounds)
        self.k_bounds = tuple(k_bounds)

    def __call__(self, theta):
        n, k = theta
        if not (self.n_bounds[0] < n < self.n_bounds[1]
                and self.k_bounds[0] < k < self.k_bounds[1]):
            return -np.inf
        r_acc = build_r_acc_matrix_for_all_M_jax(jnp.float64(n), jnp.float64(k))
        ll = float(logL_jax(jnp.asarray(theta, dtype=jnp.float64),
                            self.logM_obs, self.jbar_obs, self.fgas_obs,
                            self.sigma_fgas, self.sigma_jbar,
                            r_acc, self.log_M_bar_array, t0=T0))
        return ll if np.isfinite(ll) else -np.inf


class LogProbabilityEmcee4Obs:
    """Picklable emcee log-probability for the inside-out 4-observable likelihood."""

    def __init__(self, logM_obs, jbar_obs, Mgas_obs, sigma_Mgas, Mstar_obs, sigma_Mstar,
                 jgas_obs, sigma_jgas, jstar_obs, sigma_jstar, sigma_jbar,
                 log_M_bar_array, n_bounds, k_bounds):
        self.args = (logM_obs, jbar_obs, Mgas_obs, sigma_Mgas, Mstar_obs, sigma_Mstar,
                     jgas_obs, sigma_jgas, jstar_obs, sigma_jstar, sigma_jbar)
        self.log_M_bar_array = log_M_bar_array
        self.n_bounds = tuple(n_bounds)
        self.k_bounds = tuple(k_bounds)

    def __call__(self, theta):
        n, k = theta
        if not (self.n_bounds[0] < n < self.n_bounds[1]
                and self.k_bounds[0] < k < self.k_bounds[1]):
            return -np.inf
        r_acc = build_r_acc_matrix_for_all_M_jax(jnp.float64(n), jnp.float64(k))
        ll = float(logL_4obs_jax(jnp.asarray(theta, dtype=jnp.float64), *self.args,
                                 r_acc, self.log_M_bar_array, t0=T0))
        return ll if np.isfinite(ll) else -np.inf


# ============================ non-inside-out ============================

def logL_4obs_single_galaxy(logM_obs, j_bar_obs, Mgas_obs, sigma_Mgas, Mstar_obs, sigma_Mstar,
                            jgas_obs, sigma_jgas, jstar_obs, sigma_jstar, a, b, sfl_type):
    f_gas, j_bar, j_gas, j_star, M_star, M_gas = run_single_galaxy_from_jbar(
        logM_obs, j_bar_obs, a, b, sfl_type=sfl_type)
    if np.isnan(f_gas) or not np.isfinite(f_gas) or f_gas <= 0 or f_gas >= 1:
        return -1e6
    chi2 = ((M_gas - Mgas_obs) / max(sigma_Mgas, 1e-6))**2
    chi2 += ((M_star - Mstar_obs) / max(sigma_Mstar, 1e-6))**2
    chi2 += ((j_gas - jgas_obs) / max(sigma_jgas, 1e-6))**2
    chi2 += ((j_star - jstar_obs) / max(sigma_jstar, 1e-6))**2
    return -0.5 * chi2


def logL_4obs_all_galaxies(a, b, logM, jbar, Mgas, e_Mgas, Mstar, e_Mstar,
                           jgas, e_jgas, jstar, e_jstar, sfl_type):
    total = 0.0
    for i in range(len(logM)):
        total += logL_4obs_single_galaxy(
            logM[i], jbar[i], Mgas[i], e_Mgas[i], Mstar[i], e_Mstar[i],
            jgas[i], e_jgas[i], jstar[i], e_jstar[i], a, b, sfl_type)
    return total


def log_prior_Mdep_omega(theta, a_min, a_max, b_min, b_max):
    a, b = theta
    return 0.0 if (a_min < a < a_max and b_min < b < b_max) else -np.inf


def compute_fgas_for_galaxy(logM, j_bar_obs, b, a=0.0):
    """Model f_gas for one galaxy (r_btfr_def already returns pc)."""
    Mbar = 10.0**logM
    r_acc_array = jnp.array([float(r_btfr_def(Mbar, j_bar_obs))], dtype=jnp.float64)
    f_gas, _, _, _, _, _ = Full_final_definer_Mdep_omega_jax(
        logM, a, b, r_acc_array, star_formation_law="cutoff_ksl", at_t0=True)
    return float(f_gas[0])


def logL_fgas_single_galaxy(logM_obs, j_bar_obs, fgas_obs, sigma_fgas, b, a=0.0):
    f_mod = compute_fgas_for_galaxy(logM_obs, j_bar_obs, b, a)
    if not np.isfinite(f_mod) or f_mod <= 0 or f_mod >= 1:
        return -1e6
    return -0.5 * ((f_mod - fgas_obs) / max(sigma_fgas, 1e-6))**2


def logL_fgas_all_galaxies(a, b, logM, jbar, fgas, e_fgas):
    return sum(logL_fgas_single_galaxy(logM[i], jbar[i], fgas[i], e_fgas[i], b, a)
               for i in range(len(logM)))


class NIOPosterior4Obs:
    """Picklable (a, b) posterior for the NIO 4-observable likelihood."""

    def __init__(self, obs, sfl_type, bounds):
        self.obs = obs  # (logM,jbar,Mgas,e_Mgas,Mstar,e_Mstar,jgas,e_jgas,jstar,e_jstar)
        self.sfl_type = sfl_type
        self.bounds = tuple(bounds)  # (a_min,a_max,b_min,b_max)

    def __call__(self, theta):
        lp = log_prior_Mdep_omega(theta, *self.bounds)
        if not np.isfinite(lp):
            return -np.inf
        a, b = theta
        return lp + logL_4obs_all_galaxies(a, b, *self.obs, self.sfl_type)


class NIOPosteriorA0:
    """Picklable b-only (a=0) posterior for the NIO 4-observable likelihood."""

    def __init__(self, obs, sfl_type, b_min, b_max):
        self.obs = obs
        self.sfl_type = sfl_type
        self.b_min, self.b_max = b_min, b_max

    def __call__(self, theta):
        b = theta[0]
        if not (self.b_min < b < self.b_max):
            return -np.inf
        return logL_4obs_all_galaxies(0.0, b, *self.obs, self.sfl_type)


class NIOPosteriorFgas:
    """Picklable (a, b) posterior for the NIO f_gas likelihood."""

    def __init__(self, obs, bounds):
        self.obs = obs  # (logM, jbar, fgas, e_fgas)
        self.bounds = tuple(bounds)

    def __call__(self, theta):
        lp = log_prior_Mdep_omega(theta, *self.bounds)
        if not np.isfinite(lp):
            return -np.inf
        a, b = theta
        return lp + logL_fgas_all_galaxies(a, b, *self.obs)
