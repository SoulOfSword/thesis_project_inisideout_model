"""Shared model engine: time grid, normalisation, integration, Sigma -> observables."""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from ..config import load_config
from ..physics.sfl import SFL_jax
from ..physics.kinematics import exp_vrot_jax

_cfg = load_config()
T0 = _cfg["time"]["t0"]
DT = _cfg["time"]["dt"]

# time grid: 0 .. t0+dt in steps of dt (one step past t0, as in the reference model)
M_times1 = np.arange(0.0, T0 + 2 * DT, DT)
M_times1_jax = jnp.array(M_times1, dtype=jnp.float64)
N_T = M_times1_jax.shape[0]
T_END = float(M_times1[-1])

_mg = _cfg["mass_grid"]
log_M_bar_array = np.linspace(_mg["logM_min"], _mg["logM_max"], _mg["n"])
log_M_bar_array_jax = jnp.array(log_M_bar_array, dtype=jnp.float64)


def C_def_jax(M_bar, t_acc, t0=T0):
    """Accretion normalisation C so the integrated baryonic mass reaches M_bar at t0."""
    t_acc = jnp.asarray(t_acc, dtype=jnp.float64)
    pos = t_acc > 0.0
    infmask = jnp.isinf(t_acc)
    C_pos = M_bar / (t_acc * (1.0 - jnp.exp(-t0 / t_acc)))
    abs_t = jnp.abs(t_acc)
    C_neg = M_bar / (abs_t * (jnp.exp(t0 / abs_t) - 1.0))
    C_inf = M_bar / t0
    C = jnp.where(pos, C_pos, C_neg)
    C = jnp.where(infmask, C_inf, C)
    return C


def interp1d_jax(x_grid, y_grid, x):
    """Linear interpolation on a sorted 1D grid, edge-clamped."""
    x_grid = jnp.asarray(x_grid)
    y_grid = jnp.asarray(y_grid)
    idx = jnp.searchsorted(x_grid, x, side="right") - 1
    idx = jnp.clip(idx, 0, x_grid.size - 2)
    x0, x1 = x_grid[idx], x_grid[idx + 1]
    y0, y1 = y_grid[idx], y_grid[idx + 1]
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


def simpson_uniform_jax(y, dx, axis=0):
    """Simpson integration with uniform spacing; trapezoid on the last interval if needed."""
    y = jnp.asarray(y)
    y = jnp.moveaxis(y, axis, 0)
    N = y.shape[0]
    if N < 2:
        return jnp.zeros_like(y[0])

    def simpson_all(y_local):
        w = jnp.ones(N)
        w = w.at[1:N - 1:2].set(4.0)
        w = w.at[2:N - 1:2].set(2.0)
        return dx / 3.0 * jnp.tensordot(w, y_local, axes=(0, 0))

    def simpson_plus_trap(y_local):
        Nsim = N - 1
        y_s = y_local[:Nsim]
        w = jnp.ones(Nsim)
        w = w.at[1:Nsim - 1:2].set(4.0)
        w = w.at[2:Nsim - 1:2].set(2.0)
        simp = dx / 3.0 * jnp.tensordot(w, y_s, axes=(0, 0))
        trap = dx * 0.5 * (y_local[-2] + y_local[-1])
        return simp + trap

    result = jax.lax.cond((N % 2 == 1), simpson_all, simpson_plus_trap, y)
    return jnp.moveaxis(result, 0, axis)


@partial(jax.jit, static_argnames=("star_formation_law", "at_t0"))
def full_from_sigma_jax(SD_gas, Mbar, r_pc, dt, star_formation_law, at_t0=True):
    """Reduce Sigma_gas(batch, r, t) to (f_gas, j_bar, j_gas, j_star, M_star, M_gas)."""
    SD_gas = jnp.asarray(SD_gas)
    r_pc = jnp.asarray(r_pc)
    dr = r_pc[1] - r_pc[0]
    n_tacc, n_r, n_t = SD_gas.shape
    r_col = r_pc[:, None]
    r2_col = r_col**2

    R_broad = r_col[None, :, :]
    Sigma_sfr = SFL_jax(SD_gas, star_formation_law, R_broad, Mbar)

    integrand_gas = r_col[None, :, :] * SD_gas
    integrand_sfr = r_col[None, :, :] * Sigma_sfr

    def integrate_radius(y):
        return 2.0 * jnp.pi * simpson_uniform_jax(y, dr, axis=0)

    M_gas = jax.vmap(integrate_radius)(integrand_gas)
    M_sfr = jax.vmap(integrate_radius)(integrand_sfr)
    M_star = dt * jnp.cumsum(M_sfr, axis=-1)
    M_bar_tot = M_star + M_gas
    f_gas_global = M_gas / M_bar_tot

    v_rot = exp_vrot_jax(r_pc, Mbar)[:, None]
    integrand_gas_j = r2_col[None, :, :] * SD_gas * v_rot[None, :, :]
    integrand_sfr_j = r2_col[None, :, :] * Sigma_sfr * v_rot[None, :, :]
    nom_gas = jax.vmap(integrate_radius)(integrand_gas_j)
    nom_sfr = jax.vmap(integrate_radius)(integrand_sfr_j)
    nom_star = dt * jnp.cumsum(nom_sfr, axis=-1)
    nom_bar = nom_star + nom_gas

    j_bar = nom_bar / jnp.where(M_bar_tot > 0, M_bar_tot, jnp.inf)
    j_gas = nom_gas / jnp.where(M_gas > 0, M_gas, jnp.inf)
    j_star = nom_star / jnp.where(M_star > 0, M_star, jnp.inf)

    if at_t0:
        return (f_gas_global[:, -1], j_bar[:, -1] / 1000.0, j_gas[:, -1] / 1000.0,
                j_star[:, -1] / 1000.0, M_star[:, -1], M_gas[:, -1])
    return (f_gas_global, j_bar / 1000.0, j_gas / 1000.0, j_star / 1000.0, M_star, M_gas)


@partial(jax.jit, static_argnames=("star_formation_law", "at_t0"))
def full_from_sigma_jax_mcmc(SD_gas, Mbar, r_pc, dt, star_formation_law, at_t0=True):
    """f_gas-only reduction (cheaper path for the f_gas likelihood)."""
    SD_gas = jnp.asarray(SD_gas)
    r_pc = jnp.asarray(r_pc)
    dr = r_pc[1] - r_pc[0]
    n_tacc, n_r, n_t = SD_gas.shape
    r_col = r_pc[:, None]

    R_broad = r_col[None, :, :]
    Sigma_sfr = SFL_jax(SD_gas, star_formation_law, R_broad, Mbar)
    integrand_gas = r_col[None, :, :] * SD_gas
    integrand_sfr = r_col[None, :, :] * Sigma_sfr

    def integrate_radius(y):
        return 2.0 * jnp.pi * simpson_uniform_jax(y, dr, axis=0)

    M_gas = jax.vmap(integrate_radius)(integrand_gas)
    M_sfr = jax.vmap(integrate_radius)(integrand_sfr)
    M_star = dt * jnp.cumsum(M_sfr, axis=-1)
    f_gas_global = M_gas / (M_star + M_gas)
    return f_gas_global[:, -1] if at_t0 else f_gas_global
