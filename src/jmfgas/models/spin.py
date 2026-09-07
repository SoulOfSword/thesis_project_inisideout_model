"""Spin-bias model.

Same inside-out engine as the accretion-bias model (it lives in ``common``), but the accretion
history is FIXED by mass — omega = omega_Mdep(M_bar) — and the galaxy-to-galaxy scatter at fixed
M_bar comes from the spin parameter k (the cap on j_acc). j_acc(t) grows as the inside-out law
with n=1; k varies from galaxy to galaxy, capping j_acc(t0)=k*j_max and hence j_bar, at fixed
accretion history. Given an observed j_bar, k is inverted in closed form (at fixed omega, j_bar is
linear in k). Each galaxy carries its own r_acc(t) row into the engine (via the explicit-r_acc
forward), so galaxies that share a mass but differ in j_bar (hence k) are handled correctly.

The tail of this module is the legacy constant-radius engine (the former non-inside-out model),
kept only so the parked 4-obs / f_gas fits still import; it is not used by the spin-bias physics.
"""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import lax

from ..physics.sfl import SFL_jax
from ..physics.angmom import j_maxer, j_acc_def
from ..physics.radius import r_btfr_def, r_btfr_def_jax
from .common import (M_times1_jax, N_T as n_t, T_END as t_end, DT as dt, T0,
                     C_def_jax, full_from_sigma_jax, F_omega_jax,
                     build_r_acc_matrix_for_all_M_jax, all_obs_for_galaxies_jax,
                     all_obs_for_galaxies_explicit_racc_jax)


# ============================ spin-bias model ============================

@jax.jit
def omega_Mdep(logM, a=0.1, b=0.5):
    """Mass-determined accretion rate omega(M_bar) = a*(Mbar/10^10 Msun)^b (fixed history)."""
    return a * (((10**logM) / 1e10) ** b)


@jax.jit
def tacc_Mdep(logM, a, b):
    """Accretion timescale 1/omega (no clamping)."""
    return 1.0 / omega_Mdep(logM, a, b)


@partial(jax.jit, static_argnames=("n",))
def invert_k_jax(logM, jbar, a, b, n=1.0, t0=T0):
    """Spin parameter k that reproduces observed j_bar at fixed omega = omega_Mdep(M_bar).

    With j_max = k*j_MP and j_min = j_max/10 (both scaling with k), the mass-weighted average gives
    j_bar = k*j_MP*(0.1 + 0.9*F(omega)); invert for k. F from F_omega_jax; j_MP = j_maxer(M_bar).
    Analogue of the accretion model's omega inversion (there k is fixed, omega is solved)."""
    j_MP = j_maxer(10.0 ** jnp.asarray(logM, dtype=jnp.float64))
    F = F_omega_jax(omega_Mdep(logM, a, b), n, t0)
    return jbar / (j_MP * (0.1 + 0.9 * F))


@partial(jax.jit, static_argnames=("n",))
def build_r_acc_rows_per_galaxy_jax(logM_array, k_array, n=1.0):
    """r_acc(t) [pc], one row per galaxy, from its own (logM, k). j_acc(t) is the inside-out
    growth law with that galaxy's k; used when k varies galaxy-to-galaxy (the spin residual)."""
    logM_array = jnp.asarray(logM_array, dtype=jnp.float64)
    k_array = jnp.asarray(k_array, dtype=jnp.float64)

    def per_gal(logM, k):
        Mbar = 10.0 ** logM
        j_acc = j_acc_def(j_maxer(Mbar), M_times1_jax, n=n, con=k)
        return r_btfr_def_jax(Mbar, j_acc)

    return jax.vmap(per_gal)(logM_array, k_array)


def spin_obs_for_galaxies(logM_array, jbar_array, a, b, star_formation_law,
                          n=1.0, Rmax=100.1):
    """Spin-model observables per galaxy: omega fixed by mass, k inverted from the observed j_bar,
    then the inside-out engine run forward. Returns (f_gas, j_bar, j_gas, j_star, M_star, M_gas).

    Each galaxy carries its own r_acc row (no nearest-mass argmin), so galaxies that share a mass
    but differ in j_bar (hence k) are handled correctly."""
    logM = jnp.asarray(logM_array, dtype=jnp.float64)
    jbar = jnp.asarray(jbar_array, dtype=jnp.float64)
    k = invert_k_jax(logM, jbar, a, b, n=n)
    r_acc_rows = build_r_acc_rows_per_galaxy_jax(logM, k, n=n)
    Mbar = 10.0 ** logM
    t_acc = tacc_Mdep(logM, a, b)
    return all_obs_for_galaxies_explicit_racc_jax(Mbar, t_acc, star_formation_law, r_acc_rows,
                                                  Rmax=Rmax)


def spin_grids_over_k(logM_grid, a, b, k_grid, star_formation_law, n=1.0, Rmax=100.1):
    """Spin-model observable grids over a k-sweep at fixed omega = omega_Mdep(M) — the band for the
    planes (k is the scatter dimension, the analogue of the accretion model's omega sweep).

    Returns a dict of (n_mass, n_k) arrays: f_gas, j_bar, j_gas, j_star, M_star, M_gas."""
    logM = jnp.asarray(logM_grid, dtype=jnp.float64)
    Mbar = 10.0 ** logM
    t_acc = tacc_Mdep(logM, a, b)
    keys = ("f_gas", "j_bar", "j_gas", "j_star", "M_star", "M_gas")
    cols = {key: [] for key in keys}
    for kval in np.asarray(k_grid, dtype=float):
        r_acc = build_r_acc_matrix_for_all_M_jax(jnp.float64(n), jnp.float64(kval), logM)
        obs = all_obs_for_galaxies_jax(Mbar, t_acc, star_formation_law, r_acc, logM, Rmax=Rmax)
        for key, v in zip(keys, obs):
            cols[key].append(np.asarray(v))
    return {key: np.stack(v, axis=1) for key, v in cols.items()}   # (n_mass, n_k)


# ==================== legacy constant-radius engine (parked fit only) ====================
# The former non-inside-out model: gas accretes at a single constant radius r_acc. Kept so the
# parked 4-obs / f_gas fits (inference/likelihoods, grid, mcmc) still import. NOT used by the
# spin-bias physics above.

def sigma_acc_const_jax(t, r_value, C, t_acc, r_acc_pc):
    return (C / (2.0 * jnp.pi * r_acc_pc**2.0)) * \
           jnp.exp(-t / t_acc) * jnp.exp(-r_value / r_acc_pc)


def choose_dt_const_jax(t, S, r_value, C, t_acc, M_bar, sfl_type, r_acc_pc,
                        base_dt=0.1, dt_min=1e-3, safety=2.0, Rf=0.45):
    S = jnp.asarray(S)
    Sigma_acc = sigma_acc_const_jax(t, r_value, C, t_acc, r_acc_pc)
    Sigma_sfr = SFL_jax(S, sfl_type, r_value, M_bar, Rf=Rf)
    t_sup = jnp.where((S > 0.0) & (Sigma_acc > 0.0), S / Sigma_acc, jnp.inf)
    t_dep = jnp.where((S > 0.0) & (Sigma_sfr > 0.0), S / Sigma_sfr, jnp.inf)
    t_char = jnp.minimum(t_sup, t_dep)
    dt_raw = jnp.minimum(base_dt, t_char / safety)
    dt_raw = jnp.maximum(dt_raw, dt_min)
    return jnp.where(S <= 0.0, base_dt, dt_raw)


def dydt_const_jax(t, S, r_value, C, t_acc, M_bar, sfl_type, r_acc_pc, Rf=0.45):
    Sigma_acc = sigma_acc_const_jax(t, r_value, C, t_acc, r_acc_pc)
    Sigma_sfr = SFL_jax(S, sfl_type, r_value, M_bar, Rf=Rf)
    return Sigma_acc - Sigma_sfr


def RungeKutta_const_jax(t, S, dt, r_value, C, t_acc, M_bar, sfl_type, r_acc_pc, Rf=0.45):
    k1 = dydt_const_jax(t,          S,             r_value, C, t_acc, M_bar, sfl_type, r_acc_pc, Rf=Rf)
    k2 = dydt_const_jax(t + 0.5*dt, S + 0.5*dt*k1, r_value, C, t_acc, M_bar, sfl_type, r_acc_pc, Rf=Rf)
    k3 = dydt_const_jax(t + 0.5*dt, S + 0.5*dt*k2, r_value, C, t_acc, M_bar, sfl_type, r_acc_pc, Rf=Rf)
    k4 = dydt_const_jax(t + dt,     S + dt*k3,     r_value, C, t_acc, M_bar, sfl_type, r_acc_pc, Rf=Rf)
    S_new = S + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return t + dt, jnp.maximum(S_new, 0.0)


@partial(jax.jit, static_argnames=("sfl_type",))
def compute_row_static_racc_jax(r_value, M_bar, C, t_acc, sfl_type, r_acc_pc, Rf=0.45):
    """Sigma_gas(t) at one radius, constant accretion radius, onto the time grid."""
    row0 = jnp.zeros(n_t, dtype=jnp.float64).at[0].set(0.0)
    state0 = (0.0, 0.0, 1, row0)

    def cond_fun(state):
        t, S, save_idx, row = state
        return jnp.logical_and(t < t_end, save_idx < n_t)

    def body_fun(state):
        t, S, save_idx, row = state
        dt_loc = choose_dt_const_jax(t, S, r_value, C, t_acc, M_bar, sfl_type, r_acc_pc,
                                     base_dt=0.1, dt_min=1e-3, safety=2.0, Rf=Rf)
        t_new, S_new = RungeKutta_const_jax(t, S, dt_loc, r_value, C, t_acc, M_bar,
                                            sfl_type, r_acc_pc, Rf=Rf)
        carry = (t, S, t_new, S_new, save_idx, row)

        def save_branch(c):
            t_old, S_old, t_n, S_n, idx, row_i = c
            t_grid = M_times1_jax[idx]
            denom = jnp.maximum(t_n - t_old, 1e-12)
            theta = (t_grid - t_old) / denom
            row_i = row_i.at[idx].set(S_old + theta * (S_n - S_old))
            return (t_n, S_n, t_n, S_n, idx + 1, row_i)

        def nosave_branch(c):
            t_old, S_old, t_n, S_n, idx, row_i = c
            return (t_n, S_n, t_n, S_n, idx, row_i)

        cond_save = jnp.logical_and(save_idx < n_t, t_new >= M_times1_jax[save_idx])
        t2, S2, _t2, _S2, save_idx2, row2 = lax.cond(cond_save, save_branch, nosave_branch, carry)
        return (t2, S2, save_idx2, row2)

    t_f, S_f, save_idx_f, row_f = lax.while_loop(cond_fun, body_fun, state0)

    def fill_rest(row):
        last_val = row[save_idx_f - 1]
        mask = jnp.arange(n_t) >= save_idx_f
        return jnp.where(mask, last_val, row)

    return lax.cond(save_idx_f < n_t, fill_rest, lambda x: x, row_f)


@partial(jax.jit, static_argnames=("sfl_type",))
def Sigma_definer_static_racc_jax(r_pc, t_acc, M_bar, C, sfl_type, r_acc_array_pc, Rf=0.45):
    """Sigma_gas(r_acc, r, t) for a set of constant accretion radii."""
    r_pc = jnp.asarray(r_pc, dtype=jnp.float64)

    def Sigma_for_one_racc(r_acc_pc):
        return jax.vmap(lambda rv: compute_row_static_racc_jax(
            rv, M_bar, C, t_acc, sfl_type, r_acc_pc, Rf=Rf))(r_pc)

    return jax.vmap(Sigma_for_one_racc)(r_acc_array_pc)


@partial(jax.jit, static_argnames=("star_formation_law", "res", "Rmax", "at_t0"))
def Full_final_definer_Mdep_omega_jax(logM, a, b, r_acc_array_pc, star_formation_law,
                                      res=120, Rmax=100.1, at_t0=True):
    """Single galaxy: omega(M)-driven t_acc, constant r_acc grid -> observables."""
    t_acc = tacc_Mdep(logM, a, b)
    Mbar = 10.0**logM
    r_pc = jnp.arange(0.0, 1000.0 * Rmax, res, dtype=jnp.float64)
    C_val = C_def_jax(Mbar, t_acc)
    SD_gas = Sigma_definer_static_racc_jax(r_pc, t_acc, Mbar, C_val,
                                           star_formation_law, r_acc_array_pc)
    return full_from_sigma_jax(SD_gas, Mbar, r_pc, dt, star_formation_law, at_t0=at_t0)


@partial(jax.jit, static_argnames=("star_formation_law", "res", "Rmax", "at_t0"))
def run_all_masses_Mdep_omega_jax(log_M_array, a, b, r_acc_grid_pc,
                                  star_formation_law="kennicutt_modern",
                                  res=120, Rmax=100.1, at_t0=True):
    """Vmap the single-galaxy model over a mass grid."""
    def per_mass(logM, r_acc_row):
        return Full_final_definer_Mdep_omega_jax(
            logM, a, b, r_acc_row, star_formation_law=star_formation_law,
            res=res, Rmax=Rmax, at_t0=at_t0)
    return jax.vmap(per_mass)(log_M_array, r_acc_grid_pc)


def build_r_acc_for_single_M(logM, n_j=10):
    """Accretion radii [pc] over the j_acc range [j_max/10, 2 j_max] for one mass (legacy band)."""
    Mbar = 10.0**logM
    j_max = j_maxer(Mbar)
    j_acc_array = np.linspace(j_max / 10.0, 2.0 * j_max, n_j)
    r_acc_pc = r_btfr_def(Mbar, j_acc_array)
    return r_acc_pc, j_acc_array


def run_single_galaxy_Mdep_omega_jax(logM, a, b, sfl_type, n_j=10):
    """Convenience: build r_acc grid for a mass and run the model."""
    r_acc_pc, j_acc_array = build_r_acc_for_single_M(logM, n_j)
    r_acc_jax = jnp.array(r_acc_pc, dtype=jnp.float64)
    f_gas, j_bar, j_gas, j_star, Ms, Mg = Full_final_definer_Mdep_omega_jax(
        logM, a, b, r_acc_jax, star_formation_law=sfl_type)
    return np.array(f_gas), np.array(j_bar), j_acc_array


def run_single_galaxy_from_jbar(logM, j_bar_obs, a, b, sfl_type):
    """Run the model with j_acc set to the observed j_bar; returns 6 scalar observables."""
    Mbar = 10.0**logM
    try:
        r_acc_pc = float(r_btfr_def(Mbar, j_bar_obs))
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    r_acc_array = jnp.array([r_acc_pc], dtype=jnp.float64)
    f_gas, j_bar, j_gas, j_star, M_star, M_gas = Full_final_definer_Mdep_omega_jax(
        logM, a, b, r_acc_array, star_formation_law=sfl_type)
    return (float(np.array(f_gas)[0]), float(np.array(j_bar)[0]), float(np.array(j_gas)[0]),
            float(np.array(j_star)[0]), float(np.array(M_star)[0]), float(np.array(M_gas)[0]))
