"""Inside-out model: time-varying accretion radius."""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import lax

from ..physics.sfl import SFL_jax
from ..physics.angmom import j_maxer, j_acc_def
from ..physics.radius import r_btfr_def, r_btfr_def_jax
from .common import (M_times1, M_times1_jax, N_T as n_t, T_END as t_end, DT as dt,
                     log_M_bar_array, log_M_bar_array_jax,
                     C_def_jax, interp1d_jax, simpson_uniform_jax,
                     full_from_sigma_jax, full_from_sigma_jax_mcmc)


def build_r_acc_matrix_for_all_M(ns, ks):
    """r_acc(M_bar, t) [pc] on the mass and time grids (inside-out: j_acc grows with t)."""
    r_acc_matrix = np.zeros((len(log_M_bar_array), len(M_times1)), dtype=float)
    for i, logM in enumerate(log_M_bar_array):
        Mbar = 10.0**logM
        j_acc = j_acc_def(j_maxer(Mbar), M_times1, n=ns, con=ks)
        r_acc_matrix[i, :] = r_btfr_def(np.full_like(j_acc, Mbar), j_acc)
    return r_acc_matrix


@jax.jit
def build_r_acc_matrix_for_all_M_jax(ns, ks):
    """JAX r_acc(M_bar, t) [pc] on the mass and time grids."""
    Mbar = 10.0 ** log_M_bar_array_jax

    def per_mass(Mbar_val):
        j_acc = j_acc_def(j_maxer(Mbar_val), M_times1_jax, n=ns, con=ks)
        return r_btfr_def_jax(Mbar_val, j_acc)

    return jax.vmap(per_mass)(Mbar)


def get_dt_params_from_r_acc(r_acc_vec, times, t_acc=None):
    """Adaptive timestep parameters from how fast r_acc changes and the t_acc scale."""
    dr_rel = (r_acc_vec[1] - r_acc_vec[0]) / jnp.maximum(r_acc_vec[0], 1e-10)
    dt_early = times[1] - times[0]
    rate = jnp.abs(dr_rel) / jnp.maximum(dt_early, 1e-10)
    scale_r = 1.0 / (1.0 + (rate / 5.0) ** 1.5)
    base_dt = 0.01 + 0.09 * scale_r
    dt_min = 1e-4 + (1e-3 - 1e-4) * scale_r
    safety = 5.0 - 3.0 * scale_r
    if t_acc is not None:
        abs_t_acc = jnp.abs(t_acc)
        base_dt = jnp.maximum(jnp.minimum(base_dt, 0.5 * abs_t_acc), 1e-5)
        dt_min = jnp.maximum(jnp.minimum(dt_min, 0.05 * abs_t_acc), 1e-7)
    return base_dt, dt_min, safety


def sigma_acc_jax(t, r_value, C, t_acc, r_acc_vec):
    r_acc = interp1d_jax(M_times1_jax, r_acc_vec, t)   # pc
    return (C / (2.0 * jnp.pi * r_acc**2.0)) * jnp.exp(-t / t_acc) * jnp.exp(-r_value / r_acc)


def choose_dt_jax(t, S, r_value, C, t_acc, M_bar, sfl_type, r_acc_vec,
                  base_dt=0.1, dt_min=1e-3, safety=2.0):
    S = jnp.asarray(S)
    Sigma_acc = sigma_acc_jax(t, r_value, C, t_acc, r_acc_vec)
    Sigma_sfr = SFL_jax(S, sfl_type, r_value, M_bar)
    t_sup = jnp.where((S > 0.0) & (Sigma_acc > 0.0), S / Sigma_acc, jnp.inf)
    t_dep = jnp.where((S > 0.0) & (Sigma_sfr > 0.0), S / Sigma_sfr, jnp.inf)
    t_char = jnp.minimum(t_sup, t_dep)
    dt_raw = jnp.maximum(jnp.minimum(base_dt, t_char / safety), dt_min)
    return jnp.where(S <= 0.0, base_dt, dt_raw)


def dydt_jax(t, S, r_value, C, t_acc, M_bar, sfl_type, r_acc_vec):
    return (sigma_acc_jax(t, r_value, C, t_acc, r_acc_vec)
            - SFL_jax(S, sfl_type, r_value, M_bar))


def RungeKutta_jax(t, S, dt, r_value, C, t_acc, M_bar, sfl_type, r_acc_vec):
    k1 = dydt_jax(t,          S,             r_value, C, t_acc, M_bar, sfl_type, r_acc_vec)
    k2 = dydt_jax(t + 0.5*dt, S + 0.5*dt*k1, r_value, C, t_acc, M_bar, sfl_type, r_acc_vec)
    k3 = dydt_jax(t + 0.5*dt, S + 0.5*dt*k2, r_value, C, t_acc, M_bar, sfl_type, r_acc_vec)
    k4 = dydt_jax(t + dt,     S + dt*k3,     r_value, C, t_acc, M_bar, sfl_type, r_acc_vec)
    S_new = S + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return t + dt, jnp.maximum(S_new, 0.0)


@partial(jax.jit, static_argnames=("sfl_type",))
def compute_row_jax(r_value, Mbar_index, M_bar, C, t_acc, sfl_type, r_acc_matrix_for_all_M):
    """Sigma_gas(t) at one radius with a time-varying accretion radius."""
    r_acc_vec = r_acc_matrix_for_all_M[Mbar_index]
    base_dt, dt_min, safety = get_dt_params_from_r_acc(r_acc_vec, M_times1_jax, t_acc=t_acc)
    row0 = jnp.zeros(n_t, dtype=jnp.float64).at[0].set(0.0)
    state0 = (0.0, 0.0, 1, row0)

    def cond_fun(state):
        t, S, save_idx, row = state
        return jnp.logical_and(t < t_end, save_idx < n_t)

    def body_fun(state):
        t, S, save_idx, row = state
        dt_loc = choose_dt_jax(t, S, r_value, C, t_acc, M_bar, sfl_type, r_acc_vec,
                               base_dt=base_dt, dt_min=dt_min, safety=safety)
        t_new, S_new = RungeKutta_jax(t, S, dt_loc, r_value, C, t_acc, M_bar, sfl_type, r_acc_vec)
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
def Sigma_definer_jax(r, t_acc, M_bar, C, sfl_type, r_acc_matrix_for_all_M, log_M_bar_array):
    """Sigma_gas(r, t) for one (mass, t_acc)."""
    r = jnp.asarray(r, dtype=jnp.float64)
    Mbar_index = jnp.argmin(jnp.abs(log_M_bar_array - jnp.log10(M_bar)))

    def solve_at_radius(r_value):
        return compute_row_jax(r_value, Mbar_index, M_bar, C, t_acc, sfl_type,
                               r_acc_matrix_for_all_M)

    return jax.vmap(solve_at_radius)(r)


@partial(jax.jit, static_argnames=("star_formation_law", "res", "Rmax", "at_t0"))
def Full_final_definer_jax(Mbar, t_acc_arr, star_formation_law,
                           r_acc_matrix_for_all_M_jax, log_M_bar_array_jax,
                           res=120, Rmax=100.1, at_t0=True):
    """Full observables for one mass over an array of t_acc."""
    r_pc = jnp.arange(0.0, 1000.0 * Rmax, res, dtype=jnp.float64)
    C_vals = C_def_jax(Mbar, t_acc_arr)

    def sigma_for_one_tacc(t_acc_single, C_single):
        return Sigma_definer_jax(r_pc, t_acc_single, Mbar, C_single, star_formation_law,
                                 r_acc_matrix_for_all_M_jax, log_M_bar_array_jax)

    SD_gas = jax.vmap(sigma_for_one_tacc, in_axes=(0, 0))(t_acc_arr, C_vals)
    return full_from_sigma_jax(SD_gas, Mbar, r_pc, dt, star_formation_law, at_t0=at_t0)


@partial(jax.jit, static_argnames=("star_formation_law", "res", "Rmax", "at_t0"))
def Full_final_definer_jax_mcmc(Mbar, t_acc_arr, star_formation_law,
                                r_acc_matrix_for_all_M_jax, log_M_bar_array_jax,
                                res=120, Rmax=100.1, at_t0=True):
    """f_gas-only path for one mass over an array of t_acc."""
    r_pc = jnp.arange(0.0, 1000.0 * Rmax, res, dtype=jnp.float64)
    C_vals = C_def_jax(Mbar, t_acc_arr)

    def sigma_for_one_tacc(t_acc_single, C_single):
        return Sigma_definer_jax(r_pc, t_acc_single, Mbar, C_single, star_formation_law,
                                 r_acc_matrix_for_all_M_jax, log_M_bar_array_jax)

    SD_gas = jax.vmap(sigma_for_one_tacc, in_axes=(0, 0))(t_acc_arr, C_vals)
    return full_from_sigma_jax_mcmc(SD_gas, Mbar, r_pc, dt, star_formation_law, at_t0=at_t0)


@partial(jax.jit, static_argnames=("star_formation_law",))
def run_all_masses(Mbar_grid, t_acc_arr, r_acc_matrix_for_all_M_jax,
                   log_M_bar_array_jax, star_formation_law):
    """Vmap the full model over a mass grid (shared t_acc array)."""
    def per_mass(Mbar):
        return Full_final_definer_jax(Mbar, t_acc_arr, star_formation_law,
                                      r_acc_matrix_for_all_M_jax, log_M_bar_array_jax, at_t0=True)
    return jax.vmap(per_mass)(Mbar_grid)


@partial(jax.jit, static_argnames=("star_formation_law", "res", "Rmax", "at_t0"))
def all_obs_for_galaxies_jax(Mbar_array, t_acc_array, star_formation_law,
                             r_acc_matrix_for_all_M_jax, log_M_bar_array_jax,
                             res=120, Rmax=100.1, at_t0=True):
    """All 6 observables per galaxy, each with its own (Mbar, t_acc)."""
    def per_gal(Mbar, t_acc):
        f_gas, j_bar, j_gas, j_star, M_star, M_gas = Full_final_definer_jax(
            Mbar, jnp.array([t_acc], dtype=jnp.float64), star_formation_law,
            r_acc_matrix_for_all_M_jax, log_M_bar_array_jax, res=res, Rmax=Rmax, at_t0=at_t0)
        return f_gas[0], j_bar[0], j_gas[0], j_star[0], M_star[0], M_gas[0]
    return jax.vmap(per_gal)(Mbar_array, t_acc_array)


@partial(jax.jit, static_argnames=("star_formation_law", "res", "Rmax", "at_t0"))
def fgas_and_jbar_for_galaxies_jax(Mbar_array, t_acc_array, star_formation_law,
                                   r_acc_matrix_for_all_M_jax, log_M_bar_array_jax,
                                   res=120, Rmax=100.1, at_t0=True):
    """(f_gas, j_bar) per galaxy."""
    def per_gal(Mbar, t_acc):
        f_gas, j_bar, _, _, _, _ = Full_final_definer_jax(
            Mbar, jnp.array([t_acc], dtype=jnp.float64), star_formation_law,
            r_acc_matrix_for_all_M_jax, log_M_bar_array_jax, res=res, Rmax=Rmax, at_t0=at_t0)
        return f_gas[0], j_bar[0]
    return jax.vmap(per_gal)(Mbar_array, t_acc_array)


# --- omega inversion (j_bar -> omega) for the inside-out likelihood ---
x_grid_jax = jnp.linspace(0.0, 1.0, 256, dtype=jnp.float64)
dx_x = x_grid_jax[1] - x_grid_jax[0]


@jax.jit
def incomplete_gamma_integral_jax(omega, n, t0, upper_limit=1.0):
    omega = jnp.asarray(omega)
    x = x_grid_jax[None, :] * upper_limit
    dx = dx_x * upper_limit
    integrand = (x**n) * jnp.exp(-omega[..., None] * (t0 * x))
    return simpson_uniform_jax(integrand, dx, axis=-1)


@jax.jit
def F_omega_jax(omega, n, t0):
    """Dimensionless F(omega) = int x^n e^{-w t0 x} / int e^{-w t0 x} on [0,1]."""
    num = incomplete_gamma_integral_jax(omega, n, t0, upper_limit=1.0)
    den = incomplete_gamma_integral_jax(omega, 0.0, t0, upper_limit=1.0)
    return num / den


@jax.jit
def solve_omega_bisect_autobracket_jax(y_target, n, t0, omega0=1.0, max_expand=60, n_iter=40):
    """Auto-bracket + bisection for the monotone-decreasing F(omega). Returns (omega, ok)."""
    y_target = jnp.asarray(y_target)
    lo = jnp.full_like(y_target, -omega0)
    hi = jnp.full_like(y_target, omega0)

    def expand_step(i, state):
        lo, hi = state
        need_lo = F_omega_jax(lo, n, t0) < y_target
        need_hi = F_omega_jax(hi, n, t0) > y_target
        return (jnp.where(need_lo, lo * 1.5, lo), jnp.where(need_hi, hi * 1.5, hi))

    lo, hi = lax.fori_loop(0, max_expand, expand_step, (lo, hi))
    F_lo = F_omega_jax(lo, n, t0)
    F_hi = F_omega_jax(hi, n, t0)
    ok = (jnp.isfinite(F_lo) & jnp.isfinite(F_hi) & (F_lo >= y_target) & (F_hi <= y_target))
    lo = jnp.where(ok, lo, jnp.nan)
    hi = jnp.where(ok, hi, jnp.nan)

    def bisect_step(i, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        go_right = F_omega_jax(mid, n, t0) > y_target
        return (jnp.where(go_right, mid, lo), jnp.where(go_right, hi, mid))

    lo, hi = lax.fori_loop(0, n_iter, bisect_step, (lo, hi))
    return 0.5 * (lo + hi), ok
