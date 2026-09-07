"""Shared model engine: time grid, normalisation, integration, Sigma -> observables."""

from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import lax

from ..config import load_config
from ..physics.sfl import SFL_jax
from ..physics.kinematics import exp_vrot_jax
from ..physics.angmom import j_maxer, j_acc_def
from ..physics.radius import r_btfr_def, r_btfr_def_jax

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


# ======================= inside-out engine (shared) =======================
# Time-varying accretion radius r_acc(t) from a growing j_acc(t). Both models use it:
# the accretion-bias model scatters omega (t_acc) at fixed (n, k); the spin-bias model
# scatters k at fixed omega = omega_Mdep(M_bar). Moved from the former inside_out.py.

n_t = N_T
t_end = T_END
dt = DT


def build_r_acc_matrix_for_all_M(ns, ks, lambda_ratio=1.0):
    """r_acc(M_bar, t) [pc] on the mass and time grids (inside-out: j_acc grows with t)."""
    r_acc_matrix = np.zeros((len(log_M_bar_array), len(M_times1)), dtype=float)
    for i, logM in enumerate(log_M_bar_array):
        Mbar = 10.0**logM
        j_acc = j_acc_def(j_maxer(Mbar), M_times1, n=ns, con=ks, lambda_ratio=lambda_ratio)
        r_acc_matrix[i, :] = r_btfr_def(np.full_like(j_acc, Mbar), j_acc)
    return r_acc_matrix


@jax.jit
def build_r_acc_matrix_for_all_M_jax(ns, ks, log_M_bar_arr=log_M_bar_array_jax,
                                     lambda_ratio=1.0):
    """JAX r_acc(M_bar, t) [pc] on a mass grid (default: the config grid) and the time grid.

    Pass log_M_bar_arr to build r_acc on a custom mass grid (e.g. extended past the config
    cap so the massive end isn't snapped to the edge bin); default callers are unaffected.
    lambda_ratio scales j_min for a halo of non-median spin (1.0 leaves j_acc unchanged)."""
    Mbar = 10.0 ** jnp.asarray(log_M_bar_arr, dtype=jnp.float64)

    def per_mass(Mbar_val):
        j_acc = j_acc_def(j_maxer(Mbar_val), M_times1_jax, n=ns, con=ks,
                          lambda_ratio=lambda_ratio)
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


def RungeKutta_jax(t, S, dt_step, r_value, C, t_acc, M_bar, sfl_type, r_acc_vec):
    k1 = dydt_jax(t,               S,                  r_value, C, t_acc, M_bar, sfl_type, r_acc_vec)
    k2 = dydt_jax(t + 0.5*dt_step, S + 0.5*dt_step*k1, r_value, C, t_acc, M_bar, sfl_type, r_acc_vec)
    k3 = dydt_jax(t + 0.5*dt_step, S + 0.5*dt_step*k2, r_value, C, t_acc, M_bar, sfl_type, r_acc_vec)
    k4 = dydt_jax(t + dt_step,     S + dt_step*k3,     r_value, C, t_acc, M_bar, sfl_type, r_acc_vec)
    S_new = S + (dt_step / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return t + dt_step, jnp.maximum(S_new, 0.0)


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


@partial(jax.jit, static_argnames=("star_formation_law", "res", "Rmax"))
def run_all_masses(Mbar_grid, t_acc_arr, r_acc_matrix_for_all_M_jax,
                   log_M_bar_array_jax, star_formation_law, res=120, Rmax=100.1):
    """Vmap the full model over a mass grid (shared t_acc array)."""
    def per_mass(Mbar):
        return Full_final_definer_jax(Mbar, t_acc_arr, star_formation_law,
                                      r_acc_matrix_for_all_M_jax, log_M_bar_array_jax,
                                      res=res, Rmax=Rmax, at_t0=True)
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


@partial(jax.jit, static_argnames=("star_formation_law", "res", "Rmax", "at_t0"))
def all_obs_for_galaxies_explicit_racc_jax(Mbar_array, t_acc_array, star_formation_law,
                                           r_acc_rows, res=120, Rmax=100.1, at_t0=True):
    """All 6 observables per galaxy, each carrying its OWN r_acc(t) row directly — no shared mass
    grid, no nearest-mass argmin. Use when r_acc differs galaxy-to-galaxy at possibly-equal masses
    (variable torques: per-galaxy k). Each galaxy is fed a 1-row r_acc matrix and a matching 1-mass grid,
    so the engine's argmin trivially selects that galaxy's own row."""
    def per_gal(Mbar, t_acc, r_acc_vec):
        f_gas, j_bar, j_gas, j_star, M_star, M_gas = Full_final_definer_jax(
            Mbar, jnp.array([t_acc], dtype=jnp.float64), star_formation_law,
            r_acc_vec[None, :], jnp.array([jnp.log10(Mbar)], dtype=jnp.float64),
            res=res, Rmax=Rmax, at_t0=at_t0)
        return f_gas[0], j_bar[0], j_gas[0], j_star[0], M_star[0], M_gas[0]
    return jax.vmap(per_gal)(Mbar_array, t_acc_array, r_acc_rows)


# --- j_bar shape function F(omega) = A(z,n)/B(z), z = omega*t0 (both models use it) ---
# A(z,n) = int_0^1 x^n e^{-z x} dx as a strictly-positive Taylor series (no cancellation),
# B(z) = int_0^1 e^{-z x} dx closed-form. Machine-precise for the |z| the inversions need.


@partial(jax.jit, static_argnames=("K",))
def A_integral(z, n, K=256):
    """A(z) = int_0^1 x^n e^{-z x} dx as a positive-term Taylor series.

    z >= 0:  e^{-z} sum_k z^k / (n+1)_{k+1}        z < 0:  sum_k (-z)^k / (k! (n+1+k)).
    Each branch sums strictly positive terms, so both reach ~machine precision; the off-branch
    is fed 0 to stay finite. K=256 gives ~1e-16 truncation out to |z|~150 (omega-cap is |z|=120).
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    n = jnp.asarray(n, dtype=jnp.float64)
    k = jnp.arange(1, K, dtype=jnp.float64)
    ones = jnp.ones(jnp.shape(z) + (1,), dtype=jnp.float64)
    zp = jnp.where(z >= 0.0, z, 0.0)
    cp = jnp.cumprod(zp[..., None] / (n + 1.0 + k), axis=-1)
    A_pos = jnp.exp(-zp) * jnp.sum(jnp.concatenate([ones, cp], axis=-1), axis=-1) / (n + 1.0)
    s = jnp.where(z < 0.0, -z, 0.0)
    cn = jnp.cumprod((s[..., None] / k) * (n + k) / (n + 1.0 + k), axis=-1)
    A_neg = jnp.sum(jnp.concatenate([ones, cn], axis=-1), axis=-1) / (n + 1.0)
    return jnp.where(z >= 0.0, A_pos, A_neg)


@jax.jit
def B_integral(z):
    """B(z) = int_0^1 e^{-z x} dx = (1 - e^{-z})/z, exact; B(0)=1."""
    z = jnp.asarray(z, dtype=jnp.float64)
    safe = jnp.where(z == 0.0, 1.0, z)
    return jnp.where(z == 0.0, jnp.ones_like(safe), -jnp.expm1(-z) / safe)


@partial(jax.jit, static_argnames=("K",))
def F_omega_jax(omega, n, t0, K=256):
    """Dimensionless F(omega) = A(omega t0, n) / B(omega t0) on [0,1]."""
    z = jnp.asarray(omega, dtype=jnp.float64) * t0
    return A_integral(z, n, K) / B_integral(z)
