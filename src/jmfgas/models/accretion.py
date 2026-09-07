"""Accretion-bias model.

Inside-out growth (the engine lives in ``common``): j_acc(t) has a fixed shape (n, k) and the
galaxy-to-galaxy scatter at fixed M_bar comes from the accretion rate omega (t_acc = 1/omega),
which is inverted per galaxy from its observed j_bar. Physics is identical to the former
inside-out model; only the name changed. The engine (r_acc builder, Sigma integrator, forward
observables, F(omega)) is re-exported here so callers can import it from this module.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax

from .common import (build_r_acc_matrix_for_all_M, build_r_acc_matrix_for_all_M_jax,
                     Sigma_definer_jax, Full_final_definer_jax, Full_final_definer_jax_mcmc,
                     run_all_masses, all_obs_for_galaxies_jax, fgas_and_jbar_for_galaxies_jax,
                     A_integral, B_integral, F_omega_jax)


@jax.jit
def solve_omega_bisect_autobracket_jax(y_target, n, t0, omega0=1.0, max_expand=60, n_iter=40):
    """Auto-bracket + bisection for the monotone-decreasing F(omega). Returns (omega, ok).

    Inverts the accretion rate: given y = (j_bar - j_min)/(k j_max - j_min) = F(omega), find omega.
    """
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
