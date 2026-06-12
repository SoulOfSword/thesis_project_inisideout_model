"""Disc scale radius R_v and the accretion radius r_acc."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy import optimize
import jax.numpy as jnp

from ..config import load_rv_relation
from .btfr import v_btfr_def

_rv_cache = {}


def _rv_params():
    """Lazily load and cache (alpha, beta) of the R_v-v_flat power law (kpc)."""
    if "p" not in _rv_cache:
        rel = load_rv_relation()
        _rv_cache["p"] = (rel["alpha"], rel["beta"])
    return _rv_cache["p"]


def rv_def(M_bar):
    """Disc scale radius R_v [pc] from the fitted power law (divide by 1000 for kpc)."""
    alpha, beta = _rv_params()
    return (10.0**alpha * v_btfr_def(M_bar) ** beta) * 1000.0


def analytical_r(x, c, y):
    return x - (c**3 * x) / (c + x) ** 3 - y


def r_btfr_def(M_bar, j_acc, init=1.0):
    """Accretion radius r_acc [pc] for a given accreted specific angular momentum.

    Solved in kpc (R_v and j_acc/2v are both kpc) and returned in pc.
    """
    rv_kpc = rv_def(M_bar) / 1000.0
    y = j_acc / (2.0 * v_btfr_def(M_bar))
    if hasattr(y, "__len__"):
        out = [optimize.newton(analytical_r, init, args=(rv_kpc[i], y[i]))
               if hasattr(rv_kpc, "__len__") else
               optimize.newton(analytical_r, init, args=(rv_kpc, y[i]))
               for i in range(len(y))]
        return np.array(out) * 1000.0
    return optimize.newton(analytical_r, init, args=(rv_kpc, y)) * 1000.0


def _analytical_r_jax(x, c, y):
    return x - (c**3 * x) / (c + x) ** 3 - y


def _analytical_r_prime_jax(x, c):
    return 1.0 - (c**3 * (c - 2.0 * x)) / (c + x) ** 4


def newton_solve_r_jax(c, y, x0=1.0, n_iter=30):
    """Vectorisable Newton solve for the accretion radius (single c, array/scalar y)."""
    def body(_, x):
        f = _analytical_r_jax(x, c, y)
        fp = _analytical_r_prime_jax(x, c)
        fp = jnp.where(jnp.abs(fp) < 1e-12, jnp.sign(fp) * 1e-12, fp)
        x_new = x - f / fp
        return jnp.where(jnp.isfinite(x_new), x_new, x)
    return jax.lax.fori_loop(0, n_iter, body, jnp.asarray(x0, dtype=jnp.float64))


def r_btfr_def_jax(M_bar, j_acc, init=1.0, n_iter=30):
    """JAX accretion radius r_acc [pc] for an array of j_acc at one mass (solved in kpc)."""
    rv_kpc = rv_def(M_bar) / 1000.0
    y = j_acc / (2.0 * v_btfr_def(M_bar))
    solve_one = lambda yv: newton_solve_r_jax(rv_kpc, yv, x0=init, n_iter=n_iter)
    return jax.vmap(solve_one)(y) * 1000.0
