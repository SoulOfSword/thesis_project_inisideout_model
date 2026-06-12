"""Invert the inside-out j_acc(t) law for each galaxy's accretion frequency omega.

Given an observed (M_bar, j_bar) and the IO growth params (n, k), the dimensionless
F(omega; n, t0) = <x^n> over [0,1] with weight exp(-omega t0 x) equals
y = (j_bar - j_min) / (k j_max - j_min); F decreases monotonically in omega, so a
bracket-and-bisect inversion gives omega per galaxy.
"""

import numpy as np
from scipy.integrate import simpson

from ..physics.angmom import j_maxer

_X = np.linspace(0.0, 1.0, 256)          # fixed grid for the dimensionless integral


def _gamma_integral(omega, n, t0):
    """∫_0^1 x^n exp(-omega t0 x) dx over the fixed x grid (Simpson)."""
    omega = np.atleast_1d(np.asarray(omega, float))
    integrand = (_X[None, :] ** n) * np.exp(-omega[:, None] * t0 * _X[None, :])
    return simpson(integrand, x=_X, axis=-1)


def F_omega(omega, n, t0):
    """F(omega) = num(n) / num(0); -> 1/(n+1) at omega=0, -> 0 as omega->+inf, -> 1 as -inf."""
    return _gamma_integral(omega, n, t0) / _gamma_integral(omega, 0.0, t0)


def solve_omega(y_target, n, t0=12.0, omega0=1.0, max_expand=60, n_iter=50):
    """Bracket-and-bisect F_omega(.) = y for each y_target.

    Returns (omega, ok); ok is False where y is outside (0, 1) and no bracket exists.
    """
    y = np.asarray(y_target, float)
    lo = np.full_like(y, -omega0)
    hi = np.full_like(y, omega0)
    for _ in range(max_expand):
        lo = np.where(F_omega(lo, n, t0) < y, lo * 1.5, lo)
        hi = np.where(F_omega(hi, n, t0) > y, hi * 1.5, hi)
    Flo, Fhi = F_omega(lo, n, t0), F_omega(hi, n, t0)
    ok = np.isfinite(Flo) & np.isfinite(Fhi) & (Flo >= y) & (Fhi <= y)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        go_right = F_omega(mid, n, t0) > y       # decreasing F: F(mid)>y -> root to the right
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
    return 0.5 * (lo + hi), ok


def omega_per_galaxy(logM, jbar, n, k, t0=12.0):
    """Per-galaxy omega from observed (logM, j_bar) under the IO model (n, k).

    Galaxies whose j_bar is unreachable are pinned to the omega caps: y > 1 -> -10
    (would need omega -> -inf), y < 0 -> +10 (omega -> +inf). Returns (omega, ok, y).
    """
    Mbar = 10.0 ** np.asarray(logM, float)
    j_max = j_maxer(Mbar)
    j_min = j_max / 10.0
    y = (np.asarray(jbar, float) - j_min) / (k * j_max - j_min)
    omega, ok = solve_omega(y, n, t0)
    unsolved = ~ok
    omega[unsolved & (y > 1)] = -10.0
    omega[unsolved & (y < 0)] = 10.0
    return omega, ok, y
