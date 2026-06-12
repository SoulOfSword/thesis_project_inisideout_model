"""Accreted specific angular momentum."""

from ..config import load_config

_cfg = load_config()
_j = _cfg["j_max"]
J_SLOPE = _j["slope"]
J_LOG10_NORM = _j["log10_norm"]
T0 = _cfg["time"]["t0"]


def j_maxer(M_bar):
    """Maximum accreted specific angular momentum [kpc km/s]."""
    return (M_bar**J_SLOPE) * 10.0**J_LOG10_NORM


def j_minner(M_bar):
    """Minimum accreted specific angular momentum: j_max / 10."""
    return j_maxer(M_bar) / 10.0


def j_acc_def(j_max, t, t0=T0, n=1.0, con=1.0, lambda_ratio=1.0):
    """Accreted specific angular momentum at time t.

    Grows from j_min = (j_max/10)*lambda_ratio to con*j_max as (t/t0)**n.
    lambda_ratio scatters the birth size; 1.0 is the median spin.
    """
    j_min = (j_max / 10.0) * lambda_ratio
    return j_min + (con * j_max - j_min) * (t / t0) ** n
