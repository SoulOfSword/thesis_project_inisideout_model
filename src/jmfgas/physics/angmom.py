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


def j_acc_def(j_MP, t, t0=T0, n=1.0, con=1.0, lambda_ratio=1.0):
    """Accreted specific angular momentum at time t, growing as (t/t0)**n.

    con = k (the spin parameter). The ceiling is j_max = k*j_MP and the floor is j_min = j_max/10,
    so raising k lifts BOTH ends of the growth curve. j_MP is the asymptotic MP+21 relation value.

    lambda_ratio is a halo's spin relative to the median one; it scales the floor only, leaving
    the ceiling at k*j_MP. 1.0 (the default) is the median halo and reproduces j_min = j_max/10.
    """
    j_max = con * j_MP
    j_min = (j_max / 10.0) * lambda_ratio
    return j_min + (j_max - j_min) * (t / t0) ** n
