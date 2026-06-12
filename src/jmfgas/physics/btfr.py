"""Baryonic mass - flat rotation velocity relation."""

from ..config import load_config

_b = load_config()["btfr"]
A_LOG10 = _b["A_log10"]
EXPONENT = _b["exponent"]


def v_btfr_def(M_bar, A_log10=A_LOG10, exponent=EXPONENT):
    """Flat rotation velocity [km/s] from baryonic mass: v = (M_bar / 10**A_log10)**(1/exponent)."""
    return (M_bar / 10.0**A_log10) ** (1.0 / exponent)


def mbar_from_vflat(v_flat, A_log10=A_LOG10, exponent=EXPONENT):
    """Inverse relation: M_bar = 10**A_log10 * v_flat**exponent."""
    return 10.0**A_log10 * v_flat**exponent
