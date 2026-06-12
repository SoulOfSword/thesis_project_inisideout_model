"""The fixed-f_gas model tracks match the reference per-mass interpolation, and the
three plane functions draw without error on the saved inside-out grids."""

import numpy as np
import scipy.interpolate as spl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from jmfgas.config import ROOT, load_config
from jmfgas.viz.planes import (_fgas_tracks, plane_jM_fgas, plane_stellar, plane_gaseous)

_GRIDS = ROOT / "data" / "data9_JAX_aKSL"


def _io_grids():
    load = lambda name: np.nan_to_num(np.loadtxt(_GRIDS / f"final_{name}_cutoff_ksl.txt"))
    cfg = load_config()
    mg = cfg["mass_grid"]
    logM = np.linspace(mg["logM_min"], mg["logM_max"], mg["n"])
    return logM, {"f_gas": load("f_gas"), "j_bar": load("j_bar"),
                  "j_star": load("j_star"), "j_gas": load("j_gas"),
                  "M_star": load("Mstar_grid"), "M_gas": load("Mgas_grid")}


def _reference_track(logM, fgas, logj, fv):
    """Reference cell-158 construction: per-mass f_gas->logj on a dense f grid, picking
    the nearest sample to fv, then a linear spline through the masses that contain fv."""
    n = 1000
    intf = np.zeros((len(logM), n))
    intj = np.zeros((len(logM), n))
    for i in range(len(logM)):
        intf[i] = np.linspace(fgas[i].min(), fgas[i].max(), n)
        intj[i] = spl.interp1d(fgas[i], logj[i], kind="linear")(intf[i])
    tl, idx = [], []
    for i in range(len(logM)):
        if fv > intf[i].min() and fv < intf[i].max():
            idx.append(int(np.argmin(np.abs(intf[i] - fv)))); tl.append(1)
        else:
            idx.append(0); tl.append(0)
    ij = [intj[j][idx[j]] for j in range(len(logM)) if tl[j] == 1]
    if len(ij) <= 1:
        return None
    ix = np.argwhere(np.array(tl) == 1)
    iM = logM[ix]
    iMl = np.linspace(iM[0], iM[-1], 1000).reshape(-1)
    f = spl.InterpolatedUnivariateSpline(iM.reshape(len(iM)), ij, k=1)
    return iMl, f(iMl).reshape(-1)


@pytest.mark.parametrize("fv", [0.1, 0.2, 0.3, 0.5, 0.7])
def test_tracks_match_reference(fv):
    logM, g = _io_grids()
    fgas = g["f_gas"]
    logj = np.log10(np.where(g["j_bar"] > 0, g["j_bar"], np.nan))
    logMmat = np.broadcast_to(logM[:, None], logj.shape)
    tracks = {lvl: (xs, ys) for lvl, xs, ys in _fgas_tracks(logMmat, logj, fgas, (fv,))}
    ref = _reference_track(logM, fgas, logj, fv)
    assert fv in tracks and ref is not None
    xm, ym = tracks[fv]
    xn, yn = ref
    lo, hi = max(xn.min(), xm.min()), min(xn.max(), xm.max())
    xs = np.linspace(lo, hi, 50)
    diff = np.abs(np.interp(xs, xn, yn) - np.interp(xs, xm, ym))
    # both are piecewise-linear through the same mass nodes -> agreement is tight
    assert np.nanmax(diff) < 0.02


def test_planes_render():
    logM, g = _io_grids()
    rng = np.random.default_rng(0)
    nobs = 20
    obs = {"fgas": rng.uniform(0.1, 0.9, nobs)}
    for tag in ("bar", "gas", "star"):
        obs[f"log_M{tag}"] = rng.uniform(8, 11, nobs)
        obs[f"log_j{tag}"] = rng.uniform(1, 4, nobs)
        obs[f"log_M{tag}_err"] = np.full(nobs, 0.1)
        obs[f"log_j{tag}_err"] = np.full(nobs, 0.1)
    comp = {}
    for fn, args in [(plane_jM_fgas, (logM, g["j_bar"], g["f_gas"])),
                     (plane_stellar, (g["M_star"], g["j_star"], g["f_gas"])),
                     (plane_gaseous, (g["M_gas"], g["j_gas"], g["f_gas"]))]:
        fig, ax = plt.subplots()
        sc = fn(ax, *args, obs, comp, params_label="(test)")
        assert sc is not None
        assert len(ax.lines) > 0      # at least one f_gas model track was drawn
        plt.close(fig)
