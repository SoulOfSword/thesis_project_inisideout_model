import numpy as np

import jmfgas


def test_config_loads():
    cfg = jmfgas.load_config()
    assert cfg["time"]["t0"] == 12.0
    assert cfg["mcmc"]["nio"]["bounds"]["a"] == [-1.5, 7.0]


def test_npz_roundtrip(tmp_path):
    p = jmfgas.save_npz(tmp_path / "x.npz", a=np.arange(3.0))
    out = jmfgas.load_npz(p)
    assert out["a"].tolist() == [0.0, 1.0, 2.0]
