"""Load the shared YAML config and the fitted R_v-v_flat relation."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
_CONFIG = ROOT / "config" / "model.yaml"
_RV_RELATION = ROOT / "data" / "rv_vflat_relation.json"


def load_config(path=None):
    """Read config/model.yaml into a dict."""
    p = Path(path) if path is not None else _CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


def load_rv_relation(path=None):
    """Read the fitted R_v-v_flat relation (slope, intercept) written by fit_rv_vflat.

    R_v is returned in pc by the radius helpers; this just holds the line coefficients.
    """
    p = Path(path) if path is not None else _RV_RELATION
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found - run scripts/data/fit_rv_vflat.py first."
        )
    with open(p) as f:
        return json.load(f)
