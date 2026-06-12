"""Small save/load helpers and path utilities."""

from pathlib import Path

import numpy as np

from .config import ROOT


def resolve(path):
    """Make a path absolute against the project root if it is relative."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ensure_dir(path):
    """Create the parent directory of a file path (or the directory itself)."""
    p = resolve(path)
    target = p if p.suffix == "" else p.parent
    target.mkdir(parents=True, exist_ok=True)
    return p


def save_npz(path, **arrays):
    """Write arrays to an .npz and return the path."""
    p = ensure_dir(path)
    np.savez(p, **arrays)
    return p


def load_npz(path):
    """Load an .npz into a plain dict."""
    data = np.load(resolve(path), allow_pickle=True)
    return {k: data[k] for k in data.files}


def wrote(path):
    """Print a uniform 'wrote' line and return the path."""
    p = resolve(path)
    print(f"wrote {p}")
    return p
