from pathlib import Path
import subprocess, os

def get_project_root() -> Path:
    # Prefer git root if this is inside a repo
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
        return Path(root)
    except Exception:
        # Fallback: assume notebook lives in <root>/notebooks/*
        # Climb up until we find data/ and outputs/ (or give up at parent)
        p = Path.cwd()
        candidates = [p, p.parent, p.parent.parent]
        for c in candidates:
            if (c / "data").exists() and (c / "outputs").exists():
                return c
        return p.parent  # last resort

ROOT = get_project_root()
DATA = ROOT / "data"
OUT  = ROOT / "outputs"

# Convenience joiners
def datapath(*parts) -> Path:
    return DATA.joinpath(*parts)

def outpath(*parts) -> Path:
    return OUT.joinpath(*parts)

# I/O helpers that auto-create parent dirs
def savefig(fig, relpath: str, **kwargs):
    p = outpath(relpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, **kwargs)

def save_txt(relpath: str, array, **kwargs):
    import numpy as np
    p = datapath(relpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(p, array, **kwargs)

def save_csv(df, relpath: str, **kwargs):
    p = datapath(relpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, **kwargs)

def read_csv(relpath: str, **kwargs):
    import pandas as pd
    return pd.read_csv(datapath(relpath), **kwargs)

print("ROOT =", ROOT)
print("DATA =", DATA)
print("OUT  =", OUT)