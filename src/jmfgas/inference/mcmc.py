"""emcee MCMC runner with an HDF5 backend and loky parallelism."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import emcee
from loky import get_reusable_executor


def init_worker():
    """Enable JAX float64 in each worker process."""
    os.environ["JAX_ENABLE_X64"] = "1"
    import jax
    jax.config.update("jax_enable_x64", True)


_MOVES = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]


def run_mcmc(log_prob, ndim, nwalkers, nsteps, filename, init=None,
             fresh_start=True, max_workers=12, nsteps_is_total=False, progress=True):
    """Run (or resume) emcee, saving every step to an HDF5 backend. Returns the sampler."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    backend = emcee.backends.HDFBackend(str(filename))

    if fresh_start:
        backend.reset(nwalkers, ndim)
        if init is None:
            raise ValueError("init required for a fresh start")
        pos = init
        steps_to_run = nsteps
    else:
        current = backend.iteration
        if current == 0:
            raise ValueError("No previous run found; use fresh_start=True")
        pos = backend.get_last_sample()
        if nsteps_is_total:
            steps_to_run = nsteps - current
            if steps_to_run <= 0:
                return emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                             backend=backend, moves=_MOVES)
        else:
            steps_to_run = nsteps

    executor = get_reusable_executor(max_workers=max_workers, initializer=init_worker)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob,
                                    pool=executor, backend=backend, moves=_MOVES)
    sampler.run_mcmc(pos, steps_to_run, progress=progress)
    return sampler


def get_sampler_from_backend(filename):
    """Open an HDF5 chain read-only (for plotting/analysis, never re-runs)."""
    return emcee.backends.HDFBackend(str(filename), read_only=True)


@dataclass
class MCMCResult:
    flat_samples: np.ndarray
    medians: np.ndarray
    p16: np.ndarray
    p84: np.ndarray

    @classmethod
    def from_backend(cls, filename, burn_in=0, thin=1):
        flat = get_sampler_from_backend(filename).get_chain(
            discard=burn_in, thin=thin, flat=True)
        p16, p50, p84 = np.percentile(flat, [16, 50, 84], axis=0)
        return cls(flat, p50, p16, p84)
