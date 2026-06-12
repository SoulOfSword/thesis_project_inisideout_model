# CLAUDE.md

Guidance for working in this repository.

## Project

Semi-analytical galaxy evolution models for the baryonic specific-angular-momentum –
mass – gas-fraction (j_bar – M_bar – f_gas) relation. Two accretion scenarios:

- **Inside-out (io):** gas accretes at growing radii over time (galaxies grow in size).
- **Non-inside-out (nio):** gas accretes at constant radius; galaxies are born at
  different fixed sizes. Its accretion rate carries a mass-dependent timescale.

The code is being migrated from the notebooks in `notebooks/` into the `jmfgas`
package (`src/jmfgas/`) plus thin scripts in `scripts/`. **The notebooks are the
authoritative reference for the physics**; ported code must match them numerically.
Migration plan and decisions: `MIGRATION_PLAN.md`.

## Layout

```
src/jmfgas/
  config.py    load_config (config/model.yaml), load_rv_relation (data json)
  io.py        save/load npz, path helpers
  physics/     btfr, radius (R_v), angmom (j_max, j_acc), sfl
  models/      common, inside_out, non_inside_out, profiles  (JAX engine)
  data/        sample build + corrections
  inference/   likelihoods, mcmc, grid
  viz/         cornerplot, chains, planes
scripts/       data/  model/  inference/  plots/   (terminal entry points)
config/model.yaml   shared defaults
notebooks/     frozen reference, not imported
```

## Running scripts

Scripts add `src/` to the path themselves, so no install is needed to run them:

```bash
python scripts/inference/run_mcmc.py --model nio --likelihood 4obs
```

For tests / interactive use you may `pip install -e .`.

## Conventions

- **Config, not constants.** Anything tunable (t0, mass/time grids, prior bounds,
  SFL name, R_v and BTFR coefficients) is a CLI argument or a `config/model.yaml`
  value. The only inline hardcoded numbers allowed are the star-formation-law
  coefficient tables in `physics/sfl.py`. `Rf` and the SFL threshold live in the
  config; the R_v-v_flat power law (R_v = 10**alpha * v_flat**beta) is read from
  `data/rv_vflat_relation.json` (written by `scripts/data/fit_rv_vflat.py`).
- **Radii are in pc** by default in the model engine; divide by 1000 for the rare
  kpc consumer.
- **JAX only in `models/`** (and the likelihoods that call them). `@jit`, float64
  (`jax.config.update("jax_enable_x64", True)` + the loky worker initializer).
- **Run vs plot are separate.** One script produces chains/grids on disk; another
  reads them and plots (burn-in / thinning applied at read time).
- **Comments and docstrings: short and human.** Write a comment only when the *why*
  is non-obvious. No multi-paragraph boilerplate. **Never** put a paper name, a
  person's name, or a notebook-cell reference in code, docstrings, variable names,
  filenames, or plot text. Traceability to the notebooks stays in test names and
  out-of-code notes.

## Verification

Every ported unit gets a numerical parity test against the notebook code path
(`tests/`), tight tolerance (rtol ~1e-6 numpy, ~1e-5 JAX float64). Do not consider a
unit done until its parity test passes.

## HPC (Leonardo / CINECA)

- Account `EUHPC_R05_084`. CPU work on `dcgp` nodes (~112 cores), partition
  `dcgp_usr_prod`. Large outputs to `$SCRATCH`, not `$HOME`.
- Grid inference fans out over independent cells with a fresh process pool per
  refinement level (float64 worker init). MCMC walkers run on CPU via loky.
- Entry points: `slurm/run_grid.sbatch <model> <lik>`, `slurm/run_mcmc.sbatch
  <model> <lik> [sample nwalkers nsteps]`; `slurm/pipeline.sh` chains the whole
  reproduction. Never run the parallel grid/MCMC on a login node — the per-user
  memory cap kills the JAX workers; single-process engine runs are fine locally.
- The grid is the preferred inference path: it parallelizes cleanly across cores.
  The emcee+loky MCMC currently gets poor pool speedup (~100 s/step for 32 walkers
  on io/4obs); use the grid for the posterior, MCMC only as a slow cross-check.
- Use `tmux` to survive SSH drops; `sbatch --wait` to chain pipeline stages.
