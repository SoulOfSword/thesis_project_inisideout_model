# CLAUDE.md

Guidance for working in this repository.

## Project

Semi-analytical galaxy evolution models for the baryonic specific-angular-momentum –
mass – gas-fraction (j_bar – M_bar – f_gas) relation. The paper weighs two ingredients that
regulate the j_bar–f_gas relation at fixed M_bar. **Both models are inside-out** (gas accretes
at growing radii, j_acc(t) grows with n=1); they differ only in what scatters at fixed M_bar:

- **Accretion bias (`accretion`, formerly io):** j_acc(t) shape fixed (n, k); the scatter is the
  accretion rate omega (t_acc = 1/omega), inverted per galaxy from its observed j_bar.
- **Spin bias (`spin`, formerly nio):** omega = a·(M_bar/1e10)^b fixed by mass; the scatter is the
  spin parameter k (0.2–2, the cap on j_acc), inverted per galaxy from its observed j_bar.

The shared inside-out engine lives in `models/common.py`; `accretion.py`/`spin.py` add only their
scatter parameterization. `spin.py` also keeps a legacy constant-radius engine for the parked fit.

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
  models/      common (shared inside-out engine), accretion, spin, profiles  (JAX engine)
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
  value. The fitted star-formation-law coefficients (boissier, new_ksl, old_ksl and
  the cutoff threshold) are read from `data/sfl_relations.json` (written by
  `scripts/data/fit_sfl.py`); the alternative literature laws (kennicutt_modern,
  elise_steep) stay as inline constants in `physics/sfl.py`. `Rf` lives in the config;
  the R_v-v_flat power law (R_v = 10**delta * (v_flat/100)**gamma) is read from
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
