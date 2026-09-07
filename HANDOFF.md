# Handoff — λ-scatter port (in progress)

Context for resuming this work in a fresh session, off the Leonardo cluster.
Repo conventions are in `CLAUDE.md`.

## Task

Port the λ (halo spin) scatter from the old notebook into the current framework, so the
**accretion** model can be run with five j_min values instead of one. λ scales the *floor*
`j_min` only; the ceiling stays at `k·j_MP`.

Five ratios λ/λ_median = `exp(0.57 × [-1, -0.5, 0, 0.5, 1])` = **0.566, 0.752, 1.000, 1.330, 1.768**
(σ_ln λ = 0.57). Ratio 1.0 is the median halo and reproduces the current fiducial exactly.

Origin: `notebooks/model_inside_out.ipynb` cells 20, 39, 40, 149, 152. The old code used ±0.5
rather than ±0.57, and its plotting half no longer exists. **The saved `data/data9_JAX_aKSL/
final_*_lambda.txt` tables are stale — do not use them** (t0 went 12→13, and the R_v / SFL /
BTFR coefficients are now fitted from `data/*.json`).

## Done

| File | Change |
|---|---|
| `physics/angmom.py` | `j_acc_def` gained `lambda_ratio=1.0`; `j_min = (j_max/10)*lambda_ratio` |
| `models/common.py` | `lambda_ratio` threaded through `build_r_acc_matrix_for_all_M[_jax]` (trailing kwarg) |
| `scripts/model/run_final_planes.py` | `lambda_ratio` on `accretion_model_grids`; new `lambda_ratios(cfg)` helper |
| `config/model.yaml` | new `lambda_scatter: {sigma_ln: 0.57, offsets: [-1,-0.5,0,0.5,1]}` |

Default is `lambda_ratio=1.0`, which is bit-identical to the previous behaviour (×1.0 is exact),
so nothing changes unless λ is switched on.

Separately delivered and complete: panel titles removed from all six planes
(`viz/planes.py`, `params_label` dropped), and non-fiducial parameter values now go in the PDF
filename via `param_suffix()` (fiducials are accretion `n=1,k=2`, spin `a=0.1,b=0.5`).

## Not done

1. `scripts/model/save_lambda_grids.py` — one λ per invocation, writes an npz.
2. `run_final_planes.py --lambda-from <dir>` — reads the five, plots.
3. **The rendering decision.** Open question, see below.

## Open decisions

**How to draw the λ spread.** At fixed M_bar *and* fixed f_gas the model no longer gives one
j_bar — each λ has an ω that hits that f_gas, so a constant-f_gas *line* becomes a *band*.
`viz/planes.py::_fgas_tracks` cannot be used unmodified on a flattened λ×ω grid: `np.unique`
does not collapse near-equal floats, so `np.interp` would hop between λ slices and produce a
plausible-looking meaningless line. Options were: median line + shaded band; five thin lines
per level; or scatter of all points. **Undecided — depends on the band width, which was never
measured** (see below).

**Where the production λ grids should live.** Suggested `outputs/grids/lambda/`; `outputs/grids/`
itself holds stale `grid_nio_*` files.

**Floor convention.** Old notebook: `j_min = (j_MP/10)·λ`, independent of k. Current
implementation: `j_min = (k·j_MP/10)·λ`, i.e. the floor scales with k, matching the shipped
`j_acc_def` docstring ("raising k lifts BOTH ends"). The new convention was chosen deliberately.
Note `tests/test_physics.py::test_j_acc_def` encodes the *old* one and therefore still fails.

## The blocker: memory

Five engine grids in one process get OOM-killed on the Leonardo login node (exit 137; it died
after 2 of 5). Rough estimate: ~2–3 GB peak per λ slice — `SD_gas` is (10 ω, 835 radii, 132 times)
float64 per mass ≈ 8.8 MB, vmapped over 50 masses ≈ 440 MB, and `full_from_sigma_jax` holds
several arrays of that shape at once.

This is a single-threaded CPU job producing ~25 KB per λ. **It should run comfortably on a laptop
with 8–16 GB RAM**, which is why the work is moving off the cluster. Needs: the repo plus `data/`,
Python ~3.11, JAX (CPU), numpy, scipy, pandas, matplotlib, pyyaml. `emcee` only for the `--from`
chain path. No GPU, no MPI.

Regardless of machine, keep the run/plot split that `CLAUDE.md` mandates — one process per λ
writing to disk, then a separate plotting step reading them back.

## Measurement still owed

The λ band width under *current* parameters is unknown. Two of five slices completed before the
kill: λ=0.566 gave f_gas 0.029–0.996, λ=0.752 gave 0.041–0.996. An earlier figure of "≤0.08 dex"
came from the stale tables and should be ignored.

Get this number first — it decides the rendering. If the band is a few hundredths of a dex it
belongs in the text, not a figure; if a few tenths, the shaded band is worth drawing.

## Test suite

10 pre-existing failures, verified as such by reverting the λ change and re-running: config drift
(`test_scaffold` expects `t0 == 12.0`, config says 13.0) and notebook-parity tests. The λ change
*fixes* one (`test_inference.py::test_wrappers_picklable_and_finite`). `tests/test_viz_planes.py`
was updated this session for the title removal.

## Parameter sweep (separate, ready to run)

For the paper's parameter-variation panels, one at a time, baryonic only:

- **accretion** — `k` is a pure vertical shift of log(k/2); `n` sets track separation and
  **saturates above n≈1** (spread 0.46/0.75/0.94/0.95/0.94 dex at n=0.25/0.5/1/1.5/2), because
  by n=1 the fast-accretion end already sits on the hardwired floor `j_min = j_max/10`.
  Suggested: `(0.25,2) (0.5,2) (1,1) (1,3)`.
- **spin** — `a` sets the overall accretion rate, `b` its mass slope, pivoting at 1e10 where
  ω = a for any b. Suggested: `(0.03,0.5) (0.3,0.5) (0.1,0.25) (0.1,1.0)`.

```bash
python scripts/model/run_final_planes.py --model accretion --params 0.5 2 --planes baryonic --sample full
```
