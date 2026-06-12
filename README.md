# Galaxy Evolution Models for the Baryonic j-M-f<sub>gas</sub> Relation

This repository contains a semi-analytical galaxy evolution model developed to investigate the origin of the tight baryonic specific angular momentum–mass–gas fraction (j<sub>bar</sub> − M<sub>bar</sub> − f<sub>gas</sub>) scaling relation discovered by [Mancera Piña et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...651L..15M/abstract).

**Disclaimer**: The code present here is a collection of the work I have done over 4 years, starting from the beginning of my second year of the Bachelor's. As you may expect, my coding skills were pretty horrendous back then, and I wrote spaghetti code which unfortunately I ended up using throughout the project. It would've taken too much time to rewrite everything and I needed to keep moving to write a thesis and eventually a paper. So cut me some slack :) 

## Code: layout & usage

The model lives in an installable package `jmfgas` (`src/jmfgas/`) with thin terminal
scripts in `scripts/`. Scripts add `src/` to the path themselves, so nothing needs
installing to run them; for tests or interactive use, `pip install -e .`. Shared
defaults are in `config/model.yaml`; run-specific values are CLI arguments. The
notebooks in `notebooks/` are the frozen reference the package was ported from.
Conventions and the module map are in `CLAUDE.md`; the migration record is
`MIGRATION_PLAN.md`.

```
src/jmfgas/   physics/  models/ (JAX engine)  data/  inference/  viz/
scripts/      data/  model/  inference/  plots/
config/model.yaml   slurm/   tests/
```

### Pipeline

```bash
# 1. inputs: R_v(v_flat) power law + the observational samples
python scripts/data/fit_rv_vflat.py
python scripts/data/build_sample.py --sample converged          # + --with-hix for MCMC obs

# 2. model artifacts (present-day grids, NIO band curves, comparison tracks)
python scripts/model/save_io_grids.py       --params 0.39 1.28
python scripts/model/save_nio_band.py       --params 0.96 0.23
python scripts/model/save_comparison_npz.py --model io
python scripts/model/save_comparison_npz.py --model nio

# 3. inference — grid (CPU fan-out) or MCMC; both share the likelihoods
python scripts/inference/run_grid.py --model io  --likelihood 4obs
python scripts/inference/run_mcmc.py --model nio --likelihood 4obs --nsteps 2000
python scripts/inference/plot_inference.py --grid  outputs/grids/grid_io_4obs_mcmc-obs.npz
python scripts/inference/plot_inference.py --chain outputs/mcmc_chains/chain_nio_4obs_mcmc-obs.h5 --burn-in 500

# 4. figures
python scripts/model/run_final_planes.py --model io           # 3 planes (j-M-fgas, stellar, gaseous)
python scripts/plots/compare_profiles.py all
```

On Leonardo, the heavy inference runs on a `dcgp` node: `sbatch slurm/run_grid.sbatch io 4obs`,
`sbatch slurm/run_mcmc.sbatch nio 4obs mcmc-obs 32 2000`. `slurm/pipeline.sh` chains the whole
thing (`sbatch --wait`); run it inside `tmux`. Tests: `pytest`.

## Scientific Objectives

The primary goal is to demonstrate that the observed j<sub>bar</sub> − M<sub>bar</sub> − f<sub>gas</sub> relation can emerge naturally from first principles using galaxy evolution models. We compare **two distinct accretion scenarios**:

1. **Inside-out model**: Gas accretes at progressively larger radii over time, with specific angular momentum increasing as j<sub>acc</sub>(t) ∝ t<sup>n</sup>. This reflects the hierarchical assembly of angular momentum in ΛCDM cosmology.

2. **Non-inside-out model**: Gas accretes with constant specific angular momentum throughout the galaxy's evolution, and galaxies are naturally born at different (constant) sizes.

Both models aim to reproduce the empirical relation:

```
log(j_bar) = 0.73 log(M_bar) + 0.46 log(f_gas) − 4.25
```

which exhibits remarkably low intrinsic scatter.

## Model Framework

### Core Physics

The model tracks the evolution of gas and stellar surface densities through:

```
dΣ_gas/dt = Σ̇_acc − Σ_rSFR
dΣ_★/dt = Σ_rSFR
```

where:
- **Gas accretion**: Exponential temporal profile M̊<sub>acc</sub>(t) = C·e<sup>−ω<sub>acc</sub>t</sup> with accretion frequency ω<sub>acc</sub>
- **Star formation**: Piecewise Kennicutt-Schmidt law based on [Kennicutt & de los Reyes (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...908...61K/abstract) with a density cutoff
- **Angular momentum**: Prescribed evolution j<sub>acc</sub>(t) linking to accretion radius via rotation curve

### Star Formation Law

We implement a piecewise Kennicutt-Schmidt relation with parameters from Kennicutt & de los Reyes (2021):

- **High-density regime** (Σ<sub>gas</sub> > Σ<sub>threshold</sub>): Standard power-law relation
- **Low-density regime**: Cutoff or modified slope to account for reduced star formation efficiency in galaxy outskirts

Gas surface densities are corrected for helium with a factor of 1.36.

### Rotation Curves

Rotation velocity follows an exponential rise to a flat asymptotic value:

```
v_rot(R) = v_flat · [1 − exp(−R/R_v)]
```

where:
- v<sub>flat</sub> is determined from the Baryonic Tully-Fisher Relation ([McGaugh 2012](https://ui.adsabs.harvard.edu/abs/2012AJ....143...40M/abstract))
- R<sub>v</sub> is calibrated using the [SPARC catalogue](http://astroweb.cwru.edu/SPARC/)

## References

### Observational Constraints

- **Scaling relations**: j<sub>bar</sub> − M<sub>bar</sub> − f<sub>gas</sub> from [Mancera Piña et al. (2021a,b)](https://ui.adsabs.harvard.edu/abs/2021Natur.594..485M/abstract)
- **Baryonic Tully-Fisher**: [McGaugh (2012)](https://ui.adsabs.harvard.edu/abs/2012AJ....143...40M/abstract)

### Rotation Curve Calibration

- **SPARC catalogue**: [Lelli et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016AJ....152..157L/abstract) — exponential rotation curve fitting

### Star Formation Law Data

- **Dwarf galaxies**: HI from [Iorio et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.466.4159I/abstract), SFR from [McQuinn et al. (2015)](https://ui.adsabs.harvard.edu/abs/2015ApJ...808..109M/abstract)
- **Spiral galaxies**: HI/SFR from [Leroy et al. (2008)](https://ui.adsabs.harvard.edu/abs/2008AJ....136.2782L/abstract), H₂ from [Frank et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.1722F/abstract)
- **Comprehensive compilation**: [Bacchini et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...641A..70B/abstract)

## Authors

**Andrea Costa**  
Kapteyn Astronomical Institute, University of Groningen  
📧 costa@astro.rug.nl

**Supervisors**: Prof. dr. F. Fraternali, Dr. G. Pezzulli

**Co-authors**: Dr. Pavel E. Mancera-Piña, Dr. Cecilia Bacchini

This work builds upon:

- Costa, A. (2024). *Investigating the Dependence of the Baryonic j-M-f<sub>gas</sub> Relationship on Different Star Formation Laws*. Bachelor thesis, University of Groningen. [Link](https://fse.studenttheses.ub.rug.nl/33339/)
- Cammilleri, E. (2022). *Origin of the Baryonic j-M-f<sub>gas</sub> relation*. Bachelor thesis, University of Groningen. [Link](https://fse.studenttheses.ub.rug.nl/28062/)