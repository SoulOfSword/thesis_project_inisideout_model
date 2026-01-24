# Galaxy Evolution Models for the Baryonic j-M-f<sub>gas</sub> Relation

This repository contains a semi-analytical galaxy evolution model developed to investigate the origin of the tight baryonic specific angular momentum–mass–gas fraction (j<sub>bar</sub> − M<sub>bar</sub> − f<sub>gas</sub>) scaling relation discovered by [Mancera Piña et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...651L..15M/abstract).

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

This work builds upon:

- Costa, A. (2024). *Investigating the Dependence of the Baryonic j-M-f<sub>gas</sub> Relationship on Different Star Formation Laws*. Bachelor thesis, University of Groningen. [Link](https://fse.studenttheses.ub.rug.nl/33339/)
- Cammilleri, E. (2022). *Origin of the Baryonic j-M-f<sub>gas</sub> relation*. Bachelor thesis, University of Groningen. [Link](https://fse.studenttheses.ub.rug.nl/28062/)