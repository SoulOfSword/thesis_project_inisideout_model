#!/bin/bash
# End-to-end reproduction: inputs -> model artifacts -> inference -> figures.
# Light, single-process steps run locally; the parallel grid/MCMC go to a dcgp node
# via `sbatch --wait`. Run inside tmux so an SSH drop doesn't kill it:
#   tmux new -s jmfgas ; bash slurm/pipeline.sh ; # detach Ctrl-b d
#
# IO/NIO best-fit parameters (override on the CLI: IO_PARAMS="n k", NIO_PARAMS="a b").

set -euo pipefail

PROJECT=/leonardo/home/userexternal/acosta01/thesis_project_inisideout_model
cd "$PROJECT"

IO_PARAMS="${IO_PARAMS:-0.39 1.28}"
NIO_PARAMS="${NIO_PARAMS:-0.96 0.23}"
ts() { date +%H:%M:%S; }

echo "[$(ts)] [1/5] inputs: R_v fit + observational samples"
python scripts/data/fit_rv_vflat.py
python scripts/data/build_sample.py --sample converged
python scripts/data/build_sample.py --sample full --with-hix

echo "[$(ts)] [2/5] model artifacts (single-process engine runs)"
python scripts/model/save_io_grids.py       --params $IO_PARAMS
python scripts/model/save_nio_band.py       --params $NIO_PARAMS
python scripts/model/save_comparison_npz.py --model io
python scripts/model/save_comparison_npz.py --model nio

echo "[$(ts)] [3/5] inference on dcgp (grid; one node each, blocking)"
sbatch --wait slurm/run_grid.sbatch io  4obs
sbatch --wait slurm/run_grid.sbatch nio 4obs
# MCMC cross-checks (optional, slower): uncomment to run
# sbatch --wait slurm/run_mcmc.sbatch io  4obs mcmc-obs 32 2000
# sbatch --wait slurm/run_mcmc.sbatch nio 4obs mcmc-obs 32 2000

echo "[$(ts)] [4/5] inference figures"
python scripts/inference/plot_inference.py --grid outputs/grids/grid_io_4obs_mcmc-obs.npz  --title "IO 4obs"
python scripts/inference/plot_inference.py --grid outputs/grids/grid_nio_4obs_mcmc-obs.npz --title "NIO 4obs"

echo "[$(ts)] [5/5] final planes + comparison figures"
python scripts/model/run_final_planes.py --model io  --params $IO_PARAMS
python scripts/model/run_final_planes.py --model nio --params $NIO_PARAMS
python scripts/plots/compare_profiles.py all

echo "[$(ts)] pipeline done -> outputs/"
