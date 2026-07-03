#!/bin/bash
#SBATCH -A EUHPC_R05_084
#SBATCH -p dcgp_usr_prod
#SBATCH -J jmfgas_mcmc
#SBATCH -t 02:00:00
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /leonardo_scratch/large/userexternal/acosta01/thesis_project_inisideout_model/logs/slurm/mcmc_%j.out
#
# emcee MCMC on one dcgp node, walkers over a loky pool. Usage:
#   sbatch run_mcmc.sbatch <model> <likelihood> [sample] [nwalkers] [nsteps] [extra run_mcmc.py flags]
#   sbatch run_mcmc.sbatch io fgas full 112 2000 --init uniform --out outputs/mcmc_chains/foo.h5

set -euo pipefail

MODEL="${1:-io}"
LIK="${2:-4obs}"
SAMPLE="${3:-mcmc-obs}"
NWALKERS="${4:-32}"
NSTEPS="${5:-100}"
if [ "$#" -gt 5 ]; then shift 5; else set --; fi   # remaining args -> extra run_mcmc.py flags

PROJECT=/leonardo/home/userexternal/acosta01/thesis_project_inisideout_model
LOGS_DIR="/leonardo_scratch/large/userexternal/acosta01/thesis_project_inisideout_model/logs/slurm"
mkdir -p "$LOGS_DIR"
cd "$PROJECT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

NWORKERS="${SLURM_CPUS_ON_NODE:-112}"

echo "[$(date +%H:%M:%S)] mcmc $MODEL/$LIK ($SAMPLE) ${NWALKERS}w x ${NSTEPS}s on $NWORKERS workers $*"
python scripts/inference/run_mcmc.py \
    --model "$MODEL" --likelihood "$LIK" --sample "$SAMPLE" \
    --nwalkers "$NWALKERS" --nsteps "$NSTEPS" --max-workers "$NWORKERS" "$@"
echo "[$(date +%H:%M:%S)] mcmc done"