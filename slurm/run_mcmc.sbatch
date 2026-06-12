#!/bin/bash
#SBATCH -A EUHPC_R05_084
#SBATCH -p dcgp_usr_prod
#SBATCH -J jmfgas_mcmc
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /leonardo_scratch/large/userexternal/acosta01/thesis_project_inisideout_model/logs/slurm/mcmc_%j.out
#
# emcee MCMC on one dcgp node, walkers over a loky pool. Usage:
#   sbatch run_mcmc.sbatch <model> <likelihood> [sample] [nwalkers] [nsteps]
#   sbatch run_mcmc.sbatch io 4obs mcmc-obs 32 100

set -euo pipefail

MODEL="${1:-io}"
LIK="${2:-4obs}"
SAMPLE="${3:-mcmc-obs}"
NWALKERS="${4:-32}"
NSTEPS="${5:-100}"
OUT="${6:-}"          # optional custom chain path; keeps older chains from being overwritten

PROJECT=/leonardo/home/userexternal/acosta01/thesis_project_inisideout_model
LOGS_DIR="/leonardo_scratch/large/userexternal/acosta01/thesis_project_inisideout_model/logs/slurm"
mkdir -p "$LOGS_DIR"
cd "$PROJECT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

NWORKERS="${SLURM_CPUS_ON_NODE:-112}"

echo "[$(date +%H:%M:%S)] mcmc $MODEL/$LIK ($SAMPLE) ${NWALKERS}w x ${NSTEPS}s on $NWORKERS workers ${OUT:+-> $OUT}"
if [ -n "$OUT" ]; then
    python scripts/inference/run_mcmc.py \
        --model "$MODEL" --likelihood "$LIK" --sample "$SAMPLE" \
        --nwalkers "$NWALKERS" --nsteps "$NSTEPS" --max-workers "$NWORKERS" --out "$OUT"
else
    python scripts/inference/run_mcmc.py \
        --model "$MODEL" --likelihood "$LIK" --sample "$SAMPLE" \
        --nwalkers "$NWALKERS" --nsteps "$NSTEPS" --max-workers "$NWORKERS"
fi
echo "[$(date +%H:%M:%S)] mcmc done"