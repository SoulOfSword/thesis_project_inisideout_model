#!/bin/bash
#SBATCH -A EUHPC_R05_084
#SBATCH -p dcgp_usr_prod
#SBATCH -J jmfgas_grid
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /leonardo_scratch/large/userexternal/acosta01/thesis_project_inisideout_model/logs/slurm/grid_%j.out
#
# Adaptive logL grid on one dcgp node. Cells fan out over a fresh worker pool
# per refinement level. Usage: sbatch run_grid.sbatch <model> <likelihood> [sample]
#   sbatch run_grid.sbatch io 4obs
#   sbatch run_grid.sbatch nio fgas full-hix

set -euo pipefail

MODEL="${1:-io}"
LIK="${2:-4obs}"
SAMPLE="${3:-mcmc-obs}"
if [ "$#" -gt 3 ]; then shift 3; else set --; fi   # remaining args -> extra run_grid.py flags

PROJECT=/leonardo/home/userexternal/acosta01/thesis_project_inisideout_model
LOGS_DIR="/leonardo_scratch/large/userexternal/acosta01/thesis_project_inisideout_model/logs/slurm"
mkdir -p "$LOGS_DIR"
cd "$PROJECT"

# one thread per worker so the loky pool isn't oversubscribed
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

NWORKERS="${SLURM_CPUS_ON_NODE:-112}"

echo "[$(date +%H:%M:%S)] grid $MODEL/$LIK ($SAMPLE) on $NWORKERS workers"
python scripts/inference/run_grid.py \
    --model "$MODEL" --likelihood "$LIK" --sample "$SAMPLE" \
    --max-workers "$NWORKERS" "$@"
echo "[$(date +%H:%M:%S)] grid done"
