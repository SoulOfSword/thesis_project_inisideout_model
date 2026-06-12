#!/bin/bash
#SBATCH -A EUHPC_R05_084
#SBATCH -p dcgp_usr_prod
#SBATCH -J jmfgas_omega_nio
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -o /leonardo_scratch/large/userexternal/acosta01/thesis_project_inisideout_model/logs/slurm/omega_nio_%j.out
#
# NIO per-mass-bin omega scan (engine, loky pool over bins). Extra flags pass through:
#   sbatch run_omega_nio.sbatch --scan-n 500

set -euo pipefail

PROJECT=/leonardo/home/userexternal/acosta01/thesis_project_inisideout_model
LOGS_DIR="/leonardo_scratch/large/userexternal/acosta01/thesis_project_inisideout_model/logs/slurm"
mkdir -p "$LOGS_DIR"
cd "$PROJECT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

echo "[$(date +%H:%M:%S)] omega-vs-mass NIO scan"
python scripts/plots/omega_vs_mass.py --model nio "$@"
echo "[$(date +%H:%M:%S)] done"
