#!/bin/bash
#SBATCH --job-name=ebft_qa_occupancy
#SBATCH --account=bcvd-delta-gpu
#SBATCH --output=ebft_qa_occupancy_%A_%a.out
#SBATCH --error=ebft_qa_occupancy_%A_%a.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=192G
#SBATCH --array=27-32

# Estimated wall time: ~14–18 h per run (20 h includes buffer).
# Array indices 27-32 = Set 3 in configs/qa_code.yaml (occupancy reward sweep):
#   occupancy_distance [mmd_rbf, energy, l_alpha] × ce_loss_coef [0.0, 0.03] = 6 configs
# To run a single config:  sbatch --array=27 scripts/run_qa_code_occupancy.sh

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate ebft

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
export DASHBOARD_PORT=8265

REPO_DIR="${EBFT_DIR:-/projects/bcvd/ldasu/ebft_openrlhf}"
cd "${REPO_DIR}"

ray start --head \
    --port=6379 \
    --node-ip-address=0.0.0.0 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --num-gpus "${GPUS_PER_NODE}"

mkdir -p openrlhf_work_dir

python3 scripts/ebft_sweep.py sweep_config=configs/qa_code.yaml
