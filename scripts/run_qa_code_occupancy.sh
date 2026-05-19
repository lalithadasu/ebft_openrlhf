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
#SBATCH --cpus-per-task=16
#SBATCH --array=27

# Estimated wall time: ~14–18 h per run (20 h includes buffer).
# Array index 27 = Set 3 in configs/qa_code.yaml (single occupancy config):
#   occupancy_distance=energy, ce_loss_coef=0.3

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate ebft

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GPUS_PER_NODE=$SLURM_GPUS_PER_NODE

REPO_DIR="${EBFT_DIR:-/projects/bcvd/ldasu/ebft_openrlhf}"
cd "${REPO_DIR}"

# Find a free Ray internal port (other jobs on the same node may hold 6379).
RAY_PORT=""
for RPC in 6379 6380 6381 6382; do
    if [ "$(lsof -Pi :${RPC} -sTCP:LISTEN 2>/dev/null | wc -l)" -eq 0 ]; then
        RAY_PORT=${RPC}
        echo "Using Ray port ${RAY_PORT}."
        break
    fi
done
if [ -z "${RAY_PORT}" ]; then
    echo "ERROR: No free Ray port found in 6379-6382. Exiting." >&2
    exit 1
fi

# Find a free dashboard port (8265 may be taken by another array task on the same node).
DASHBOARD_PORT=""
for DPC in 8265 8266 8267 8268; do
    if [ "$(lsof -Pi :${DPC} -sTCP:LISTEN 2>/dev/null | wc -l)" -eq 0 ]; then
        DASHBOARD_PORT=${DPC}
        echo "Using Ray dashboard port ${DASHBOARD_PORT}."
        break
    fi
done
if [ -z "${DASHBOARD_PORT}" ]; then
    echo "ERROR: No free dashboard port found in 8265-8268. Exiting." >&2
    exit 1
fi

# Export so ebft_sweep.py reads the correct address for ray job submit.
export DASHBOARD_PORT

# Use the node's actual routable IP — 0.0.0.0 prevents the job agent from
# registering with the head node because it is not a valid return address.
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Node IP: ${NODE_IP}"

ray start --head \
    --port="${RAY_PORT}" \
    --node-ip-address="${NODE_IP}" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${DASHBOARD_PORT}" \
    --num-gpus "${GPUS_PER_NODE}" \
    --num-cpus "${SLURM_CPUS_PER_TASK:-16}"

if [ $? -ne 0 ]; then
    echo "ERROR: ray start failed. Exiting." >&2
    exit 1
fi

# Wait for Ray dashboard to be ready before submitting the job.
echo "Waiting for Ray dashboard on port ${DASHBOARD_PORT}..."
READY=0
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${DASHBOARD_PORT}/api/version" > /dev/null 2>&1; then
        echo "Ray is ready (attempt ${i})."
        READY=1
        break
    fi
    sleep 2
done
if [ "${READY}" -eq 0 ]; then
    echo "ERROR: Ray dashboard did not become ready after 60 s. Exiting." >&2
    exit 1
fi

# Give the job agent a few extra seconds to register after the dashboard HTTP server comes up.
echo "Waiting 15 s for Ray job agent to initialize..."
sleep 15

mkdir -p openrlhf_work_dir

python3 scripts/ebft_sweep.py sweep_config=configs/qa_code.yaml
