#!/bin/bash
#SBATCH --job-name=setup_ebft_env
#SBATCH --account=bcvd-delta-gpu
#SBATCH --output=setup_ebft_env_%j.out
#SBATCH --error=setup_ebft_env_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G

# Estimated time: ~45 minutes
# GPU needed for final CUDA verification and flash-attn wheel selection.
#
# Usage (submit once, then reuse the env):
#   sbatch scripts/setup_ebft_env.sh

module load miniforge3-python
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh

ENV_NAME="ebft"
REPO_DIR="${EBFT_DIR:-/projects/bcvd/ldasu/ebft_openrlhf}"

echo "==> Creating conda environment '${ENV_NAME}' with Python 3.12.11"
conda create -y -n "${ENV_NAME}" python=3.12.11

conda activate "${ENV_NAME}"

echo "==> Installing PyTorch 2.6.0 (CUDA 12.4)"
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing vLLM 0.8.4"
pip install vllm==0.8.4

echo "==> Installing Flash Attention 2.7.4"
pip install "flash-attn==2.7.4post1" --no-build-isolation

echo "==> Installing auxiliary dependencies"
pip install \
    psutil \
    sacrebleu \
    sentence-transformers \
    "unbabel-comet==2.2.7" \
    humanize \
    loguru \
    rouge-score \
    datatrove \
    math-verify \
    "omegaconf==2.4.0dev3" \
    "setuptools==81.0.0"

echo "==> Installing OpenRLHF (editable)"
cd "${REPO_DIR}"
pip install -e .

echo "==> Verifying CUDA"
python -c "import torch; print('CUDA device:', torch.cuda.get_device_name(0))"

echo "Done. Activate with: conda activate ${ENV_NAME}"
