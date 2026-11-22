#!/bin/bash
# Train the HEADLINE architecture (sp_dec_gl1000, group_lasso) on ONE healthy-vs-disease split.
# Usage: sbatch --job-name=hd_<split> scripts/slurm/hd_split_train.sh <split>   where <split> in {healthy,disease}
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --mem=350G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/hd_split/%x_%j.out
set -u
SPLIT=$1
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
mkdir -p logs/hd_split
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
export PYTHONUNBUFFERED=1
PY=${PY:-python}
CFG=configs/hd_split/${SPLIT}_config.json

echo "=== TRAIN hd_${SPLIT} (headline sp_dec_gl1000 arch, group_lasso) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY src/train.py --config "$CFG" || { echo "TRAIN FAIL hd_${SPLIT}"; exit 1; }
echo "HD_SPLIT_TRAIN_DONE ${SPLIT}"
