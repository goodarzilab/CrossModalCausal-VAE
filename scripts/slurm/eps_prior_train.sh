#!/bin/bash
# Train ONE non-Gaussian-eps-prior model (headline sp_dec_gl1000 architecture + eps_prior flag).
# Usage: sbatch --job-name=epsp_<name> scripts/slurm/eps_prior_train.sh <config_basename>
#   config at configs/eps_prior/<config_basename>_config.json
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --mem=350G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/eps_prior/%x_%j.out
#SBATCH --error=logs/eps_prior/%x_%j.err
set -u
NAME=$1
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
mkdir -p logs/eps_prior
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
export PYTHONUNBUFFERED=1
PY=${PY:-python}
CFG=configs/eps_prior/${NAME}_config.json
echo "=== TRAIN eps_prior $NAME  $(date) ==="
$PY src/train.py --config "$CFG" || { echo "TRAIN FAIL $NAME"; exit 1; }
echo "EPS_PRIOR_TRAIN_DONE $NAME"
