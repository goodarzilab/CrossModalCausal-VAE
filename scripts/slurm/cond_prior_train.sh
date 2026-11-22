#!/bin/bash
# Train ONE conditional-exogenous-prior (iVAE / CausalVAE auxiliary-variable prior) model:
# headline sp_dec_gl1000 architecture + cond_prior_on flag (celltype|donor|disease).
# Usage: sbatch --job-name=cptr_<name> scripts/slurm/cond_prior_train.sh <config_basename>
#   config at configs/cond_prior/<config_basename>_config.json  (e.g. cp_celltype)
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --mem=350G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/cond_prior/%x_%j.out
#SBATCH --error=logs/cond_prior/%x_%j.err
set -u
NAME=$1
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
mkdir -p logs/cond_prior
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
export PYTHONUNBUFFERED=1
PY=${PY:-python}
CFG=configs/cond_prior/${NAME}_config.json
echo "=== TRAIN cond_prior $NAME  $(date) ==="
$PY src/train.py --config "$CFG" --no_wandb || { echo "TRAIN FAIL $NAME"; exit 1; }
echo "COND_PRIOR_TRAIN_DONE $NAME"
