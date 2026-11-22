#!/bin/bash
# Train ONE reversal-discrepancy-loss model (headline sp_dec_gl1000 architecture + lambda_reversal).
# Usage: sbatch --job-name=rev_<name> scripts/slurm/rev_loss_train.sh <name>   (name in configs/rev_loss/<name>.json)
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --mem=350G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/rev_loss/%x_%j.out
#SBATCH --error=logs/rev_loss/%x_%j.err
set -u
NAME=$1
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
mkdir -p logs/rev_loss
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
export PYTHONUNBUFFERED=1
PY=${PY:-python}
CFG=configs/rev_loss/${NAME}.json
echo "=== TRAIN rev_loss $NAME  $(date) ==="
$PY src/train.py --config "$CFG" || { echo "TRAIN FAIL $NAME"; exit 1; }
echo "REV_LOSS_TRAIN_DONE $NAME"
