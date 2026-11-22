#!/bin/bash
# Train ONE residual-independence model (headline sp_dec_gl1000 architecture + resid_indep penalty on eps).
# Usage: sbatch --job-name=ri_<name> scripts/slurm/resid_indep_train.sh <name>   (config at configs/resid_indep/<name>.json)
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --mem=350G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/resid_indep/%x_%j.out
#SBATCH --error=logs/resid_indep/%x_%j.err
set -u
NAME=$1
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
mkdir -p logs/resid_indep
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
export PYTHONUNBUFFERED=1
PY=${PY:-python}
CFG=configs/resid_indep/${NAME}.json
echo "=== TRAIN resid_indep $NAME  $(date) ==="
$PY src/train.py --config "$CFG" || { echo "TRAIN FAIL $NAME"; exit 1; }
echo "RESID_INDEP_TRAIN_DONE $NAME"
