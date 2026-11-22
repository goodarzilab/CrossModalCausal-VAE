#!/bin/bash
# One CMC-VAE synthetic fit (+ interventional FT). Usage: sbatch synthetic_cmcvae.sbatch <family> <seed> <dr> <dp> <genes> <prots>
#SBATCH --partition=cpu_batch,cpu_batch_high_mem,cpu_high_mem,cpu
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -u
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
OUT=${CMCVAE_RESULTS:-results}/paper/analyses/identifiability/synthetic/v2_cmcvae_${1}_s${2}_dr${3}.json
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
mkdir -p "$(dirname "$OUT")"
python -I analysis/fig5_interventions/synthetic/synthetic_cmcvae.py \
  --families $1 --seeds $2 --dr $3 --dp $4 --genes $5 --proteins $6 --out $OUT
