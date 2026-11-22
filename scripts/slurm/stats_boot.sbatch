#!/bin/bash
#SBATCH -J st_boot
#SBATCH -p cpu_batch,cpu_batch_high_mem,cpu_high_mem,cpu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --array=0-39
#SBATCH -o logs/%x_%A_%a.log
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
python -I analysis/fig3_zero_shot/stats/boot.py $SLURM_ARRAY_TASK_ID ${NREP:-25}
