#!/bin/bash
#SBATCH -J st_pbde
#SBATCH -p cpu_batch_high_mem,cpu_high_mem,cpu_batch
#SBATCH -c 4
#SBATCH --mem=160G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x_%j.log
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
python -I data_prep/pseudobulk_de/pb_de.py "$@"
