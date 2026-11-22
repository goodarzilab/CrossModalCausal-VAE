#!/bin/bash
#SBATCH -J ps_ft
#SBATCH -p cpu_batch_high_mem,cpu_high_mem,cpu_batch
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o logs/ft_%x_%j.log
# usage: PS_PRE=obs|ctrl RP_PARTS=obs,ft,nb RP_OUT=... sbatch -J NAME --export=ALL scripts/slurm/perturbseq_ft.sbatch LINE SEED shift
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
python -I analysis/fig5_interventions/perturbseq/ft_eval.py "$@"
