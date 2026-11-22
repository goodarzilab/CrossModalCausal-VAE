#!/bin/bash
#SBATCH -J ps_train
#SBATCH -p goodarzilab_gpu_priority,gpu_batch,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/train_%x_%j.log
# usage: sbatch -J t_LINE_CELLS_SEED scripts/slurm/perturbseq_train.sbatch LINE SEED CELLS
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
python -I analysis/fig5_interventions/perturbseq/train_rna.py "$@"
