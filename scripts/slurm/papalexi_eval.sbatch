#!/bin/bash
#SBATCH --job-name=unt_eval
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_%x_%j.log
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
# usage: RUN=sp_dec_gl1000 sbatch --export=ALL scripts/slurm/papalexi_eval.sbatch   (also _s1, _s2)
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
RES=${CMCVAE_RESULTS:-results}
mkdir -p $RES/paper/analyses/untouched
python -I -u analysis/fig3_zero_shot/papalexi_untouched/rc_untouched.py $RES/sweep_sparse/$RUN $RES/paper/analyses/untouched/rc_$RUN.json
