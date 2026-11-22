#!/bin/bash
# Train ONE cohort-combo under group_lasso (sp_dec_gl1000 sparsity) + run the 3-cohort perturbation
# eval. Usage: sbatch --job-name=gl_<name> scripts/slurm/cohort_sweep_train_eval.sh <combo_name_gl>
#   e.g. scripts/slurm/cohort_sweep_train_eval.sh combo_hao-ste_nodag_gl
#SBATCH --partition=goodarzilab_gpu_priority,gpu_batch_high_mem,gpu_batch,gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=GPUCA6E,GPU724A
#SBATCH --mem=350G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/combo_gl/%x_%j.out
set -u
NAME=$1
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
mkdir -p logs/combo_gl
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
export PYTHONUNBUFFERED=1
CFG=configs/combo_gl/${NAME}_config.json
RD=${CMCVAE_RESULTS:-results}/combo_gl/${NAME}
PY=${PY:-python}
D=${CMCVAE_DATA:-data}

echo "=== TRAIN $NAME ==="
$PY src/train.py --config "$CFG" || { echo "TRAIN FAIL $NAME"; exit 1; }

echo "=== PERTURB EVAL $NAME (3 cohorts, paper harness) ==="
export CMC_CKPT=final CMC_DE_LFC_RNA=0.10 CMC_DE_LFC_PROT=0.05
PERTURB_RNA_PATH=$D/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad PERTURB_PROT_PATH=$D/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad CMC_DS_TAG=frangieh $PY analysis/common/full_perturb_eval.py "$RD" || echo "EVAL FRANGIEH FAIL"
PERTURB_RNA_PATH=$D/chen_perturb_icciteseq/export/iccite_rna.h5ad PERTURB_PROT_PATH=$D/chen_perturb_icciteseq/export/iccite_prot.h5ad CMC_DS_TAG=iccite $PY analysis/common/full_perturb_eval.py "$RD" || echo "EVAL ICCITE FAIL"
PERTURB_RNA_PATH=$D/arce_perturb_citeseq/arce_rna.h5ad PERTURB_PROT_PATH=$D/arce_perturb_citeseq/arce_prot.h5ad CMC_DS_TAG=arce $PY analysis/common/full_perturb_eval.py "$RD" || echo "EVAL ARCE FAIL"
echo "COMBO_GL_DONE $NAME"
