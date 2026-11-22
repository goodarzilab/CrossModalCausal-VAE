#!/bin/bash
#SBATCH --partition=cpu_high_mem,cpu_batch_high_mem,cpu_batch,cpu,cpu_preemptible
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep_eval/%x_%j.out
set -u
NAME=$1
cd "${CMCVAE_ROOT:-${SLURM_SUBMIT_DIR:-.}}"   # submit from the repository root (or set CMCVAE_ROOT)
mkdir -p logs/sweep_eval
# Activate the CMC-VAE environment before submitting (sbatch inherits it), e.g. `conda activate cmcvae`.
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
PY=${PY:-python}
RD=${CMCVAE_RESULTS:-results}/${NAME}
D=${CMCVAE_DATA:-data}
export CMC_CKPT=final CMC_DE_LFC_RNA=0.10
declare -A DSR=([frangieh]="$D/perturb_citeseq/perturb_cite_seq_rna_processed.h5ad" [iccite]="$D/chen_perturb_icciteseq/export/iccite_rna.h5ad" [arce]="$D/arce_perturb_citeseq/arce_rna.h5ad")
declare -A DSP=([frangieh]="$D/perturb_citeseq/perturb_cite_seq_protein_processed.h5ad" [iccite]="$D/chen_perturb_icciteseq/export/iccite_prot.h5ad" [arce]="$D/arce_perturb_citeseq/arce_prot.h5ad")
echo "=== [1/3] block_isolated (RR/RP perturbation, mutilated do_op) x3: $NAME $(date) ==="
for ds in frangieh iccite arce; do
  CMC_MUTILATE=1 CMC_DS_TAG=$ds PERTURB_RNA_PATH="${DSR[$ds]}" PERTURB_PROT_PATH="${DSP[$ds]}" CMC_DE_LFC_PROT=0.05 \
    $PY analysis/common/block_isolated_ablation.py "$RD" || echo "FAIL block_isolated $NAME $ds"
done
echo "=== [2/3] edge_reversal (fwd/reversal test RR+RP) x3: $NAME ==="
for ds in frangieh iccite arce; do
  CMC_MUTILATE=1 CMC_DS_TAG=$ds PERTURB_RNA_PATH="${DSR[$ds]}" PERTURB_PROT_PATH="${DSP[$ds]}" CMC_DE_LFC_PROT=0.05 \
    $PY analysis/common/edge_reversal_ablation.py "$RD" || echo "FAIL edge_reversal $NAME $ds"
done
echo "=== [3/3] recovery + perturb_mean -> sweep_metrics.json: $NAME ==="
$PY analysis/common/sweep_eval_model.py "$RD" || echo "FAIL sweep_eval_model $NAME"
echo "SWEEP_EVAL_DONE $NAME $(date)"
