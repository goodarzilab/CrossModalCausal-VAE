#!/bin/bash
#SBATCH -J st_hto
#SBATCH -p cpu_batch,cpu_batch_high_mem,cpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 6:00:00
#SBATCH -o logs/%x_%j.log
# Extract the 4 hashtag (HTO) rows (features 36804-36807: Resting_Treg, Stimulated_Treg, Resting_Teff, Stimulated_Teff)
# from the GEO GSE278572 aggregated matrix -> barcode-index x HTO counts (sparse triplets).
SRC=${CMCVAE_DATA:-data}/arce_perturb_citeseq
OUT=${CMCVAE_RESULTS:-results}/paper/analyses/stats/de/arce_hto_triplets.tsv
mkdir -p "$(dirname "$OUT")"
zcat $SRC/GSE278572_matrix.mtx.gz | awk 'NR>3 && $1>=36804 {print $1"\t"$2"\t"$3}' > $OUT
wc -l $OUT; echo HTO_DONE
