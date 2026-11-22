#!/usr/bin/env python
"""
Generate biological prior masks for RNA (CollecTRI) and Protein (OmniPath).
Reads gene/protein names from processed h5ad files and builds prior masks.

Usage:
  python generate_priors.py --rna_path data/combined_healthy/rna_combined_healthy.h5ad \
                            --prot_path data/combined_healthy/citeseq_combined_healthy.h5ad \
                            --out_prefix combined_healthy
"""
import torch
import numpy as np
import anndata as ad
import decoupler as dc
import omnipath as op
import mygene
import argparse
import os


def generate_rna_prior(rna_path, out_path):
    """Generate GRN prior mask from CollecTRI for RNA HVGs."""
    print("=" * 60)
    print("GENERATING RNA PRIOR (CollecTRI)")
    print("=" * 60)

    print(f"  Loading RNA: {rna_path}")
    adata = ad.read_h5ad(rna_path, backed="r")
    target_genes = adata.var_names.tolist()
    n_genes = len(target_genes)
    adata.file.close()
    print(f"  Genes: {n_genes}")

    print("  Downloading CollecTRI...")
    collectri = dc.op.collectri(organism='human')
    print(f"  Downloaded {len(collectri)} interactions")

    gene_to_idx = {gene: i for i, gene in enumerate(target_genes)}
    prior_mask = torch.zeros((n_genes, n_genes))

    valid_edges = collectri[
        collectri['source'].isin(target_genes) & collectri['target'].isin(target_genes)
    ]
    print(f"  Valid edges (both source & target in HVGs): {len(valid_edges)}")

    if not valid_edges.empty:
        source_indices = [gene_to_idx[g] for g in valid_edges['source']]
        target_indices = [gene_to_idx[g] for g in valid_edges['target']]
        prior_mask[source_indices, target_indices] = 1.0

    n_edges = int(prior_mask.sum().item())
    density = n_edges / (n_genes * n_genes) * 100
    print(f"  Prior mask: {n_genes}x{n_genes}, {n_edges} edges ({density:.2f}%)")

    save_dict = {
        'prior_mask': prior_mask,
        'gene_names': target_genes,
        'source': 'CollecTRI (decoupler)',
        'n_edges': n_edges,
    }
    torch.save(save_dict, out_path)
    print(f"  Saved: {out_path}")
    return prior_mask


def generate_protein_prior(prot_path, out_path):
    """Generate PPI prior mask from OmniPath for proteins."""
    print("\n" + "=" * 60)
    print("GENERATING PROTEIN PRIOR (OmniPath)")
    print("=" * 60)

    print(f"  Loading protein: {prot_path}")
    adata = ad.read_h5ad(prot_path, backed="r")
    target_proteins = adata.var_names.tolist()
    n_proteins = len(target_proteins)
    adata.file.close()
    print(f"  Proteins: {n_proteins}")

    print("  Downloading OmniPath interactions...")
    interactions = op.interactions.OmniPath.get(organism='human', genesymbols=True)
    print(f"  Downloaded {len(interactions)} interactions")

    # Map protein names to gene symbols using mygene
    print("  Mapping protein names to gene symbols...")
    mg = mygene.MyGeneInfo()
    query_results = mg.querymany(
        target_proteins, scopes='symbol,alias,name',
        fields='symbol', species='human', verbose=False
    )

    name_map = {}
    for res in query_results:
        if 'symbol' in res:
            name_map[res['query']] = res['symbol']

    print(f"  Mapped {len(name_map)}/{n_proteins} proteins to gene symbols")

    # Build prior mask
    prot_to_idx = {p: i for i, p in enumerate(target_proteins)}
    prior_mask = torch.zeros((n_proteins, n_proteins))

    all_sources = set(interactions['source_genesymbol'])
    all_targets = set(interactions['target_genesymbol'])

    mapped_symbols = set(name_map.values())
    valid_sources = mapped_symbols & all_sources
    valid_targets = mapped_symbols & all_targets

    # Reverse map: symbol → protein name
    symbol_to_prot = {}
    for prot, sym in name_map.items():
        if sym not in symbol_to_prot:
            symbol_to_prot[sym] = prot

    for _, row in interactions.iterrows():
        src_sym = row['source_genesymbol']
        tgt_sym = row['target_genesymbol']
        if src_sym in symbol_to_prot and tgt_sym in symbol_to_prot:
            src_prot = symbol_to_prot[src_sym]
            tgt_prot = symbol_to_prot[tgt_sym]
            if src_prot in prot_to_idx and tgt_prot in prot_to_idx:
                i = prot_to_idx[src_prot]
                j = prot_to_idx[tgt_prot]
                prior_mask[i, j] = 1.0

    n_edges = int(prior_mask.sum().item())
    density = n_edges / (n_proteins * n_proteins) * 100
    print(f"  Prior mask: {n_proteins}x{n_proteins}, {n_edges} edges ({density:.2f}%)")

    save_dict = {
        'prior_mask': prior_mask,
        'protein_names': target_proteins,
        'name_map': name_map,
        'source': 'OmniPath',
        'n_edges': n_edges,
    }
    torch.save(save_dict, out_path)
    print(f"  Saved: {out_path}")
    return prior_mask


def main():
    parser = argparse.ArgumentParser(description="Generate biological priors for VAE training")
    parser.add_argument("--rna_path", type=str, required=True, help="Path to RNA h5ad")
    parser.add_argument("--prot_path", type=str, required=True, help="Path to protein h5ad")
    parser.add_argument("--out_prefix", type=str, required=True, help="Prefix for output files")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    rna_out = os.path.join(args.out_dir, f"rna_hvg_priors_collectri_sc_{args.out_prefix}.pt")
    prot_out = os.path.join(args.out_dir, f"protein_priors_sc_{args.out_prefix}.pt")

    generate_rna_prior(args.rna_path, rna_out)
    generate_protein_prior(args.prot_path, prot_out)

    print("\nDone!")


if __name__ == "__main__":
    main()
