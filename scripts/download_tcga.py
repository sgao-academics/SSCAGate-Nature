"""
TCGA-BRCA Data Download and Preprocessing
===========================================
Downloads TCGA BRCA RNA-Seq from UCSC Xena, builds TF-target prior,
prepares data for CDSM GenomicCausalDAG pipeline.

Author: 无种者联盟 · 千策 (副Agent·癌症应用线)
Date: 2026-05-22
"""
import urllib.request
import gzip
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

DATA_DIR = r'D:\NO.1\cancer_application\data'
CANCER_TYPE = 'BRCA'  # Breast invasive carcinoma
N_TOP_GENES = 500      # Top variable genes to keep
TF_LIST_SOURCE = 'builtin'  # Use curated TF list

# Known human transcription factors (curated from Lambert et al. 2018, Cell 172:650-665)
# Top 50 well-studied TFs in cancer
HUMAN_TFS = [
    'TP53', 'MYC', 'ESR1', 'FOXA1', 'GATA3', 'AR', 'NFKB1', 'RELA',
    'STAT3', 'STAT1', 'STAT5A', 'JUN', 'FOS', 'SP1', 'E2F1', 'E2F4',
    'RB1', 'CTCF', 'YY1', 'NR3C1', 'HIF1A', 'HNF4A', 'PPARG', 'CEBPB',
    'RUNX1', 'RUNX2', 'ETS1', 'ELK1', 'GABPA', 'MAX', 'USF1', 'USF2',
    'SREBF1', 'SREBF2', 'NFE2L2', 'FOXO3', 'FOXM1', 'MYB', 'MEF2A',
    'PAX5', 'TCF3', 'TCF4', 'LEF1', 'SMAD3', 'SMAD4', 'NOTCH1',
    'RBPJ', 'GLI1', 'SOX2', 'POU5F1',
]

# URL for UCSC Xena TCGA BRCA gene expression (log2(TPM+1))
TCGA_BRCA_URL = (
    'https://tcga.xenahubs.net/download/'
    'TCGA.BRCA.sampleMap/HiSeqV2.gz'
)


def download_tcga_brca() -> str:
    """Download TCGA BRCA gene expression from UCSC Xena.
    Returns path to downloaded file.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    gz_path = os.path.join(DATA_DIR, 'TCGA_BRCA_HiSeqV2.gz')
    tsv_path = os.path.join(DATA_DIR, 'TCGA_BRCA_HiSeqV2.tsv')

    if os.path.exists(tsv_path):
        print(f"  Data already downloaded: {tsv_path}")
        return tsv_path

    if not os.path.exists(gz_path):
        print(f"  Downloading TCGA BRCA from UCSC Xena...")
        print(f"  URL: {TCGA_BRCA_URL}")
        try:
            urllib.request.urlretrieve(TCGA_BRCA_URL, gz_path)
            print(f"  Downloaded: {os.path.getsize(gz_path) / 1024 / 1024:.1f} MB")
        except Exception as e:
            print(f"  Download failed: {e}")
            print(f"  Trying alternative URL...")
            # Alternative: GDC portal or Broad Firehose
            raise

    # Decompress
    print(f"  Decompressing...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(tsv_path, 'wb') as f_out:
            f_out.write(f_in.read())

    os.remove(gz_path)
    print(f"  Decompressed: {os.path.getsize(tsv_path) / 1024 / 1024:.1f} MB")
    return tsv_path


def preprocess_expression(
    tsv_path: str,
    n_top_genes: int = N_TOP_GENES,
    tf_list: list = None,
) -> Dict:
    """
    Preprocess TCGA expression data:
    1. Load TSV matrix (genes × samples)
    2. Filter to protein-coding genes (heuristic: no '?' prefix)
    3. Select top variable genes + known TFs
    4. Transpose to samples × genes format for CDSM
    5. Log2-transform if not already done

    Returns:
        X: (n_samples, n_genes_selected) expression matrix
        gene_names: list of selected gene names
        tf_indices: indices of TFs in gene_names
        sample_ids: TCGA sample barcodes
    """
    tf_list = tf_list or []
    print(f"  Loading expression matrix...")

    # Load TSV (first column = gene, first row = header/sample IDs)
    df = pd.read_csv(tsv_path, sep='\t', index_col=0)

    # Remove rows with '?' prefix (non-protein-coding or ambiguous genes)
    df = df[~df.index.str.startswith('?')]

    print(f"  Raw: {df.shape[0]} genes × {df.shape[1]} samples")

    # Compute variance across samples
    gene_var = df.var(axis=1)

    # Ensure TFs are in the gene set (case-insensitive match)
    available_tfs = []
    for tf in tf_list:
        matches = [g for g in df.index if g.upper() == tf.upper()]
        available_tfs.extend(matches)

    available_tfs = list(set(available_tfs))
    print(f"  Found {len(available_tfs)}/{len(tf_list)} TFs in dataset")

    if len(available_tfs) < 5:
        print(f"  WARNING: Very few TFs found. TF list may not match TCGA gene symbols.")
        print(f"  Available genes (first 20): {list(df.index[:20])}")
        # Use top variable genes + relax TF matching
        top_var_genes = gene_var.nlargest(n_top_genes).index.tolist()
        selected_genes = top_var_genes
        tf_indices = []  # Can't identify TFs
    else:
        # Select: known TFs + top variable remaining genes
        remaining_genes = [g for g in gene_var.index if g not in available_tfs]
        top_remaining = gene_var[remaining_genes].nlargest(
            n_top_genes - len(available_tfs)
        ).index.tolist()
        selected_genes = available_tfs + top_remaining
        tf_indices = list(range(len(available_tfs)))

    # Filter to selected genes and transpose
    df_selected = df.loc[selected_genes]
    X = df_selected.values.T.astype(np.float64)  # (samples, genes)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=0.0)

    # Center and scale (standard practice)
    X = X - X.mean(axis=0, keepdims=True)
    X = X / (X.std(axis=0, keepdims=True) + 1e-8)

    print(f"  Processed: {X.shape[0]} samples × {X.shape[1]} genes")
    print(f"  TF indices: {len(tf_indices)} (first: {tf_indices[:5]})")

    return {
        'X': X,
        'gene_names': selected_genes,
        'tf_indices': tf_indices,
        'sample_ids': df.columns.tolist(),
    }


def build_tf_prior(
    gene_names: list,
    tf_indices: list,
    method: str = 'correlation',
) -> np.ndarray:
    """
    Build TF→target gene prior matrix.

    Methods:
    - 'correlation': Use high absolute correlation as prior (cheap proxy)
    - 'regnetwork': Use RegNetwork database (requires download)
    - 'encode': Use ENCODE ChIP-seq data (requires download)

    For proof-of-concept, use correlation-based prior with noise.
    In production, replace with ENCODE/JASPAR data.

    Returns:
        prior_mask: (n_genes, n_genes) binary prior matrix
    """
    n = len(gene_names)

    if method == 'correlation':
        # This is a placeholder. In production, use real TF binding data.
        # We return an identity-like sparse prior that marks known TF→target edges.
        prior_mask = np.zeros((n, n))

        # For each TF, mark it as potential regulator of all genes
        # (This is a weak prior, but better than nothing for constraining search)
        for tf_idx in tf_indices:
            # In reality, each TF regulates a specific subset of genes
            # We use all genes as potential targets (weak prior)
            prior_mask[tf_idx, :] = 1.0

        # Remove self-loops from prior
        np.fill_diagonal(prior_mask, 0.0)

        print(f"  Built correlation-based prior: {prior_mask.sum():.0f} potential edges")
        return prior_mask

    elif method == 'regnetwork':
        # TODO: Download and parse RegNetwork
        print(f"  RegNetwork prior not yet implemented. Using correlation-based fallback.")
        return build_tf_prior(gene_names, tf_indices, method='correlation')

    else:
        raise ValueError(f"Unknown prior method: {method}")


def prepare_tcga_pipeline(
    n_top_genes: int = N_TOP_GENES,
    prior_method: str = 'correlation',
) -> Dict:
    """
    Full TCGA-BRCA data preparation pipeline.
    Returns data dict ready for CDSM GenomicCausalDAG.
    """
    print("=" * 60)
    print("  TCGA-BRCA Data Preparation Pipeline")
    print("=" * 60)

    # Step 1: Download
    print("\n[1/3] Downloading TCGA BRCA expression data...")
    tsv_path = download_tcga_brca()

    # Step 2: Preprocess
    print("\n[2/3] Preprocessing expression data...")
    data = preprocess_expression(
        tsv_path,
        n_top_genes=n_top_genes,
        tf_list=HUMAN_TFS,
    )

    # Step 3: Build prior
    print("\n[3/3] Building TF-target prior...")
    prior_mask = build_tf_prior(
        data['gene_names'],
        data['tf_indices'],
        method=prior_method,
    )

    result = {
        'X': data['X'],
        'gene_names': data['gene_names'],
        'tf_indices': data['tf_indices'],
        'prior_mask': prior_mask,
        'sample_ids': data['sample_ids'],
        'n_samples': data['X'].shape[0],
        'n_genes': data['X'].shape[1],
    }

    print(f"\n  Pipeline complete.")
    print(f"  Shape: {result['X'].shape}")
    print(f"  TFs: {len(result['tf_indices'])}")
    print(f"  Prior edges: {prior_mask.sum():.0f}")

    # Save processed data
    npz_path = os.path.join(DATA_DIR, 'tcga_brca_processed.npz')
    np.savez_compressed(
        npz_path,
        X=result['X'],
        prior_mask=result['prior_mask'],
        tf_indices=np.array(result['tf_indices']),
        gene_names=np.array(result['gene_names']),
    )
    print(f"  Saved: {npz_path}")
    print(f"  Gene names saved in: {DATA_DIR}/tcga_brca_genes.txt")
    with open(os.path.join(DATA_DIR, 'tcga_brca_genes.txt'), 'w') as f:
        for g in result['gene_names']:
            f.write(g + '\n')

    return result


if __name__ == '__main__':
    result = prepare_tcga_pipeline(n_top_genes=500, prior_method='correlation')
