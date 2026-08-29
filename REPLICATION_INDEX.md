# Replication Index
## Cluster decomposition: a partition-then-fit strategy for heterogeneous causal discovery

This package reproduces **all three main figures and the supplementary tables** of
the manuscript from the bundled result files in `data/`. The two matplotlib figure
scripts read their data from `data/` (path-patched; no hard-coded data-directory dependency) and
write the figure to `figures/`. **Fig1 is the manuscript's schematic figure and is
reproduced from its TikZ source** (`figures_src/Fig1_Overview.tikz`), which the
manuscript `\input`s directly inside the LaTeX figure environment.

## Figure -> source -> data
| Figure | Source | Data (data/) |
|---|---|---|
| Fig1_Overview | `figures_src/Fig1_Overview.tikz` (TikZ, compiled inside the manuscript) | (conceptual; reads no data) |
| Fig2_ClusterDecomposition | scripts/fig_generators/gen_Fig2_ClusterDecomposition.py | overnight_checkpoint.json, overnight_checkpoint_cancers.json, cluster_method_comparison.json |
| Fig3_EdgeCountPitfall | scripts/fig_generators/gen_Fig3_EdgeCountPitfall.py | synth_v2_d50.json, tcga_implant_BRCA_n200.json, spurious_phase_reproduction.json, e1_checkpoint.json |

## Reproduce the data figures (Fig2, Fig3) in one command
```bash
pip install -r requirements.txt
cd scripts/fig_generators
python gen_Fig2_ClusterDecomposition.py
python gen_Fig3_EdgeCountPitfall.py
```
Outputs land in `figures/`. Fig1 is not generated here: it is the manuscript's own
TikZ figure, compiled by LaTeX (`\input{../figures/Fig1_Overview.tikz}`).

## Reproduce the numbers (verify figure values against the manuscript)
```bash
python scripts/fig_generators/_verify_fig_numbers.py
```
This reads the bundled JSONs in `data/` and checks the key figure values printed in
the manuscript (Fig2 gain range, 16-cancer per-cluster F1, Fig3 edge/F1/precision
anchors, DAGMA/GOLEM anchors, partitioner F1). Exit shows PASS/FAIL per check.

## Supplementary tables -> underlying data
| Table | Source data |
|---|---|
| S1, S7 (paired t, Cohen's d), S14 (SHD) | overnight_checkpoint.json (Adam core) / fig1_official.json (official L-BFGS-B re-run) |
| S2 | tcga_implant_BRCA_n200.json |
| S3 | synth_v2_d50.json |
| S4 | cluster_method_comparison.json |
| S5, S6 | overnight_checkpoint_cancers.json |
| S8 | e1_checkpoint.json |
| S9 | nonlinear_edge_trap (re-run) |
| S10 | kd_robustness (re-run) |
| S11 | cagate_grid / cagate_real_value / PC-GES-DirectLiNGAM (re-run) |
| S12(a),(b) | d_scan_official.json (official solver) |
| S13 | c_sensitivity (re-run) |

## Re-run the experiments from scratch
The experiment scripts in `scripts/experiments/` regenerate the result JSONs. They
require (i) the paper's PyTorch NOTEARS core, (ii) TCGA expression data downloaded
from UCSC Xena (`TCGA_<cancer>_HiSeqV2.tsv`, see the repo's `download_tcga.py`), and
(iii) a GPU/CPU with the `requirements.txt` packages. Because the raw TCGA data is
large (~1 GB) and external, the **pre-computed results are bundled here** so the
figures and tables reproduce without re-running the (hours-long) experiments.

## Determinism
All synthetic `make_dag(seed=42)` calls and all NOTEARS/K-means runs use fixed seeds;
per-seed cells are checkpointed (unit-granular) so a partial re-run is resumable.
