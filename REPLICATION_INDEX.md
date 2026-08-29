# Replication Index
## Cluster decomposition: a partition-then-fit strategy for heterogeneous causal discovery

This package reproduces **all three main figures and the supplementary tables** of
the manuscript from the bundled result files in `data/`. The two matplotlib figure
scripts read their data from `data/` (path-patched; no hard-coded data-directory dependency) and
write the figure to `figures/`. **Fig1 is the manuscript's schematic figure and is
reproduced from its TikZ source** via a self-contained standalone launcher
(`figures_src/Fig1_Overview_standalone.tex`), so it compiles inside the package
without needing the manuscript LaTeX file.

## Figure -> source -> data
| Figure | Source | Data (data/) |
|---|---|---|
| Fig1_Overview | `figures_src/Fig1_Overview_standalone.tex` + `Fig1_Overview.tikz` (TikZ; compile the standalone launcher) | (conceptual; reads no data) |
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
| S9 | nonlinear_edge_trap.json |
| S10 | kd_robustness.json |
| S11 | cagate_grid.json / cagate_real_value.json (plus PC-GES-DirectLiNGAM comparison) |
| S12(a),(b) | d_scan_official.json (official solver) |
| S13 | c_sensitivity_checkpoint.json |
| S2 alt | e4_checkpoint.json / e10_hard_gating_trajectory.json / h_shd_grid_checkpoint.json / gate_feedback_trajectory.json (gate variants and robustness) |

> All supplementary tables now read their results directly from the bundled files
> in `data/` (pre-computed). No external TCGA download is required to reproduce the
> figures or the supplementary tables.

## Re-run the experiments from scratch
The experiment scripts in `scripts/experiments/` regenerate the result JSONs. They
require (i) the paper's PyTorch NOTEARS core, (ii) TCGA expression data downloaded
from UCSC Xena (`TCGA_<cancer>_HiSeqV2.tsv`, see `scripts/download_tcga.py`), and
(iii) a GPU/CPU with the `requirements.txt` packages. Because the raw TCGA data is
large (~1 GB) and external, the **pre-computed results are bundled in `data/`** so the
figures and tables reproduce without re-running the (hours-long) experiments. To
re-run a specific experiment, install `pip install -r requirements.txt`, download the
TCGA files with `scripts/download_tcga.py`, and run the corresponding script in
`scripts/experiments/`; it will write its checkpoint JSON into `data/`.

## Reproduce Fig. 1 (the manuscript schematic)
```bash
cd figures_src
pdflatex Fig1_Overview_standalone.tex   # run twice
```
Output: `figures_src/Fig1_Overview_standalone.pdf`. Requires a TeX distribution
(TeXLive or MiKTeX) with the standard TikZ/PGF libraries.

## Determinism
All synthetic `make_dag(seed=42)` calls and all NOTEARS/K-means runs use fixed seeds;
per-seed cells are checkpointed (unit-granular) so a partial re-run is resumable.
