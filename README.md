# Replication package — Cluster decomposition

This folder is the **reproducibility package** accompanying the manuscript
*Cluster decomposition: a partition-then-fit strategy for heterogeneous causal
discovery* (Journal of Classification submission).

**What is bundled:**
- `data/` — every result file the figures and supplementary tables read (pre-computed).
- `scripts/fig_generators/` — the figure scripts (path-patched, self-contained) and
  `_verify_fig_numbers.py` (validates figure numbers against the manuscript).
- `scripts/experiments/` — the experiment scripts that generated those results.
- `figures_src/` — the TikZ source of the manuscript's Fig1 (compiled inside the paper).
- `REPLICATION_INDEX.md` — the figure/table -> source -> data mapping.

**Quick start:** `pip install -r requirements.txt`, then:
- Reproduce Fig. 2 and Fig. 3:
  `python scripts/fig_generators/gen_Fig2_ClusterDecomposition.py`
  `python scripts/fig_generators/gen_Fig3_EdgeCountPitfall.py`
- Reproduce Fig. 1 (TikZ schematic): `pdflatex figures_src/Fig1_Overview_standalone.tex`
- Verify all figure numbers against the paper:
  `python scripts/fig_generators/_verify_fig_numbers.py`

Every main figure and supplementary table is generated from the bundled result files
in `data/`, so no external download is required.
