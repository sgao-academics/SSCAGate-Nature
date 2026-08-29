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

**Quick start:** `pip install -r requirements.txt`, then run
`scripts/fig_generators/gen_Fig2_ClusterDecomposition.py` and
`gen_Fig3_EdgeCountPitfall.py` (Fig1 is the manuscript's TikZ schematic; see
`REPLICATION_INDEX.md`). Optionally run
`scripts/fig_generators/_verify_fig_numbers.py` to confirm the figure numbers match
the paper.
