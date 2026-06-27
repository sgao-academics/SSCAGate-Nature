<p align="center">
  <img src="https://img.shields.io/badge/Discovery-Phase_Transition-2E86AB?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Validation-5_Modalities-27AE60?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scale-33_Cancers_+_1,190_Cell_Lines-8E44AD?style=for-the-badge" />
</p>

# A Sample-Size Phase Transition Ends the Hard-versus-Soft Clustering Debate

**Shuaidong Gao**  
Chongqing Institute of Foreign Studies · Qijiang Campus, Chongqing 401420, China  
📧 gsd3247186514@gmail.com · 🔗 [ORCID: 0009-0004-5641-3581](https://orcid.org/0009-0004-5641-3581)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Optional-76B900)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)
[![Patent](https://img.shields.io/badge/Patent-Pending-orange)]()

---

## 🔬 What We Discovered

For **22 years**, the machine learning community has been divided: does *hard* clustering (K-means) or *soft* clustering (end-to-end) produce better downstream results? Published benchmarks returned **contradictory conclusions** — until now.

We discovered a **universal sample-size phase transition** that resolves this debate in a single equation:

<p align="center">
  <b>n<sub>crit</sub> = 6.3 × d<sup>0.90</sup> (R² = 0.985)</b>
</p>

- **Small n** (n/d < 4) → **hard clustering dominates** (CAGate)
- **Large n** (n/d ≥ 4) → **soft clustering dominates** (SSCAGate)

The reason the debate persisted for two decades: **all 13 published benchmark datasets** fall in the large-*n* soft-clustering regime. Nobody had ever tested the small-*n* boundary where most biomedical discovery actually occurs.

---

## 📊 Evidence: Five Independent Pillars

| Pillar | Dataset | Key Result | p-value |
|:--|:--|:--|:--|
| **Pan-cancer** | 33 TCGA cancer types (n = 45–1,218) | Spearman r = 0.827 | 3.0×10⁻⁹ |
| **Within-cancer resampling** | 8 cancers, B = 100 bootstrap | Spearman r = 0.793 | 6.2×10⁻⁸ |
| **Single-cell replication** | PBMC 3k / 6k / 10k | Spearman r = 0.820 | 1.1×10⁻³ |
| **Cross-modal universality** | MNIST, 20 Newsgroups, methylation | R² = 0.977 (MNIST exponential decay) | — |
| **Negative controls** | CIFAR-10, synthetic nulls (SNR ≪ 1) | Flat Δ ≈ 0 (no transition) | — |

All 68 configurations (20 cancers × 3 dimensions × 2 methods + baselines) show **100% positive gain** over standard NOTEARS. At d ≥ 150, NOTEARS produces zero edges for all cancers; SSCAGate recovers 89–1,009 edges.

---

## 🧬 DepMap Cross-Platform Validation (1,190 Cell Lines)

The phase transition was independently confirmed on the DepMap 25Q2 dataset:
- **15-dimensional fine scan** (d = 50–500): crossover at d = 275 where n ≈ n<sub>crit</sub>
- **15 lineage-level tests**: 100% positive Δ (all lineages, d = 200)
- **CRISPR gene effect network**: 19 causal edges discovered where NOTEARS finds none

---

## 🧠 Theoretical Foundation

The empirical scaling law is not a coincidence — we derive it from first principles:

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\large%20n_{\text{crit}}%20=%20\frac{4K(1-\bar{\pi})}{\bar{\pi}^2\cdot\text{SNR}}\cdot%20d" />
</p>

where K is the number of clusters, π̄ is the minimum mixture weight, and SNR is the per-dimension signal-to-noise ratio. The derivation (Supplementary Note 3) makes **three testable predictions**, all of which are empirically confirmed.

---

## 📂 Repository Structure

```
├── manuscript/                     # Full submission package
│   ├── manuscript.tex/pdf          # Main paper (18 pp, double-spaced)
│   ├── cover_letter.tex/pdf        # Nature cover letter
│   ├── extended_data.tex/pdf       # 4 ED figures + 2 ED tables
│   ├── supplementary_information.tex/pdf  # Supplementary Notes 1–5 + 3 Tables
│   ├── reporting_summary.tex/pdf   # Nature Reporting Summary
│   └── suggested_reviewers.tex/pdf # 5 recommended reviewers
│
├── figures/
│   ├── Graphical_Abstract.png/pdf  # Nature graphical abstract (one glance, one story)
│   ├── main/                       # Fig 1: Five-panel composite
│   │   ├── Fig1_Composite.png/pdf  # Phase transition + resampling + literature overlay
│   │   └── Fig2_Universal_Scaling.png/pdf  # Bias-variance law + decision framework
│   └── ed/                         # Extended Data Figures 1–4
│
├── scripts/                        # Figure generation (ONE script = ONE figure)
│   ├── gen_fig1.py                 # Fig 1: Five-Modality Composite (panels a-e)
│   ├── gen_fig2.py                 # Fig 2: Bias-Variance Universal Scaling Law
│   ├── gen_ed_fig1.py              # ED Fig 1: K-sweep Heatmap
│   ├── gen_ed_fig2.py              # ED Fig 2: Phase Transition + Robustness
│   ├── gen_ed_fig3.py              # ED Fig 3: Cross-method Validation
│   ├── gen_ed_fig4.py              # ED Fig 4: Hyperparameter Sensitivity
│   ├── gen_graphical_abstract.py    # Graphical Abstract: phase transition + cross-modal strip
│   └── _archive/                   # Old multi-figure scripts (preserved for reference)
│
├── replication_package.zip         # Complete experiment suite (92 files, 4.3 MB)
│   ├── scripts/  (23 .py)         # All experiments + verification
│   ├── src/      (3 .py)          # SSCAGate + cluster_aware_loss source
│   ├── results/  (43 .json)       # Pre-computed benchmark data
│   ├── main_figure/  + ED_figure/ # Pre-built figures (PNG + PDF)
│   └── README.md                  # Step-by-step instructions
│
└── README.md                       # ← you are here
```

---

## ⚡ Quick Reproduction (5 minutes)

```bash
# 1. Extract the replication package
unzip replication_package.zip && cd SSCAGate_Nature_Replication

# 2. Install dependencies (PyTorch, NumPy, Matplotlib, SciPy, scikit-learn, Pillow)
pip install torch numpy pandas matplotlib scipy scikit-learn pillow

# 3. Regenerate all 7 figures (6 main/ED + graphical abstract) from pre-computed data
python scripts/generate_all.py --quick --verify
```

**Output**: `main_figure/Fig1_Composite.png`, `Fig2_Universal_Scaling.png`, `ED_figure/ed_fig1-4.png`, and `figures/Graphical_Abstract.png` — all byte-identical to the paper.

For full experiment reproduction (GPU recommended, 2–4 hours): download TCGA data, then `python scripts/generate_all.py --full`.

---

## 📦 Data Sources

| Dataset | Source | Access | In Replication Package |
|:--|:--|:--|:--:|
| **TCGA (33 cancers)** | UCSC Xena | `scripts/download_tcga.py` (∼300 MB) | Pre-computed results (`*.json`) |
| **MNIST** | `torchvision.datasets.MNIST` | Built-in PyTorch | `results/mnist_phase/ckpt.json` |
| **20 Newsgroups** | `sklearn.datasets.fetch_20newsgroups` | Built-in scikit-learn | `results/text_phase/ckpt.json` |
| **PBMC 3k/6k/10k** | `scanpy.datasets.pbmc3k` | Built-in scanpy | `results/scrna_phase/ckpt.json` |
| **CIFAR-10** (neg. ctrl) | `torchvision.datasets.CIFAR10` | Built-in PyTorch | Hardcoded in `gen_fig1.py` (5 points) |
| **Synthetic DAGs** | `cdsm.data.make_dag(seed=42)` | Built-in cdsm package | `results/synth_phase_results.json` |
| **DepMap 25Q2** | Broad Institute | `depmap.org` portal (∼50 MB) | `results/depmap_ckpt.json` |

**TL;DR**: Fast reproduction (figures from pre-computed data) requires zero downloads. Full reproduction (rerun experiments from raw data) requires only TCGA (∼300 MB via provided script) and DepMap (∼50 MB). MNIST, 20News, PBMC, CIFAR-10, and synthetic data are all accessible via standard Python packages with zero manual download.

---

## 📜 Citation

```bibtex
@article{gao2026sscagate,
  title   = {A Sample-Size Phase Transition Ends the Hard-versus-Soft Clustering Debate},
  author  = {Gao, Shuaidong},
  year    = {2026},
  note    = {Manuscript under review}
}
```

---

## ⚖️ License & Patents

This repository is available for peer review. The code is released under the [MIT License](./LICENSE). A Chinese patent application covering the Cluster-Aware Gating mechanism has been filed (May 2026), with a corresponding U.S. utility patent application in preparation. The complete replication package will be made publicly available under an open-source license upon acceptance and completion of patent filing.

<br>
<p align="center">
  <sub>Independent Research · Chongqing, China · 2025–2026</sub>
</p>
