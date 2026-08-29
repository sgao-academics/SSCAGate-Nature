# -*- coding: utf-8 -*-
"""[Curated copy] Figure-generation script for Fig. 2 (cluster decomposition
recovers heterogeneity-specific causal edges). One figure == one script.

Layout: 2x2 panel grid, matching the manuscript caption:
  (a) Synthetic linear-SEM (d=50, K=3, 10 seeds): F1 vs ground-truth union
      across nine heterogeneity x sample-size conditions (baseline /
      per-cluster / oracle).
  (b) TCGA expression (d=100, K=3, m=8 per subgroup, 20 seeds per cancer):
      per-cluster vs baseline vs oracle F1 across 16 large-sample cancers.
  (c) 33-cancer residual-gating scatter: soft-gated F1 vs baseline F1, with
      colour encoding edge-count inflation (all below y=x).
  (d) Alternative partitioners (K-means / spectral / GMM vs oracle): the gap
      to the oracle is not closed by swapping the partitioner.

Data: ../../data/*.json
Output: ../../figures/Fig2_ClusterDecomposition.pdf / .png
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fig_style as S
S.apply()

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, GREEN, ORANGE, RED, GREY = S.BLUE, S.GREEN, S.ORANGE, S.RED, S.GREY
INK = S.INK

# ============ data loading ============
with open(os.path.join(BASE, 'overnight_checkpoint.json'), encoding='utf-8') as f:
    cp = json.load(f)
with open(os.path.join(BASE, 'overnight_checkpoint_cancers.json'), encoding='utf-8') as f:
    cpC = json.load(f)
with open(os.path.join(BASE, 'cluster_method_comparison.json'), encoding='utf-8') as f:
    cmj = json.load(f)

# ============ panel (a): synthetic grouped bars ============
order = [(0.2, 200), (0.2, 300), (0.2, 600), (0.4, 200), (0.4, 300), (0.4, 600),
         (0.8, 200), (0.8, 300), (0.8, 600)]
labels = ['\n'.join([str(p), str(n)]) for p, n in order]
bm, bstd, pm, pstd, om, ostd = [], [], [], [], [], []
for p, n in order:
    cfg = 's0p%g_n%d' % (p, n)
    b = [cp['grid/%s/seed%d/baseline' % (cfg, s)]['f1'] for s in range(10)]
    pc = [cp['grid/%s/seed%d/percluster' % (cfg, s)]['f1'] for s in range(10)]
    o = [cp['grid/%s/seed%d/oracle' % (cfg, s)]['f1'] for s in range(10)]
    bm.append(np.mean(b)); bstd.append(np.std(b, ddof=1))
    pm.append(np.mean(pc)); pstd.append(np.std(pc, ddof=1))
    om.append(np.mean(o)); ostd.append(np.std(o, ddof=1))
bm, pm, om = np.array(bm), np.array(pm), np.array(om)

# ============ panel (b): 16-cancer per-cluster ============
rowsb = {}
for k, v in cpC.items():
    if v['exp'] != 'percluster':
        continue
    rowsb.setdefault((v['cancer'], v['method']), []).append(v)
cancersB = sorted(set(c for c, _ in rowsb))
bB = [np.mean([v['f1'] for v in rowsb[(c, 'baseline')]]) for c in cancersB]
pB = [np.mean([v['f1'] for v in rowsb[(c, 'percluster')]]) for c in cancersB]
oB = [np.mean([v['f1'] for v in rowsb[(c, 'oracle')]]) for c in cancersB]
bBs = [np.std([v['f1'] for v in rowsb[(c, 'baseline')]], ddof=1) for c in cancersB]
pBs = [np.std([v['f1'] for v in rowsb[(c, 'percluster')]], ddof=1) for c in cancersB]
oBs = [np.std([v['f1'] for v in rowsb[(c, 'oracle')]], ddof=1) for c in cancersB]

# ============ panel (c): 33-cancer gating scatter ============
rowsc = {}
for k, v in cpC.items():
    if v['exp'] != 'gating':
        continue
    rowsc.setdefault((v['cancer'], v['method']), []).append(v)
cancersC = sorted(set(c for c, _ in rowsc))
bC = np.array([np.mean([v['f1'] for v in rowsc[(c, 'baseline')]]) for c in cancersC])
sC = np.array([np.mean([v['f1'] for v in rowsc[(c, 'soft')]]) for c in cancersC])
infl = np.array([(np.mean([v['n_edges'] for v in rowsc[(c, 'soft')]]) -
                  np.mean([v['n_edges'] for v in rowsc[(c, 'baseline')]])) for c in cancersC])

# ============ panel (d): alternative partitioners ============
res = cmj['results']
part_ord = [('base', 'global'), ('kmeans', 'K-means'), ('spectral', 'spectral'),
            ('gmm', 'GMM'), ('oracle', 'oracle')]
pk = [r[0] for r in part_ord]
plab = [r[1] for r in part_ord]
pv = [res[k]['f1'] for k in pk]
pcol = [BLUE, GREEN, GREY, GREY, ORANGE]  # oracle highlighted

# ============ figure ============
fig = plt.figure(figsize=(9.6, 7.2))
gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.46,
                      left=0.08, right=0.98, top=0.93, bottom=0.11)
ekw = dict(elinewidth=0.6, capsize=1.8, capthick=0.6)

# --- panel a: synthetic ---
ax = fig.add_subplot(gs[0, 0]); S.panel_label(ax, 'a')
x = np.arange(len(order)); w = 0.26
ax.bar(x - w, bm, w, label='baseline', color=BLUE, yerr=bstd, error_kw=ekw)
ax.bar(x, pm, w, label='per-cluster', color=GREEN, yerr=pstd, error_kw=ekw)
ax.bar(x + w, om, w, label='oracle', color=ORANGE, yerr=ostd, error_kw=ekw)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.0)
ax.set_ylabel('$F_1$')
ax.set_xlabel('private-edge density ($s_{0}^{\\mathrm{priv}}$) / $n$', fontsize=7)
ax.set_ylim(0, 0.85)
ax.set_title('Synthetic linear-SEM', fontsize=9)
ax.legend(fontsize=6.2, loc='upper left', ncol=1, framealpha=0.95)

# --- panel b: 16-cancer ---
ax = fig.add_subplot(gs[0, 1]); S.panel_label(ax, 'b')
x = np.arange(len(cancersB)); w = 0.26
ax.bar(x - w, bB, w, color=BLUE, yerr=bBs, error_kw=ekw)
ax.bar(x, pB, w, color=GREEN, yerr=pBs, error_kw=ekw)
ax.bar(x + w, oB, w, color=ORANGE, yerr=oBs, error_kw=ekw)
ax.set_xticks(x); ax.set_xticklabels(cancersB, fontsize=5.4, rotation=45, ha='right')
ax.set_ylabel('$F_1$')
ax.set_xlabel('TCGA cancer (16 large-sample)', fontsize=7)
ax.set_ylim(0, 0.42)
ax.set_title('Pan-cancer (implanted truth)', fontsize=9)

# --- panel c: 33-cancer gating scatter ---
ax = fig.add_subplot(gs[1, 0]); S.panel_label(ax, 'c')
lims = [min(bC.min(), sC.min()) - 0.02, max(bC.max(), sC.max()) + 0.02]
ax.plot(lims, lims, '--', color=GREY, lw=1.0, zorder=1, label='$y=x$')
norm = mpl.colors.Normalize(vmin=infl.min(), vmax=infl.max())
sc = ax.scatter(bC, sC, c=infl, cmap='RdYlBu_r', norm=norm, s=44,
                edgecolors='white', linewidths=0.5, zorder=3)
cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.9)
cb.set_label('edge-count inflation', fontsize=6.4)
cb.ax.tick_params(labelsize=6)
ax.set_xlabel('baseline $F_1$', fontsize=7)
ax.set_ylabel('soft-gated $F_1$', fontsize=7)
ax.set_title('33 cancers (soft gate)', fontsize=9)
ax.legend(fontsize=6.2, loc='upper left', framealpha=0.95)
ax.set_xlim(lims); ax.set_ylim(lims)

# --- panel d: alternative partitioners ---
ax = fig.add_subplot(gs[1, 1]); S.panel_label(ax, 'd')
x = np.arange(len(pk)); w = 0.55
ax.bar(x, pv, w, color=pcol, edgecolor=INK, linewidth=0.6)
ax.set_xticks(x); ax.set_xticklabels(plab, fontsize=6.4)
ax.set_ylabel('$F_1$')
ax.set_xlabel('strategy ($d=50$, $K=3$, $s_{0}^{\\mathrm{priv}}=0.4$, $n=300$)', fontsize=6.6)
ax.set_ylim(0, 0.85)
for i, v in enumerate(pv):
    ax.text(i, v + 0.015, '$%.2f$' % v, ha='center', fontsize=6.0)
ax.set_title('Alternative partitioners', fontsize=9)

fig.savefig(os.path.join(FIG_DIR, 'Fig2_ClusterDecomposition.pdf'))
fig.savefig(os.path.join(FIG_DIR, 'Fig2_ClusterDecomposition.png'), dpi=300)
print('Saved Fig2_ClusterDecomposition.pdf/.png')
