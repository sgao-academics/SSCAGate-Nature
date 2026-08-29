# -*- coding: utf-8 -*-
"""[Curated copy] Figure-generation script for Fig. 3 (residual gating inflates
the edge count while lowering F1: the limits of the edge-count metric).
One figure == one script.

Layout: 2x2 panel grid, matching the manuscript caption:
  (a) Synthetic known-graph data (d=50): reported edge count vs F1 across
      methods and sample-to-dimension ratios, negative trend.
  (b) TCGA BRCA (n=200, d=100, m=20, three seeds): edge count (bars, left) and
      precision (line, right) for baseline / hard-gate / soft-gate.
  (c) On a single graph with no heterogeneity: edge-count advantage (left, blue)
      declines monotonically with n/d while the F1 deficit (right, red) is
      negative at every n/d -> the apparent phase transition is a metric artefact.
  (d) Method-family generality: the inflation reproduces for DAGMA (log-
      determinant acyclicity) and NOTEARS, and is absent only for GOLEM, whose
      baseline already overfits in the small-sample regime.

Data: ../../data/*.json
Output: ../../figures/Fig3_EdgeCountPitfall.pdf / .png
"""
import os, sys, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fig_style as S
S.apply()

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

BLUE, GREEN, ORANGE, RED, GREY = S.BLUE, S.GREEN, S.ORANGE, S.RED, S.GREY
INK = S.INK

with open(os.path.join(BASE, 'synth_v2_d50.json'), encoding='utf-8') as f:
    d2 = json.load(f)
with open(os.path.join(BASE, 'tcga_implant_BRCA_n200.json'), encoding='utf-8') as f:
    d3 = json.load(f)['results']
with open(os.path.join(BASE, 'spurious_phase_reproduction.json'), encoding='utf-8') as f:
    d4 = json.load(f)['results']
with open(os.path.join(BASE, 'e1_checkpoint.json'), encoding='utf-8') as f:
    e1 = json.load(f)

# ============ panel (d): method-family generality ============
grp = defaultdict(list)
for k, v in e1.items():
    m = re.match(r'(\w+)\|nd([\d.]+)\|s\d+\|(\w+)', k)
    if not m:
        continue
    grp[(m.group(1), m.group(2), m.group(3))].append(v)
cores = ['notears', 'dagma', 'golem']
core_lab = {'notears': 'NOTEARS', 'dagma': 'DAGMA', 'golem': 'GOLEM'}
ND = '0.5'
d_edges = [np.mean([x['edges'] for x in grp[(c, ND, 'base')]]) for c in cores]
d_f1b = [np.mean([x['f1'] for x in grp[(c, ND, 'base')]]) for c in cores]
d_soft_e = [np.mean([x['edges'] for x in grp[(c, ND, 'soft')]]) for c in cores]
d_f1s = [np.mean([x['f1'] for x in grp[(c, ND, 'soft')]]) for c in cores]

# ============ figure ============
fig = plt.figure(figsize=(9.6, 7.2))
gs = fig.add_gridspec(2, 2, wspace=0.40, hspace=0.50,
                      left=0.08, right=0.90, top=0.93, bottom=0.12)

# --- panel a: synthetic edge count vs F1 ---
ax = fig.add_subplot(gs[0, 0]); S.panel_label(ax, 'a')
nds = sorted(d2['results'].keys(), key=float)
methods = [('base', 'baseline', BLUE, 'o'), ('hard', 'hard', GREEN, 's'), ('soft', 'soft', RED, '^')]
all_e, all_f = [], []
for key, label, c, mk in methods:
    e = [d2['results'][nd][key]['edges'] for nd in nds]
    f = [d2['results'][nd][key]['f1'] for nd in nds]
    ax.scatter(e, f, s=58, c=c, marker=mk, label=label, edgecolors='white',
               linewidths=0.6, zorder=3)
    all_e += e; all_f += f
z = np.polyfit(all_e, all_f, 1)
xs = np.linspace(min(all_e) - 4, max(all_e) + 4, 60)
ax.plot(xs, np.polyval(z, xs), '--', color=GREY, lw=1.0, zorder=1,
        label='trend ($%.3f$/edge)' % z[0])
ax.set_xlabel('reported edge count', fontsize=7)
ax.set_ylabel('$F_1$', fontsize=7)
ax.set_xlim(0, 150); ax.set_ylim(0, 0.62)
ax.legend(fontsize=6.2, loc='upper right', framealpha=0.95)
ax.set_title('Synthetic (known truth)', fontsize=9)

# --- panel b: TCGA dual-axis bars ---
ax = fig.add_subplot(gs[0, 1]); S.panel_label(ax, 'b')
names = ['baseline', 'hard', 'soft']; keys = ['base', 'hard', 'soft']
edges = [d3[k]['pred_edges'] for k in keys]
prec = [d3[k]['precision'] for k in keys]
colors = [BLUE, GREEN, RED]
x = np.arange(len(names))
axb = ax.twinx()
ax.bar(x, edges, 0.55, color=colors, alpha=0.92)
ax.set_ylabel('edge count', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.6)
ax.set_ylim(0, 115)
for i in range(3):
    ax.text(x[i], edges[i] + 2, '$%.1f$' % edges[i], ha='center', fontsize=6.4)
axb.plot(x, prec, 'o-', color=INK, lw=1.6, ms=5.5)
axb.set_ylabel('precision', fontsize=7, color=INK)
axb.tick_params(axis='y', colors=INK, labelsize=6.6)
axb.set_ylim(0, 0.62)
axb.set_yticks([0.0, 0.2, 0.4, 0.6])
axb.set_yticklabels(['0.0', '0.2', '0.4', '0.6'])
for i in range(3):
    axb.annotate('$%.2f$' % prec[i], (x[i], prec[i]), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=6.4, color=INK)
ax.spines['top'].set_visible(False); axb.spines['top'].set_visible(False)
ax.set_title('TCGA BRCA (implanted)', fontsize=9)

# --- panel c: spurious phase transition ---
ax = fig.add_subplot(gs[1, 0]); S.panel_label(ax, 'c')
nds4 = sorted(d4.keys(), key=float); nd4 = [float(x) for x in nds4]
ed = [d4[x]['edge_delta'] for x in nds4]
fd = [d4[x]['f1_delta'] for x in nds4]
axc = ax.twinx()
ax.plot(nd4, ed, 'o-', color=BLUE, lw=1.9, ms=5.5, label='edge $\\Delta$ (soft$-$base)')
ax.set_xlabel('$n/d$', fontsize=7)
ax.set_ylabel('edge $\\Delta$', fontsize=7, color=BLUE)
ax.tick_params(axis='y', colors=BLUE, labelsize=6.6)
ax.set_xscale('log', base=2); ax.set_xticks(nd4); ax.set_xticklabels([str(x) for x in nd4])
ax.axhline(0, color=BLUE, lw=0.6, ls=':')
axc.plot(nd4, fd, 's--', color=RED, lw=1.9, ms=5.5, label='$F_1$ $\\Delta$ (soft$-$base)')
axc.set_ylabel('$F_1$ $\\Delta$', fontsize=7, color=RED)
axc.tick_params(axis='y', colors=RED, labelsize=6.6)
axc.axhline(0, color=RED, lw=0.6, ls=':')
l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = axc.get_legend_handles_labels()
ax.legend(l1 + l2, ['edge $\\Delta$', '$F_1$ $\\Delta$'], fontsize=6.2, loc='upper right',
          bbox_to_anchor=(1.0, 1.0), framealpha=0.95)
ax.spines['top'].set_visible(False); axc.spines['top'].set_visible(False)
ax.set_title('Metric fabricates a phase transition', fontsize=9)
ax.set_ylim(-5, 145)

# --- panel d: method-family generality ---
ax = fig.add_subplot(gs[1, 1]); S.panel_label(ax, 'd')
x = np.arange(len(cores)); w = 0.34
ax.bar(x - w / 2, d_edges, w, label='baseline', color=BLUE, edgecolor=INK, linewidth=0.5)
ax.bar(x + w / 2, d_soft_e, w, label='soft-gated', color=RED, edgecolor=INK, linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels([core_lab[c] for c in cores], fontsize=6.8)
ax.set_ylabel('reported edges ($n/d = 0.5$)', fontsize=7)
ax.set_ylim(-30, 175)
for i in range(len(cores)):
    ax.text(x[i] - w / 2, d_edges[i] + 4, '$%d$' % d_edges[i], ha='center', fontsize=6.0)
    ax.text(x[i] + w / 2, d_soft_e[i] + 4, '$%d$' % d_soft_e[i], ha='center', fontsize=6.0)
    ax.text(x[i], -12, '$F_1$: $%.2f\\to%.2f$' % (d_f1b[i], d_f1s[i]),
            clip_on=False, ha='center', va='top', fontsize=5.8)
ax.legend(fontsize=6.2, loc='upper left', framealpha=0.95)
ax.set_title('Pitfall is not specific to the acyclicity penalty', fontsize=9)

fig.savefig(os.path.join(FIG_DIR, 'Fig3_EdgeCountPitfall.pdf'))
fig.savefig(os.path.join(FIG_DIR, 'Fig3_EdgeCountPitfall.png'), dpi=300)
print('Saved Fig3_EdgeCountPitfall.pdf/.png')
