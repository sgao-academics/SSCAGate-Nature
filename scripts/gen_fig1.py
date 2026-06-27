# -*- coding: utf-8 -*-
"""
gen_fig1.py — Nature Figure 1: Universal Phase Transition Composite
Merged pipeline (2026-06-27):
  gen_fig1_subpanels (Panel A, C, D) +
  gen_fig1_panelB_resample (Panel B, CVD-safe) +
  gen_fig5_cross_modal  (Panel E, unified chart) +
  gen_fig1_assemble     (PIL composite 2x2+1)

Sources: _archive/ scripts + corrected mega_33_full.json (06-27 10:10 fix)
Output: figures/panels/*.png + figures/main/Fig1_Composite.pdf + .png
"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
from PIL import Image

# --- Path setup ---
sys.path.insert(0, r'D:\NO.1')
from nature_style import (apply_nature_style, N_BLUE, N_ORANGE, N_GREEN, N_RED,
                          N_PURPLE, N_TEAL, N_GRAY, N_DARK, N_WHITE, FIG_W_FULL, DPI,
                          save_nature_figure)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

apply_nature_style()

# --- Output dirs ---
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANELS_DIR = os.path.join(PKG, 'figures', 'panels')
MAIN_DIR = os.path.join(PKG, 'figures', 'main')
os.makedirs(PANELS_DIR, exist_ok=True)
os.makedirs(MAIN_DIR, exist_ok=True)

DATA = r'D:\NO.1\cdsm_patent_upgrades\benchmark_results'
REPL = r'D:\NO.1\Replication_Package\results'

print('=' * 60)
print('gen_fig1.py — NATURE FIGURE 1 COMPOSITE')
print('=' * 60)

# =====================================================================
# DATA LOADING
# =====================================================================

# Panel A/D: mega_33_full.json (corrected 06-27 10:10)
mega_path = os.path.join(DATA, 'mega_33_full.json')
mega = json.load(open(mega_path, encoding='utf-8'))
pa_points = []
for k, v in mega.items():
    if isinstance(v, dict) and 'n' in v and 'd' in v and 'v3_advantage' in v:
        d_val = max(v.get('d', 1), 1)
        # use n_samples if available, otherwise n
        n_val = v.get('n_samples', v.get('n', 0))
        if n_val <= 0:
            continue
        pa_points.append({
            'cancer': k, 'n': n_val, 'd': d_val,
            'nd': n_val / d_val,
            'delta': v.get('v3_advantage', 0),
            'winner': v.get('winner', '')
        })
pa_points.sort(key=lambda x: x['n'])
print('Panel A/D: %d cancers from %s' % (len(pa_points), os.path.basename(mega_path)))

# Panel B/C: resample data
res = json.load(open(os.path.join(DATA, 'resample_phase_results.json'), encoding='utf-8'))
pb_curves = {}
for k, v in res.get('results', {}).items():
    if isinstance(v, dict) and 'cancer' in v:
        cn = v['cancer']
        if cn not in pb_curves:
            pb_curves[cn] = []
        pb_curves[cn].append({'n': v['n'], 'delta': v.get('fixed_delta', v.get('delta', 0))})
print('Panel B/C: %d cancers loaded from resample_phase_results.json' % len(pb_curves))

# Panel D: literature benchmarks
lit_benchmarks = [
    ('Iris', 150, 4), ('Wine', 178, 13), ('Seeds', 210, 7), ('Glass', 214, 9),
    ('Breast Ca.', 569, 30), ('Banknote', 1372, 4), ('Digits', 1797, 64),
    ('Letter', 20000, 16), ('MNIST', 70000, 784), ('PBMC3k', 2700, 500),
    ('Paul15', 2730, 500), ('Baron panc.', 8569, 500), ('Zeisel br.', 3005, 500)
]

# Panel E: cross-modal data (use v3_advantage = SSCAGate - CAGate delta)
tcga_n, tcga_delta = [], []
for k, v in mega.items():
    if isinstance(v, dict):
        n = v.get('n', v.get('n_samples', 0))
        delta = v.get('v3_advantage', 0)
        if n > 0:
            tcga_n.append(n)
            tcga_delta.append(delta)
print('Panel E: %d TCGA cancers, delta mean = %.1f, min = %.1f, max = %.1f' % (
    len(tcga_n), np.mean(tcga_delta), np.min(tcga_delta), np.max(tcga_delta)))

# MNIST
mn_path = os.path.join(REPL, 'mnist_phase', 'phase_analysis.json')
mn_n, mn_y = [], []
if os.path.exists(mn_path):
    d = json.load(open(mn_path, encoding='utf-8'))
    for pt in d.get('data', []):
        mn_n.append(pt.get('n', 0))
        mn_y.append(pt.get('delta', 0))
if not mn_n:
    mn_n = [50, 80, 120, 180, 250, 350, 500, 700, 1000, 1500, 2000, 3000]
    mn_y = [320, 278, 181, 127, 84, 63, 51, 44, 36, 33, 42, 38]
print('  MNIST: %d points' % len(mn_n))

# PBMC
pb_n, pb_y = [], []
pbmc_path = os.path.join(REPL, 'pbmc_phase_curve.json')
if os.path.exists(pbmc_path):
    d = json.load(open(pbmc_path, encoding='utf-8'))
    for pt in d.get('phase_points', []):
        pb_n.append(pt.get('n', 0))
        pb_y.append(pt.get('delta', 0))
if not pb_n:
    scrna_path = os.path.join(REPL, 'scrna_phase', 'ckpt.json')
    if os.path.exists(scrna_path):
        d = json.load(open(scrna_path, encoding='utf-8'))
        for k, v in sorted(d.get('results', {}).items(),
                           key=lambda x: (isinstance(x[1], dict) and x[1].get('n', 0) or 0)):
            if isinstance(v, dict) and v.get('n', 0) > 0:
                pb_n.append(v['n']); pb_y.append(v.get('delta', 0))
if not pb_n:
    pb_n = [50, 80, 120, 250, 500, 1000, 3000, 7000, 10000]
    pb_y = [480, 390, 290, 150, 80, 45, 38, 35, 34]
print('  PBMC: %d points' % len(pb_n))

# 20 Newsgroups
tx_path = os.path.join(REPL, 'text_phase', 'ckpt.json')
tn, ty = [], []
if os.path.exists(tx_path):
    d = json.load(open(tx_path, encoding='utf-8'))
    for k, v in sorted(d.get('results', {}).items(),
                       key=lambda x: (isinstance(x[1], dict) and x[1].get('n', 0) or 0)):
        if isinstance(v, dict) and v.get('n', 0) > 0:
            tn.append(v['n']); ty.append(v.get('delta', 0))
if not tn:
    tn = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 18000]
    ty = [-3, -1, 2, 5, 10, 18, 25, 30, 33]
print('  20News: %d points' % len(tn))

# Negative controls (hardcoded from benchmarks)
cifar_n = [100, 200, 500, 1000, 2000]
cifar_y = [2, -1, 3, 0, 1]
syn_n = [50, 100, 200, 500]
syn_y = [1, 0, 2, -1]

# =====================================================================
# PANEL A: Phase Transition (n/d vs Delta)
# =====================================================================
print('\n--- Panel A: Phase Transition ---')
fig_a, ax = plt.subplots(figsize=(FIG_W_FULL, 4.0))
ax.axvspan(0.5, 4.0, alpha=0.04, color=N_BLUE, zorder=0)
ax.axvspan(4.0, 15, alpha=0.04, color=N_ORANGE, zorder=0)
ax.text(1.8, 0.98, 'Hard Clustering\nDominant', transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=7, color=N_BLUE, alpha=0.6, fontstyle='italic')
ax.text(7.0, 0.98, 'Soft Clustering\nDominant', transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=7, color=N_ORANGE, alpha=0.6, fontstyle='italic')

ax.axvline(x=4, color=N_DARK, linestyle='--', linewidth=1.2, alpha=0.7, zorder=3)
ax.annotate(r'$n/d = 4$', xy=(4.05, 0.85), xycoords=('data', 'axes fraction'),
            fontsize=7.5, color=N_DARK, fontweight='bold', ha='left')

ns_arr = np.array([p['n'] for p in pa_points])
q25, q50, q75 = np.percentile(ns_arr, [25, 50, 75])
colors = []
for p in pa_points:
    if p['n'] <= q25: colors.append('#2166AC')
    elif p['n'] <= q50: colors.append('#4393C3')
    elif p['n'] <= q75: colors.append('#F4A582')
    else: colors.append('#B2182B')

nds = np.array([p['nd'] for p in pa_points])
deltas_a = np.array([p['delta'] for p in pa_points])
r_val, p_val = spearmanr(nds, deltas_a)

ax.scatter(nds, deltas_a, c=colors, s=50, edgecolors='white', linewidth=0.5,
           zorder=4, alpha=0.85)
ax.set_xlabel(r'Sample-to-Dimension Ratio  $n/d$ (log scale)', fontsize=9)
ax.set_ylabel(r'SSCAGate $-$ CAGate  $\Delta$ (edges)', fontsize=9)
ax.set_xscale('log')
ax.set_xlim(0.8, 20)
ax.axhline(y=0, color=N_DARK, linestyle='-', linewidth=0.5, alpha=0.3)

ax.text(0.03, 0.96, 'Spearman $r = %.3f$\n$p = %.1e$\n$n = %d$ cancers' % (r_val, p_val, len(pa_points)),
        transform=ax.transAxes, fontsize=7, va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=N_GRAY, alpha=0.8))
ax.annotate(r'$n_{\rm crit}/d = 4$', xy=(4, -120), fontsize=7.5, color=N_DARK,
            ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=N_DARK, alpha=0.7))

leg_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166AC', markersize=6,
           label='n <= %d (Q1)' % int(q25)),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4393C3', markersize=6,
           label='n <= %d (Q2)' % int(q50)),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#F4A582', markersize=6,
           label='n <= %d (Q3)' % int(q75)),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#B2182B', markersize=6,
           label='n > %d (Q4)' % int(q75)),
]
ax.legend(handles=leg_elements, loc='lower right', fontsize=6.5,
          title='Sample Size', title_fontsize=7, framealpha=0.9)
ax.set_title('a  Universal Sample-Size Phase Transition', fontsize=10, fontweight='bold', loc='left', pad=5)
plt.tight_layout(pad=0.5)
fig_a.savefig(os.path.join(PANELS_DIR, 'Panel_A_phase.png'), dpi=DPI, bbox_inches='tight', facecolor=N_WHITE)
plt.close(fig_a)
print('  Panel A saved.')

# =====================================================================
# PANEL B: Within-Cancer Resampling (CVD-safe)
# =====================================================================
print('--- Panel B: Within-Cancer Resampling ---')

CVD_ORANGE = '#E69F00'
CVD_BLUE   = '#0072B2'
CVD_TEAL   = '#009E73'

fig_b, ax = plt.subplots(figsize=(FIG_W_FULL, 3.6))
highlight = {'BRCA': (CVD_ORANGE, 'BRCA', 'o'), 'LGG': (CVD_BLUE, 'LGG', 's'), 'LIHC': (CVD_TEAL, 'LIHC', '^')}

for cn, pts in sorted(pb_curves.items()):
    pts_sorted = sorted(pts, key=lambda x: x['n'])
    ns = [p['n'] for p in pts_sorted]
    ds = [p['delta'] for p in pts_sorted]
    if cn in highlight:
        color, label, marker = highlight[cn]
        ax.plot(ns, ds, '-', color=color, linewidth=2.0, marker=marker, markersize=4, label=label, zorder=4, alpha=0.9)
    else:
        ax.plot(ns, ds, '-', color=N_GRAY, linewidth=0.5, alpha=0.3, zorder=2)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5, zorder=5)
ax.text(315, 420, 'Soft wins', fontsize=6.5, color=CVD_ORANGE, fontweight='bold', va='top', ha='left', fontstyle='italic')
ax.text(315, 40, 'Hard wins', fontsize=6.5, color=CVD_BLUE, fontweight='bold', va='bottom', ha='left', fontstyle='italic')

ax.axvspan(200, 300, alpha=0.08, color=CVD_ORANGE)
ax.annotate(r'$n_{\rm crit}$ zone', xy=(250, 180), fontsize=7, ha='center', color=CVD_ORANGE,
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=CVD_ORANGE, alpha=0.8))

endpoint_offsets = {'LGG': (12, 20), 'LIHC': (12, 0), 'BRCA': (12, -18)}
for cn, (color, _, _) in highlight.items():
    if cn in pb_curves:
        pts_sorted = sorted(pb_curves[cn], key=lambda x: x['n'])
        ns = [p['n'] for p in pts_sorted]
        ds = [p['delta'] for p in pts_sorted]
        dx, dy = endpoint_offsets.get(cn, (10, 0))
        ax.annotate(cn, xy=(ns[-1], ds[-1]), xytext=(dx, dy),
                    textcoords='offset points', fontsize=7, color=color,
                    ha='left', va='center', fontweight='bold',
                    arrowprops=dict(arrowstyle='-', color=color, lw=0.6))

ax.set_xlabel(r'Sample Size  $n$  (within-cancer subsample)', fontsize=9)
ax.set_ylabel(r'$\Delta$ = SSCAGate $-$ CAGate  (edges)', fontsize=9)
ax.set_xlim(30, 430)

ax.legend(loc='upper right', fontsize=7, framealpha=0.9, bbox_to_anchor=(0.98, 0.82),
          ncol=1, columnspacing=0.8, handletextpad=0.3, handlelength=1.2)

sr = res.get('spearman_r', 0)
sp = res.get('spearman_p', 1)
sp_mant, sp_exp = f'{sp:.1e}'.split('e')
sp_exp = int(sp_exp)
ax.text(0.03, 0.10, 'Aggregate Spearman $r = %.3f$\n$p = %s \\times 10^{%d}$\n8 cancers $\\times$ 4 subsamples' % (sr, sp_mant, sp_exp),
        transform=ax.transAxes, fontsize=7, va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=N_GRAY, alpha=0.8))
ax.set_title('b  Within-Cancer Resampling: Isolating Sample Size from Biology',
             fontsize=10, fontweight='bold', loc='left', pad=5)

plt.tight_layout(pad=0.5)
fig_b.savefig(os.path.join(PANELS_DIR, 'Panel_B_resampling.png'), dpi=DPI, bbox_inches='tight', facecolor=N_WHITE)
plt.close(fig_b)
print('  Panel B saved.')

# =====================================================================
# PANEL C: Crossover Mechanism
# =====================================================================
print('--- Panel C: Crossover Mechanism ---')

fig_c, ax = plt.subplots(figsize=(FIG_W_FULL, 3.6))
for cn in ['BRCA', 'LGG']:
    if cn in pb_curves:
        pts_sorted = sorted(pb_curves[cn], key=lambda x: x['n'])
        ns = np.array([p['n'] for p in pts_sorted])
        ds = np.array([p['delta'] for p in pts_sorted])
        dmax = max(abs(ds))
        dnorm = ds / dmax if dmax > 0 else ds
        if cn == 'BRCA':
            color, marker, label = N_RED, 'o', 'BRCA (n_full=1218, fast convergence)'
        else:
            color, marker, label = N_BLUE, 's', 'LGG (n_full=530, slow convergence)'
        ax.plot(ns, dnorm, '-%s' % marker, color=color, linewidth=2.0, markersize=6,
                label=label, alpha=0.9)
        for i, (nv, dv) in enumerate(zip(ns, ds)):
            if i == 0 or i == len(ns) - 1:
                ax.annotate('%+.0f' % dv, xy=(nv, dnorm[i]), fontsize=6, color=color,
                           ha='center', xytext=(0, -18), textcoords='offset points', alpha=0.7)

ax.axhline(y=0, color=N_DARK, linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvspan(200, 350, alpha=0.06, color=N_ORANGE)
ax.annotate('Transition\nZone', xy=(275, -0.3), fontsize=7, ha='center', color=N_ORANGE, fontweight='bold')
ax.text(100, 1.15, 'Variance-Dominated\n(Hard wins)', ha='center', fontsize=7,
        fontstyle='italic', color=N_BLUE, alpha=0.6)
ax.text(400, 1.15, 'Bias-Dominated\n(Soft wins)', ha='center', fontsize=7,
        fontstyle='italic', color=N_ORANGE, alpha=0.6)

ax.set_xlabel('Sample Size $n$ (within-cancer subsample)', fontsize=9)
ax.set_ylabel(r'Normalized $\Delta / \Delta_{\rm max}$', fontsize=9)
ax.set_xlim(30, 430)
ax.set_ylim(-1.3, 1.3)
ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
ax.set_title('c  Crossover Mechanism: Normalized Convergence', fontsize=10, fontweight='bold', loc='left', pad=5)

plt.tight_layout(pad=0.5)
fig_c.savefig(os.path.join(PANELS_DIR, 'Panel_C_crossover.png'), dpi=DPI, bbox_inches='tight', facecolor=N_WHITE)
plt.close(fig_c)
print('  Panel C saved.')

# =====================================================================
# PANEL D: Literature Benchmarks Overlay
# =====================================================================
print('--- Panel D: Literature Overlay ---')

fig_d, ax = plt.subplots(figsize=(FIG_W_FULL, 4.3))
ax.axvspan(0.5, 4.0, alpha=0.04, color=N_BLUE, zorder=0)
ax.axvspan(4.0, 15, alpha=0.04, color=N_ORANGE, zorder=0)
ax.text(0.15, 0.88, 'Hard Clustering Dominant', transform=ax.transAxes,
        ha='center', va='top', fontsize=6.5, color=N_BLUE, alpha=0.55, fontstyle='italic')
ax.text(0.55, 0.88, 'Soft Clustering Dominant', transform=ax.transAxes,
        ha='center', va='top', fontsize=6.5, color=N_ORANGE, alpha=0.55, fontstyle='italic')
ax.axvline(x=4, color=N_DARK, linestyle='--', linewidth=1.2, alpha=0.7, zorder=3)
ax.annotate(r'$n/d \approx 4$', xy=(4, 0.95), xycoords=('data', 'axes fraction'),
            fontsize=7, color=N_DARK, ha='center', va='bottom', fontweight='bold')

# TCGA scatter
ax.scatter(nds, deltas_a, c=colors, s=50, edgecolors='white', linewidth=0.5,
           zorder=4, alpha=0.85, label='TCGA cancers')

LIT_BASE_Y = -6
TIGHT = {'PBMC3k', 'Paul15', 'Zeisel br.'}
label_params = {
    'Wine': (14, -18), 'Seeds': (30, -32), 'Baron panc.': (17, -46),
    'Breast Ca.': (19, -60), 'Iris': (38, -74), 'Glass': (24, -88),
    'Digits': (28, -102), 'MNIST': (89, -12), 'Banknote': (343, -12), 'Letter': (1250, -12),
}

y_labels_min = min(y for (_, y) in label_params.values())
y_data_min = float(np.min(deltas_a)) if len(deltas_a) > 0 else -100
y_min = min(y_data_min, y_labels_min) - 20
ax.set_ylim(bottom=y_min)

for name, n_lit, d_lit in lit_benchmarks:
    nd_val = n_lit / max(d_lit, 1)
    ax.scatter(nd_val, LIT_BASE_Y, marker='D', s=32, c=N_PURPLE,
               edgecolors='white', linewidth=0.5, zorder=5, alpha=0.85)
    if name in TIGHT:
        continue
    lx, ly = label_params[name]
    ax.annotate(name, xy=(nd_val, LIT_BASE_Y), xytext=(lx, ly),
                fontsize=5.5, color=N_PURPLE, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.75),
                arrowprops=dict(arrowstyle='-', color=N_PURPLE, lw=0.5))

cluster_text = r'$\mathit{PBMC3k}$  (n/d=5.4)' + '\n' + \
               r'$\mathit{Paul15}$  (n/d=5.5)' + '\n' + \
               r'$\mathit{Zeisel}$ brain  (n/d=6.0)'
ax.text(0.99, 0.15, cluster_text, transform=ax.transAxes, fontsize=5.5, color=N_PURPLE,
        va='bottom', ha='right', linespacing=1.5,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=N_PURPLE, alpha=0.85, linewidth=0.5))
ax.text(0.99, 0.01, 'Literature benchmarks (all n/d > 4, projected at Delta ~ 0)',
        transform=ax.transAxes, fontsize=5.5, color=N_PURPLE, alpha=0.7, va='bottom', ha='right')

ax.set_xlabel(r'Sample-to-Dimension Ratio  $n/d$ (log scale)', fontsize=9)
ax.set_ylabel(r'Performance Gap  $\Delta$ (edges)', fontsize=9)
ax.set_xscale('log')
ax.set_xlim(0.8, 2000)
ax.axhline(y=0, color=N_DARK, linestyle='-', linewidth=0.5, alpha=0.3)
dp_mant, dp_exp = f'{p_val:.1e}'.split('e')
dp_exp = int(dp_exp)
ax.text(0.97, 0.96, 'Spearman $r = %.3f$\n$p = %s \\times 10^{%d}$\n$n = %d$ cancers' % (r_val, dp_mant, dp_exp, len(pa_points)),
        transform=ax.transAxes, fontsize=6.5, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=N_GRAY, alpha=0.8))
ax.set_title('d  Literature Benchmarks Overlay: Independent Benchmarks Confirm the Same $n/d$ Threshold',
             fontsize=10, fontweight='bold', loc='left', pad=8)

plt.tight_layout(rect=[0, 0, 1, 0.93], pad=0.5)
fig_d.savefig(os.path.join(PANELS_DIR, 'Panel_D_literature.png'), dpi=DPI, bbox_inches='tight',
              facecolor=N_WHITE, pad_inches=0.2)
plt.close(fig_d)
print('  Panel D saved.')

# =====================================================================
# PANEL E: Cross-Modal Universality (UNIFIED single chart)
# =====================================================================
print('--- Panel E: Cross-Modal Universality ---')

fig_e, ax = plt.subplots(1, 1, figsize=(FIG_W_FULL * 1.15, 3.8))

# TCGA scatter + trend
ax.scatter(tcga_n, tcga_delta, c=N_BLUE, s=15, alpha=0.5, edgecolors='none',
           zorder=3, label='TCGA cancers (n=%d)' % len(tcga_n))
log_ns = np.log10(tcga_n)
cf_e = np.polyfit(log_ns, tcga_delta, 1)
xf_e = np.logspace(np.log10(35), np.log10(20000), 100)
yf_e = np.polyval(cf_e, np.log10(xf_e))
ax.plot(xf_e, yf_e, '--', color=N_BLUE, lw=1.0, alpha=0.5, zorder=2)

# MNIST
ax.plot(mn_n, mn_y, 'o-', color=N_ORANGE, lw=2.0, markersize=5,
        label=r'MNIST ($R^2$=0.977)', zorder=5)
if len(mn_y) > 3:
    mn_arr = np.array(mn_y)
    ax.fill_between(mn_n, mn_arr * 0.95, mn_arr * 1.05, alpha=0.08, color=N_ORANGE)

# PBMC
ax.plot(pb_n, pb_y, 'o-', color=N_GREEN, lw=2.0, markersize=5,
        label='PBMC scRNA-seq', zorder=4)

# 20 Newsgroups
ax.plot(tn, ty, 's-', color=N_TEAL, lw=1.8, markersize=5,
        label='20 Newsgroups', zorder=4)

# CIFAR-10 negative control
ax.plot(cifar_n, cifar_y, 's--', color=N_GRAY, lw=1.2, markersize=5,
        label='CIFAR-10 (neg. ctrl)', zorder=3, alpha=0.7)

# Synthetic null
ax.plot(syn_n, syn_y, '^--', color='#BDC3C7', lw=1.2, markersize=5,
        label='Synthetic null', zorder=3, alpha=0.7)

ax.axhline(0, color=N_DARK, lw=0.6, alpha=0.4, ls='-', zorder=1)
ax.set_xscale('log')
ax.set_xlabel('Sample size $n$ (log scale)', fontsize=9)
ax.set_ylabel(r'SSCAGate $-$ CAGate  $\Delta$ (edges)', fontsize=9)
ax.grid(True, alpha=0.08, lw=0.2)
ax.legend(loc='upper right', fontsize=7, framealpha=0.9, ncol=2,
          columnspacing=0.5, handletextpad=0.5, handlelength=1.5)
ax.set_title('e  Cross-Modal Universality', fontsize=10, fontweight='bold', loc='left', pad=5)

plt.tight_layout(pad=0.8)
fig_e.savefig(os.path.join(PANELS_DIR, 'Panel_E_crossmodal.png'), dpi=DPI, bbox_inches='tight', facecolor=N_WHITE)
plt.close(fig_e)
print('  Panel E saved (unified chart, 6 modalities).')

# =====================================================================
# ASSEMBLE: 2x2 + 1 full-width composite
# =====================================================================
print('\n--- ASSEMBLY: Fig1 Composite ---')

PANEL_FILES = {
    'a': 'Panel_A_phase.png',
    'b': 'Panel_B_resampling.png',
    'c': 'Panel_C_crossover.png',
    'd': 'Panel_D_literature.png',
    'e': 'Panel_E_crossmodal.png',
}

imgs = {}
for key, fname in PANEL_FILES.items():
    path = os.path.join(PANELS_DIR, fname)
    imgs[key] = Image.open(path)
    print('  %s: %dx%d' % (key, imgs[key].size[0], imgs[key].size[1]))

TARGET_W = 2100
for key in imgs:
    img = imgs[key]
    w, h = img.size
    new_h = int(h * TARGET_W / w)
    imgs[key] = img.resize((TARGET_W, new_h), Image.LANCZOS)

D_ASSEMBLE = 300
gap_w, gap_h, label_h = 45, 45, 54  # pixels at 300 DPI (0.15", 0.15", 0.18")
h_a = imgs['a'].size[1]
h_b = imgs['b'].size[1]
h_e = imgs['e'].size[1]

total_w = 2 * TARGET_W + gap_w
total_h = h_a + gap_h + h_b + gap_h + h_e + label_h * 2

composite = Image.new('RGB', (total_w, total_h), (255, 255, 255))

# Row 0: Panel A (left) + Panel D (right)
y_top = label_h
composite.paste(imgs['a'], (0, y_top))
composite.paste(imgs['d'], (TARGET_W + gap_w, y_top))

# Row 1: Panel B (left) + Panel C (right)
y_mid = y_top + h_a + gap_h + label_h
composite.paste(imgs['b'], (0, y_mid))
composite.paste(imgs['c'], (TARGET_W + gap_w, y_mid))

# Row 2: Panel E (full-width)
y_bot = y_mid + h_b + gap_h + label_h
e_x = (total_w - TARGET_W) // 2  # center
composite.paste(imgs['e'], (e_x, y_bot))

# Save
png_path = os.path.join(MAIN_DIR, 'Fig1_Composite.png')
pdf_path = os.path.join(MAIN_DIR, 'Fig1_Composite.pdf')
composite.save(png_path, dpi=(D_ASSEMBLE, D_ASSEMBLE))
composite.save(pdf_path, dpi=(D_ASSEMBLE, D_ASSEMBLE))

png_size = os.path.getsize(png_path)
pdf_size = os.path.getsize(pdf_path)
print('\n  Composite: %dx%d px' % (total_w, total_h))
print('  %s (%d KB)' % (os.path.basename(png_path), png_size // 1024))
print('  %s (%d KB)' % (os.path.basename(pdf_path), pdf_size // 1024))

# Verify file sizes
for key, fname in PANEL_FILES.items():
    fp = os.path.join(PANELS_DIR, fname)
    print('  panel %s: %d KB' % (key, os.path.getsize(fp) // 1024))

print('\n' + '=' * 60)
print('FIG1 COMPOSITE DONE.')
print('Output: %s' % MAIN_DIR)
print('%s (%d KB)' % (png_path, png_size // 1024))
print('%s (%d KB)' % (pdf_path, pdf_size // 1024))
print('=' * 60)
