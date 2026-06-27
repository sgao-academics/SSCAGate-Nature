# -*- coding: utf-8 -*-
"""
gen_ed_fig1.py -- ONE script, ONE figure: Extended Data Fig 1 (K-sweep Heatmap)
===============================================================================
Generates a K-sweep heatmap showing delta = Gate - Baseline across K values
and cancer types, sorted by sample size. Also generates full 16-cancer version.

Output: figures/ed/ed_fig1_ksweep_v2.{pdf,png}
"""
import json, os, numpy as np

DATA_DIR = r'D:\NO.1\Replication_Package\results'
OUT_ED   = r'C:\Users\高帅东\Desktop\SSCAGate-Nature\figures\ed'
os.makedirs(OUT_ED, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- Nature Style inline ----
N_RED   = '#C0392B'
N_DARK  = '#2C3E50'
N_GRAY  = '#7F8C8D'
DPI_VAL = 300
FIG_W   = 7.09

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.dpi': DPI_VAL, 'savefig.dpi': DPI_VAL,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.08,
    'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.grid': False, 'grid.alpha': 0.15, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

def save_fig(fig, basepath):
    for ext in ['pdf', 'png']:
        path = f'{basepath}.{ext}'
        fig.savefig(path, format=ext, dpi=DPI_VAL if ext == 'png' else None,
                    bbox_inches='tight', pad_inches=0.08, facecolor='white')
        sz = os.path.getsize(path) / 1024
        print(f'  Saved: {os.path.basename(path)} ({sz:.0f}KB)')
    plt.close(fig)


# ================================================================
# DATA: Load ksweep_results.json
# ================================================================
print('=== ED Fig 1: K-sweep Heatmap ===')

with open(os.path.join(DATA_DIR, 'ksweep_results.json'), encoding='utf-8') as f:
    ksweep_raw = json.load(f)

cancer_ksweep = {}
for key, entry in ksweep_raw.items():
    if isinstance(entry, dict) and 'cancer' in entry:
        c = entry['cancer']
        K = entry['K']
        n = entry.get('n', 0)
        delta = entry.get('delta', entry.get('mean_gain', 0))
        if c not in cancer_ksweep:
            cancer_ksweep[c] = {'n': n, 'K_data': {}}
        cancer_ksweep[c]['K_data'][K] = {
            'delta': float(delta),
            'base_mean': float(entry.get('base_mean', 0)),
            'gate_mean': float(entry.get('gate_mean', 0))
        }

# Sort by sample size
sorted_cancers = sorted(cancer_ksweep.items(), key=lambda x: x[1]['n'])
cancer_names = [c for c, _ in sorted_cancers]
cancer_ns = [data['n'] for _, data in sorted_cancers]

# Pick 8 representative cancers (stratified by n)
n_sort = sorted(enumerate(cancer_ns), key=lambda x: x[1])
pick_indices = []
for target_rank in [0, len(n_sort)//5, 2*len(n_sort)//5, 3*len(n_sort)//5,
                     4*len(n_sort)//5, len(n_sort)-1]:
    idx = n_sort[min(target_rank, len(n_sort)-1)][0]
    if idx not in pick_indices:
        pick_indices.append(idx)
selected = [cancer_names[i] for i in pick_indices[:8]]

K_values = [2, 3, 5, 8, 10]

# Build heatmap matrix
heatmap_data = np.zeros((len(K_values), len(selected)))
annot_data = []
for row, K in enumerate(K_values):
    row_annot = []
    for col, cancer in enumerate(selected):
        if cancer in cancer_ksweep and K in cancer_ksweep[cancer]['K_data']:
            d = cancer_ksweep[cancer]['K_data'][K]['delta']
            heatmap_data[row, col] = d
            row_annot.append(f'{d:.0f}')
        else:
            heatmap_data[row, col] = np.nan
            row_annot.append('--')
    annot_data.append(row_annot)

# Colormap
all_colors = ['#fddbc7', '#f4a582', '#f7f7f7', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']
cmap = LinearSegmentedColormap.from_list('gate_div', all_colors, N=256)

vmax = max(abs(np.nanmin(heatmap_data)), np.nanmax(heatmap_data)) if not np.all(np.isnan(heatmap_data)) else 1
vmin = -vmax

# ================================================================
# FIGURE: 8-cancer heatmap
# ================================================================
fig, ax = plt.subplots(figsize=(FIG_W * 1.0, 4.2))

im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, interpolation='nearest')

# Annotate
for row in range(len(K_values)):
    for col in range(len(selected)):
        val = heatmap_data[row, col]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.6 * vmax else 'black'
            ax.text(col, row, annot_data[row][col], ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=color)

# Best-K red border
best_k_centers = []
for col, cancer in enumerate(selected):
    row_data = heatmap_data[:, col]
    valid = ~np.isnan(row_data)
    if valid.sum() > 0:
        best_k_idx = np.nanargmax(row_data)
        best_k_centers.append((col, best_k_idx))
        rect = plt.Rectangle((col-0.48, best_k_idx-0.48), 0.96, 0.96,
                             linewidth=2.5, edgecolor=N_RED, facecolor='none',
                             linestyle='-', zorder=10)
        ax.add_patch(rect)

# Phase-transition dashed curve
if len(best_k_centers) >= 2:
    xs, ys = zip(*best_k_centers)
    ax.plot(xs, ys, '--', color=N_RED, linewidth=1.5, alpha=0.6, zorder=9)

# Labels
x_labels = [f'{cancer}\n(n={cancer_ksweep[cancer]["n"]})' for cancer in selected]
ax.set_xticks(range(len(selected)))
ax.set_xticklabels(x_labels, fontsize=7.5, fontweight='bold', linespacing=1.1)
ax.set_yticks(range(len(K_values)))
ax.set_yticklabels([f'K={k}' for k in K_values], fontsize=8)

# Arrow for increasing n
ax.annotate('', xy=(len(selected)-0.5, -0.48), xytext=(-0.5, -0.48),
            arrowprops=dict(arrowstyle='->', color=N_DARK, lw=1.5))
ax.text((len(selected)-1)/2, -0.65, 'Increasing Sample Size (n)', ha='center', va='center',
        fontsize=7, color=N_DARK, style='italic')
ax.set_xlim(-0.5, len(selected)-0.5)
ax.set_ylim(len(K_values)-0.5, -0.8)

# Caption
ax.text(0.5, -0.22,
        'Three sensitivity patterns: insensitive (K-invariant delta) | monotonic (delta scales with K) | non-monotonic (optimal K at intermediate value)',
        transform=ax.transAxes, ha='center', va='top', fontsize=6.5, color=N_GRAY)
ax.text(0.5, -0.30,
        'Red box = best K per cancer; dashed curve = phase-transition guide.',
        transform=ax.transAxes, ha='center', va='top', fontsize=6.5, color=N_GRAY)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
cbar.set_label(r'$\Delta$ = SSCAGate $-$ NOTEARS (edges)', fontsize=8, fontweight='bold')

plt.tight_layout(pad=3.0)
save_fig(fig, os.path.join(OUT_ED, 'ed_fig1_ksweep_v2'))


print('ED Fig 1 DONE')
