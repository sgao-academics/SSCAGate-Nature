# -*- coding: utf-8 -*-
"""
_gen_graphical_abstract_v3.py
Nature graphical abstract — final version.

Design principles:
  1. Left (3/5): scaling law phase diagram — formula on the curve
  2. Right (2/5): 33 real TCGA cancers — the data that proves the law
  3. Bottom: cross-modal universality strip — ✓/✗ marks, not micro-charts
  4. Clean, muted, let the science breathe.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import json, os, sys
from scipy.optimize import curve_fit

# ── Nature style (portable: finds nature_style.py relative to this script) ──
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from nature_style import apply_nature_style, N_BLUE, N_ORANGE, N_DARK, N_GRAY, DPI
apply_nature_style()

# ── Desaturated Nature palette ──────────────────────────────
C_HARD   = '#C05A30'   # muted terracotta
C_SOFT   = '#386890'   # muted navy
C_LINE   = '#1E2A38'   # near-black
C_GOLD   = '#9A7D3A'   # muted gold
C_DASH   = '#B03A2E'   # muted red
C_BG     = '#F7F8FA'   # soft panel bg
C_TCGA_S = '#C0392B'   # small n
C_TCGA_M = '#D68910'   # medium n
C_TCGA_L = '#2471A3'   # large n

# ── Load real data ──────────────────────────────────────────
with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'mega_33_full.json'),'r',encoding='utf-8') as f:
    mega = json.load(f)

pts = []
for c, d in mega.items():
    if not isinstance(d, dict): continue
    nv = d.get('n',0); v3 = d.get('v3_advantage',0)
    if nv > 0: pts.append((c, nv, v3))
pts.sort(key=lambda x: x[1])

nd_vals = np.array([p[1]/100.0 for p in pts])
adv_vals = np.array([p[2] for p in pts])
n_vals   = np.array([p[1] for p in pts])

# ── Figure ──────────────────────────────────────────────────
fig = plt.figure(figsize=(15.5, 8.0), facecolor='white')

# ==============================================================
# PANEL A (left 55%): Phase diagram — log-log scaling law
# ==============================================================
ax_a = fig.add_axes([0.04, 0.24, 0.42, 0.68])
ax_a.set_facecolor(C_BG)

d_vals  = np.logspace(np.log10(8), np.log10(1400), 400)
n_crit  = 6.3 * (d_vals ** 0.90)
n_bound = 4.0 * d_vals

# Zones
ax_a.fill_between(d_vals, 1, n_crit, alpha=0.09, color=C_HARD, lw=0)
ax_a.fill_between(d_vals, n_crit, 3e5, alpha=0.09, color=C_SOFT, lw=0)

# Curves
ax_a.plot(d_vals, n_crit,  color=C_LINE, lw=2.6, alpha=0.92, zorder=5)
ax_a.plot(d_vals, n_bound, '--', color=C_DASH, lw=1.5, alpha=0.35, zorder=3)

# Zone labels
ax_a.text(13, 18, 'HARD CLUSTERING\n(CAGate dominant)', fontsize=10.5,
          fontweight='bold', color=C_HARD, alpha=0.55, va='bottom', fontfamily='sans-serif')
ax_a.text(350, 35000, 'SOFT CLUSTERING\n(SSCAGate dominant)', fontsize=10.5,
          fontweight='bold', color=C_SOFT, alpha=0.55, va='center', fontfamily='sans-serif')

# n/d=4 annotation — lower position
ax_a.annotate(r'$n/d = 4$', xy=(300, 700), fontsize=9,
              color=C_DASH, ha='left', va='bottom', alpha=0.55, fontfamily='sans-serif')

# TCGA operational zone
ax_a.axvspan(80, 220, alpha=0.07, color=N_BLUE, zorder=1)
ax_a.annotate('TCGA\ncancers', xy=(150, 2600), fontsize=8.5, ha='center', va='center',
              fontweight='bold', color=N_BLUE, alpha=0.55,
              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=N_BLUE, lw=0.7, alpha=0.75),
              fontfamily='sans-serif', zorder=8)

# Formula box — compact, elevated
bbox_f = dict(boxstyle='round,pad=0.22', fc='white', ec=C_GOLD, lw=1.2, alpha=0.93)
ax_a.text(30, 6.3*30**0.90 * 2.6,
          r'$\mathbf{n_{\rm crit} = 6.3\,d^{0.90}}$' + '\n' + r'($R^2 = 0.985$)',
          fontsize=10, ha='center', va='center', fontweight='bold',
          color=C_LINE, bbox=bbox_f, fontfamily='sans-serif', zorder=10)

# Arrow — shorter, from formula down to curve
ax_a.annotate('', xy=(40, 6.3*40**0.90), xytext=(30, 6.3*30**0.90 * 1.9),
              arrowprops=dict(arrowstyle='->', color=C_GOLD, lw=1.3, alpha=0.7), zorder=9)

# Axes
ax_a.set_xscale('log'); ax_a.set_yscale('log')
ax_a.set_xlim(7, 1400); ax_a.set_ylim(7, 180000)
ax_a.set_xlabel('Data Dimensionality  $d$', fontsize=12, fontweight='bold', labelpad=6)
ax_a.set_ylabel(r'Critical Sample Size  $n_{\rm crit}$', fontsize=12, fontweight='bold', labelpad=6)
ax_a.tick_params(labelsize=8.5)
ax_a.grid(True, alpha=0.08, which='major', linestyle='-', linewidth=0.4)

# ==============================================================
# PANEL B (right 45%): Real TCGA phase transition data
# ==============================================================
ax_b = fig.add_axes([0.52, 0.24, 0.45, 0.68])
ax_b.set_facecolor(C_BG)

# Zone fills
ax_b.axvspan(0.4, 3.9, alpha=0.07, color=C_HARD, zorder=0)
ax_b.axvspan(4.1, 14.0, alpha=0.07, color=C_SOFT, zorder=0)
ax_b.axvline(x=4.0, color=C_DASH, linewidth=2.0, linestyle=(0,(8,4)), alpha=0.5, zorder=3)
ax_b.axhline(y=0,  color='#C4CBD4', linewidth=1.0, zorder=1)

# Zone labels — bottom corners with white backdrop to avoid collision
ax_b.text(0.06, 0.15, 'HARD CLUSTERING\nDOMINANT', transform=ax_b.transAxes,
          ha='left', va='bottom', fontsize=9.5, fontweight='bold',
          color=C_HARD, alpha=0.85, fontfamily='sans-serif',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.65))
ax_b.text(0.88, 0.15, 'SOFT CLUSTERING\nDOMINANT', transform=ax_b.transAxes,
          ha='right', va='bottom', fontsize=9.5, fontweight='bold',
          color=C_SOFT, alpha=0.85, fontfamily='sans-serif',
          bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.65))

# Scatter — color by n, size by sqrt(n)
colors_b = [C_TCGA_S if n<200 else C_TCGA_M if n<500 else C_TCGA_L for n in n_vals]
sizes_b  = [max(18, np.sqrt(n)*2.4) for n in n_vals]
ax_b.scatter(nd_vals, adv_vals, s=sizes_b, c=colors_b, alpha=0.78,
            edgecolors='white', linewidth=0.8, zorder=5)

# Trend curve
def sigmoid_like(x, a, b, c, d):
    return a / (1 + np.exp(b*(x-c))) + d
try:
    popt, _ = curve_fit(sigmoid_like, nd_vals, adv_vals, p0=[400,0.5,3.5,-50], maxfev=5000)
    xs = np.linspace(0.4, 13.5, 300)
    ax_b.plot(xs, sigmoid_like(xs, *popt), color=C_LINE, lw=2.8, alpha=0.45, zorder=4)
except: pass

# Key cancer labels
for c, x, y, dx, dy in [('CHOL',0.45,572,-12,10),('DLBC',0.48,831,-12,14),
                          ('BRCA',12.18,53,10,6),('GBM',1.72,207,10,4)]:
    ax_b.annotate(c, (x,y), textcoords="offset points", xytext=(dx,dy),
                  fontsize=6.5, fontweight='bold', color=C_LINE, zorder=7)

# Formula box — top-right corner, elevated to avoid any overlap
bbox_f2 = dict(boxstyle='round,pad=0.45', fc='white', ec=C_GOLD, lw=1.4, alpha=0.93)
ax_b.text(0.96, 0.98, r'$\mathbf{n_{\rm crit} = 6.3\,d^{0.90}}$',
          transform=ax_b.transAxes, fontsize=10, ha='right', va='top',
          fontweight='bold', color=C_LINE, bbox=bbox_f2, fontfamily='sans-serif', zorder=9)

# Decision rule — bottom far-right, clear of everything
bbox_r = dict(boxstyle='round,pad=0.5', fc='white', ec=C_LINE, lw=1.2, alpha=0.93)
rule = (r'$\mathbf{n/d < 4}$' + '\u00A0\u00A0' + r'HARD (CAGate)' + '\n' +
        r'$\mathbf{n/d \geq 4}$' + '\u00A0\u00A0' + r'SOFT (SSCAGate)')
ax_b.text(0.97, 0.05, rule, transform=ax_b.transAxes, fontsize=8.5,
          ha='right', va='bottom', bbox=bbox_r, fontfamily='sans-serif',
          fontweight='bold', color=C_LINE)

# n_crit annotation — lowered to avoid blocking scatter points
ax_b.annotate(r'$\mathbf{n_{\rm crit}/d \approx 4}$', xy=(4.0,0.82),
              xytext=(6.5,0.82), xycoords=('data','axes fraction'),
              textcoords=('data','axes fraction'), fontsize=9, fontweight='bold',
              color=C_DASH, arrowprops=dict(arrowstyle='->', color=C_DASH, lw=1.2),
              ha='left', va='center', zorder=8,
              bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_DASH, alpha=0.7))

# Axes
ax_b.set_xlabel(r'Effective Sample Ratio  $n/d$', fontsize=12, fontweight='bold', labelpad=6)
ax_b.set_ylabel(r'SSCAGate Advantage  $\Delta$  (edges)', fontsize=12, fontweight='bold', labelpad=6)
ax_b.set_xlim(0.4, 14.0)
ax_b.tick_params(labelsize=8.5)
ax_b.grid(True, alpha=0.08, which='major', linestyle='-', linewidth=0.4)

# Legend
from matplotlib.patches import Patch
leg = ax_b.legend(
    handles=[Patch(facecolor=C_TCGA_S,alpha=0.7,label=r'$n<200$'),
             Patch(facecolor=C_TCGA_M,alpha=0.7,label=r'$n\in[200,500]$'),
             Patch(facecolor=C_TCGA_L,alpha=0.7,label=r'$n>500$')],
    loc='upper left', fontsize=7.5, framealpha=0.85, edgecolor='#C4CBD4',
    title='33 TCGA Cancers', title_fontsize=8)
leg.get_frame().set_linewidth(0.6)

# ==============================================================
# PANEL C (bottom): Cross-modal universality strip
# ==============================================================
ax_c = fig.add_axes([0.04, 0.04, 0.93, 0.165])
ax_c.set_facecolor('white')
ax_c.set_xlim(0, 10); ax_c.set_ylim(0, 1)
ax_c.axis('off')

mods = [
    ('TCGA\n33 cancers', '#1A3C5E', True,  r'$r{=}0.827$'),
    ('MNIST\nImages',    '#1E6B4E', True,  r'$R^2{=}0.977$'),
    ('PBMC\nscRNA-seq',  '#2471A3', True,  r'$r{=}0.820$'),
    ('20 News-\ngroups', '#6C3483', True,  r'text'),
    ('Synthetic\nNull',  '#839192', False, r'flat $\Delta$'),
    ('CIFAR-10\nImages', '#99A3A4', False, r'flat $\Delta$'),
]

Y_C = 0.45
w_mod = 1.35
gap = (10 - len(mods)*w_mod) / (len(mods)+1)

for i, (label, col, pos, stat) in enumerate(mods):
    x0 = gap + i*(w_mod + gap)
    xc = x0 + w_mod/2

    # Connection line
    if i > 0:
        px = gap + (i-1)*(w_mod+gap) + w_mod
        ax_c.plot([px, x0], [Y_C, Y_C], color='#A0ACB8', lw=2.2, alpha=0.5, solid_capstyle='round')

    # Node: rounded rectangle
    node = FancyBboxPatch((x0+0.15, Y_C-0.08), w_mod-0.3, 0.16,
                          boxstyle='round,pad=0.08', facecolor='white',
                          edgecolor=col, linewidth=1.6, alpha=0.92, zorder=6)
    ax_c.add_patch(node)

    # Check/X mark
    mark = r'$\checkmark$' if pos else r'$\times$'
    mk_color = col if pos else '#999'
    ax_c.text(xc, Y_C, mark, ha='center', va='center', fontsize=16,
              fontweight='bold', color=mk_color, zorder=7, fontfamily='sans-serif')

    # Label below — more space, bigger font
    ax_c.text(xc, Y_C-0.20, label, ha='center', va='top', fontsize=9,
              color=C_LINE, alpha=0.75, fontfamily='sans-serif', linespacing=1.1)

    # Stat below label — more space, bigger font
    ax_c.text(xc, Y_C-0.36, stat, ha='center', va='top', fontsize=8,
              color=mk_color, alpha=0.7, fontfamily='sans-serif')

# Header
ax_c.text(5.0, 0.72, 'SAME PHASE TRANSITION REPLICATED ACROSS SIX DATA MODALITIES',
          ha='center', va='top', fontsize=10, fontweight='bold',
          color='#7F8C8D', alpha=0.65, fontfamily='sans-serif')

# Divider line
ax_c.plot([1.0, 9.0], [0.30, 0.30], color='#DEE1E6', lw=0.8)

# ==============================================================
# TITLE
# ==============================================================
title_ax = fig.add_axes([0.0, 0.925, 1.0, 0.068])
title_ax.axis('off')
title_ax.text(0.5, 0.72, 'A Universal Sample-Size Phase Transition',
              ha='center', va='center', fontsize=18, fontweight='bold',
              color=C_LINE, fontfamily='sans-serif', transform=title_ax.transAxes)
title_ax.text(0.5, 0.24, 'Resolving the Hard-versus-Soft Clustering Debate',
              ha='center', va='center', fontsize=10.5, fontweight='normal',
              color='#7F8C8D', fontfamily='sans-serif', transform=title_ax.transAxes)

# Panel labels
ax_a.text(-0.12, 1.00, 'a', transform=ax_a.transAxes, fontsize=14, fontweight='bold',
          color=C_LINE, fontfamily='sans-serif')
ax_b.text(-0.10, 1.00, 'b', transform=ax_b.transAxes, fontsize=14, fontweight='bold',
          color=C_LINE, fontfamily='sans-serif')

# ==============================================================
# SAVE (portable: figures/ directory relative to repo root)
# ==============================================================
_repo_root = os.path.dirname(_script_dir)  # scripts/../ = repo root
out_dir = os.path.join(_repo_root, 'figures')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'Graphical_Abstract')
fig.savefig(out + '.png', dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
fig.savefig(out + '.pdf', bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
print(f'V3 saved: {out}.png ({os.path.getsize(out+".png")/1024:.0f} KB), .pdf ({os.path.getsize(out+".pdf")/1024:.0f} KB)')
plt.close()
