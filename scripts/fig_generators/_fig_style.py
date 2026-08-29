"""Shared figure style for the SSCAGate figures (top-journal quality).

A refined, coherent palette built on the Okabe-Ito colorblind-safe scheme,
with a soft neutral background, hairline grid, and consistent type sizing.
Semantic mapping (kept identical across every panel so the reader can learn
it once and read every figure):

    BLUE   = baseline / global method / reconstruction-loss methods
    GREEN  = per-cluster / partition-then-fit (the remedy)
    ORANGE = oracle / highlight / positive signal
    RED    = gating / soft- or hard-gate / the pitfall / warning
    GREY   = neutral trend / reference (e.g. y=x)
"""
import matplotlib as mpl

# ---- Unified palette (Okabe-Ito, slightly tuned for print) ---------------
BLUE   = '#2E7AB0'   # baseline / global / reconstruction-loss family
GREEN  = '#20A65D'   # per-cluster / partition-then-fit (remedy)
ORANGE = '#E08A2E'   # oracle / highlight / positive
RED    = '#C94A3D'   # gating / the pitfall / warning
GREY   = '#6E6E78'   # neutral trend / reference line
GREY_L = '#C4C8CE'   # light grid / diagonal
INK    = '#1A1B1E'   # text

BG     = '#FFFFFF'   # panel background (kept white for crispness)


# ---- Base style ------------------------------------------------------------
def apply():
    mpl.rcParams.update({
        'font.size': 8.2,
        'axes.labelsize': 8.8,
        'axes.titlesize': 9.0,
        'xtick.labelsize': 7.0,
        'ytick.labelsize': 7.4,
        'legend.fontsize': 6.8,
        'axes.linewidth': 0.9,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'text.color': INK,
        'font.family': 'DejaVu Sans',
        'font.weight': 'normal',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.which': 'both',
        'grid.color': '#E4E7EA',
        'grid.linewidth': 0.6,
        'legend.frameon': False,
        'legend.edgecolor': 'none',
        'figure.facecolor': BG,
        'axes.facecolor': BG,
        'savefig.facecolor': BG,
        'pdf.fonttype': 42,   # embed TrueType fonts (No subsetting, editable)
        'ps.fonttype': 42,
        'lines.markersize': 3.6,
        'savefig.dpi': 300,
    })


def panel_label(ax, letter):
    """Bold panel letter pinned to the top-left, slightly outside the axes."""
    ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontsize=13,
            fontweight='bold', va='top', ha='left', color=INK, zorder=10)
