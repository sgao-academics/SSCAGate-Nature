# -*- coding: utf-8 -*-
"""DepMap external validation of cluster-only edges (P0-b, stage 2).

Reads results/pan_open.json (produced by _pan_cancer_open_discovery.py), then,
for each cancer and pooled across cancers, tests whether the edges recovered
ONLY by cluster decomposition (per-cluster minus baseline) carry a stronger
co-dependency signal in DepMap CRISPR gene effect than (a) the baseline edges
and (b) random gene pairs.

Co-essentiality metric: for a directed pair (i->j), correlation of the CRISPR
gene-effect vectors of gene i and gene j across cell lines. A genuine regulatory
edge should show higher |correlation| than a random pair.

Deterministic: fixed seed for random-pair sampling. Pure read/compute, no fitting.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, 'results')
CP = os.path.join(RESULTS, 'pan_open.json')
DATA_DIR = r'../../data'
DEPMAP = os.path.join(DATA_DIR, 'depmap', 'CRISPRGeneEffect.csv')
OUT = os.path.join(RESULTS, 'pan_open_depmap_validation.json')


def load_depmap_effect():
    df = pd.read_csv(DEPMAP, index_col=0)
    sym = [c.split(' (')[0] for c in df.columns]
    df.columns = sym
    # keep genes that appear exactly once (drop duplicate symbols)
    dup = df.columns[df.columns.duplicated(keep=False)]
    df = df.loc[:, ~df.columns.isin(dup)]
    X = df.values.astype(np.float64)
    X = X - np.nanmean(X, axis=0, keepdims=True)
    X = X / (np.nanstd(X, axis=0, keepdims=True) + 1e-8)
    return X, list(df.columns)


def corr_of_pair(X, g2i, a, b):
    if a not in g2i or b not in g2i:
        return np.nan
    ia, ib = g2i[a], g2i[b]
    xa, xb = X[:, ia], X[:, ib]
    mask = ~(np.isnan(xa) | np.isnan(xb))
    if mask.sum() < 20:
        return np.nan
    c = np.corrcoef(xa[mask], xb[mask])[0, 1]
    return abs(c)


def main():
    cp = json.load(open(CP, encoding='utf-8'))
    cells = cp.get('cells', {})
    X, genes = load_depmap_effect()
    g2i = {g: i for i, g in enumerate(genes)}
    print('DepMap effect: %d cell lines x %d genes' % (X.shape[0], X.shape[1]))

    rng = np.random.RandomState(42)
    pooled_only = []   # (cancer, pair, coess)
    pooled_base = []
    pooled_rand = []

    per_cancer = {}
    for key, cell in sorted(cells.items()):
        if cell.get('skipped'):
            continue
        cancer = cell['cancer']
        genes_c = cell['genes']
        # map recovered edge (idx) -> gene symbol
        def sym(edge):
            i, j = edge
            return genes_c[i], genes_c[j]
        only_edges = [sym(e) for e in cell['cluster_only_edges']]
        base_edges = [sym(e) for e in cell['baseline_edges']]
        # random pairs: same count as only_edges, drawn from genes_c
        rand_pairs = []
        while len(rand_pairs) < len(only_edges):
            a, b = rng.choice(len(genes_c), 2, replace=False)
            rand_pairs.append((genes_c[a], genes_c[b]))

        co_only = [corr_of_pair(X, g2i, a, b) for (a, b) in only_edges]
        co_base = [corr_of_pair(X, g2i, a, b) for (a, b) in base_edges]
        co_rand = [corr_of_pair(X, g2i, a, b) for (a, b) in rand_pairs]
        co_only = [c for c in co_only if not np.isnan(c)]
        co_base = [c for c in co_base if not np.isnan(c)]
        co_rand = [c for c in co_rand if not np.isnan(c)]

        per_cancer[cancer] = {
            'n_only': len(co_only), 'n_base': len(co_base), 'n_rand': len(co_rand),
            'mean_only': round(float(np.mean(co_only)), 4) if co_only else None,
            'mean_base': round(float(np.mean(co_base)), 4) if co_base else None,
            'mean_rand': round(float(np.mean(co_rand)), 4) if co_rand else None,
        }
        pooled_only.extend(co_only)
        pooled_base.extend(co_base)
        pooled_rand.extend(co_rand)

    out = {'per_cancer': per_cancer}
    if pooled_only and pooled_rand:
        out['pooled'] = {
            'mean_only': round(float(np.mean(pooled_only)), 4),
            'mean_base': round(float(np.mean(pooled_base)), 4),
            'mean_rand': round(float(np.mean(pooled_rand)), 4),
            'n_only': len(pooled_only), 'n_base': len(pooled_base), 'n_rand': len(pooled_rand),
        }
        # Mann-Whitney U test: only vs rand, only vs base
        try:
            from scipy import stats
            u_or, p_or = stats.mannwhitneyu(pooled_only, pooled_rand, alternative='greater')
            u_ob, p_ob = stats.mannwhitneyu(pooled_only, pooled_base, alternative='greater')
            out['pooled']['p_only_vs_rand'] = float(p_or)
            out['pooled']['p_only_vs_base'] = float(p_ob)
            print('\nPOOLED: only=%.4f (n=%d)  base=%.4f (n=%d)  rand=%.4f (n=%d)' % (
                out['pooled']['mean_only'], out['pooled']['n_only'],
                out['pooled']['mean_base'], out['pooled']['n_base'],
                out['pooled']['mean_rand'], out['pooled']['n_rand']))
            print('MWU only>rand: p=%.3g   only>base: p=%.3g' % (p_or, p_ob))
        except Exception as e:
            print('stats error:', e)

    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
    print('\nSAVED', OUT)
    print('per-cancer means (only vs base vs rand):')
    for c in sorted(per_cancer):
        r = per_cancer[c]
        if r['mean_only'] is None:
            continue
        print('  %-6s only=%.3f(n%d) base=%.3f(n%d) rand=%.3f(n%d)' % (
            c, r['mean_only'], r['n_only'], r['mean_base'], r['n_base'],
            r['mean_rand'], r['n_rand']))


if __name__ == '__main__':
    main()
