# -*- coding: utf-8 -*-
"""TRRUST external validation of cluster-only edges (P0-b, B1 anchor).

The DepMap co-essentiality anchor was negative (co-essentiality captures
protein complexes, not TF->target regulation). Here we use TRRUST, a curated
database of human transcriptional regulatory relationships (TF -> target), as
the external anchor.

Question: are the edges recovered ONLY by cluster decomposition (per-cluster
minus baseline) enriched for known TF->target regulatory relationships, compared
with (a) baseline edges and (b) random gene pairs?

The recovered NOTEARS edges are DIRECTED (i->j). TRRUST is DIRECTED (TF->target).
So we check: what fraction of recovered directed pairs (source, target) match a
known (TF, target) pair in TRRUST. Enrichment is tested with Fisher's exact test.

Deterministic (fixed seed for random pairs). Read-only + fast.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(ROOT, 'results')
CP = os.path.join(RESULTS, 'pan_open.json')
TRRUST = r'../../data/validation/trrust_human.tsv'
OUT = os.path.join(RESULTS, 'pan_open_trrust_validation.json')


def load_trrust():
    df = pd.read_csv(TRRUST, sep='\t', header=None)
    # cols: TF, target, mode, PMID
    pairs = set(zip(df[0].astype(str), df[1].astype(str)))
    return pairs


def main():
    cp = json.load(open(CP, encoding='utf-8'))
    cells = cp.get('cells', {})
    trrust = load_trrust()
    print('TRRUST regulatory pairs:', len(trrust))

    rng = np.random.RandomState(42)

    pooled_only_hits = 0
    pooled_only_total = 0
    pooled_base_hits = 0
    pooled_base_total = 0
    pooled_rand_hits = 0
    pooled_rand_total = 0

    per_cancer = {}
    for key, cell in sorted(cells.items()):
        if cell.get('skipped'):
            continue
        cancer = cell['cancer']
        genes_c = cell['genes']

        def directed_sym(edge):
            i, j = edge          # NOTEARS: i -> j (i is source/parent)
            return genes_c[i], genes_c[j]

        only_edges = [directed_sym(e) for e in cell['cluster_only_edges']]
        base_edges = [directed_sym(e) for e in cell['baseline_edges']]
        # random directed pairs: same count, drawn from genes_c (i != j)
        rand_pairs = []
        while len(rand_pairs) < len(only_edges):
            a, b = rng.choice(len(genes_c), 2, replace=False)
            if genes_c[a] != genes_c[b]:
                rand_pairs.append((genes_c[a], genes_c[b]))

        oh = sum(1 for (a, b) in only_edges if (a, b) in trrust)
        bh = sum(1 for (a, b) in base_edges if (a, b) in trrust)
        rh = sum(1 for (a, b) in rand_pairs if (a, b) in trrust)

        pooled_only_hits += oh; pooled_only_total += len(only_edges)
        pooled_base_hits += bh; pooled_base_total += len(base_edges)
        pooled_rand_hits += rh; pooled_rand_total += len(rand_pairs)

        per_cancer[cancer] = {
            'n_only': len(only_edges), 'hit_only': oh,
            'n_base': len(base_edges), 'hit_base': bh,
            'n_rand': len(rand_pairs), 'hit_rand': rh,
            'frac_only': round(oh / len(only_edges), 4) if only_edges else None,
            'frac_base': round(bh / len(base_edges), 4) if base_edges else None,
            'frac_rand': round(rh / len(rand_pairs), 4) if rand_pairs else None,
        }

    out = {'per_cancer': per_cancer}
    frac_only = pooled_only_hits / pooled_only_total if pooled_only_total else 0
    frac_base = pooled_base_hits / pooled_base_total if pooled_base_total else 0
    frac_rand = pooled_rand_hits / pooled_rand_total if pooled_rand_total else 0
    out['pooled'] = {
        'frac_only': round(frac_only, 5),
        'frac_base': round(frac_base, 5),
        'frac_rand': round(frac_rand, 5),
        'n_only': pooled_only_total, 'hit_only': pooled_only_hits,
        'n_base': pooled_base_total, 'hit_base': pooled_base_hits,
        'n_rand': pooled_rand_total, 'hit_rand': pooled_rand_hits,
    }

    print('\nPOOLED (TRRUST TF->target hit fraction):')
    print('  cluster-only: %.4f (%d/%d)' % (frac_only, pooled_only_hits, pooled_only_total))
    print('  baseline:     %.4f (%d/%d)' % (frac_base, pooled_base_hits, pooled_base_total))
    print('  random:       %.4f (%d/%d)' % (frac_rand, pooled_rand_hits, pooled_rand_total))

    # Fisher exact: only vs rand, only vs base
    try:
        from scipy.stats import fisher_exact
        # only vs rand
        a = pooled_only_hits; b = pooled_only_total - pooled_only_hits
        c = pooled_rand_hits; d = pooled_rand_total - pooled_rand_hits
        _, p_or = fisher_exact([[a, b], [c, d]], alternative='greater')
        # only vs base
        a2 = pooled_only_hits; b2 = pooled_only_total - pooled_only_hits
        c2 = pooled_base_hits; d2 = pooled_base_total - pooled_base_hits
        _, p_ob = fisher_exact([[a2, b2], [c2, d2]], alternative='greater')
        out['pooled']['p_only_vs_rand'] = float(p_or)
        out['pooled']['p_only_vs_base'] = float(p_ob)
        print('Fisher only>rand: p=%.4g   only>base: p=%.4g' % (p_or, p_ob))
    except Exception as e:
        print('stats error:', e)

    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
    print('\nSAVED', OUT)
    print('\nper-cancer (only/base/rand hit fraction):')
    for c in sorted(per_cancer):
        r = per_cancer[c]
        if r['frac_only'] is None:
            continue
        print('  %-6s only=%.3f(%d/%d) base=%.3f(%d/%d) rand=%.3f(%d/%d)' % (
            c, r['frac_only'], r['hit_only'], r['n_only'],
            r['frac_base'], r['hit_base'], r['n_base'],
            r['frac_rand'], r['hit_rand'], r['n_rand']))


if __name__ == '__main__':
    main()
