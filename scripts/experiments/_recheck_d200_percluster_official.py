# -*- coding: utf-8 -*-
"""d=200 official NOTEARS per-cluster + oracle, seeds 0-2 (resume-safe).

Extends the d=200 baseline single-point check (results/d200_baseline_official.json)
to the full Table S12(b) row.  Each cell = baseline + per-cluster(K-means) + oracle
(true labels), 7 official L-BFGS-B fits total, matching the d=100 protocol.

Writes into results/d_scan_official.json as d200_s{seed} cells (same schema as the
d100 cells), so Table S12(b) can report official numbers for both rows.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.cluster import KMeans
import _core
import official_notears_linear as onl

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results')
CP = os.path.join(RESULTS, 'd_scan_official.json')


def load_cp():
    return json.load(open(CP, encoding='utf-8')) if os.path.exists(CP) else {}


def save_cp(cp):
    tmp = CP + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cp, f)
    os.replace(tmp, CP)


def fit_official(X):
    return onl.notears_linear(X, lambda1=0.1, loss_type='l2',
                              max_iter=100, h_tol=1e-8, w_threshold=0.3)


def main():
    cp = load_cp()
    cp.setdefault('cells', {})
    for seed in range(3):
        key = 'd200_s%d' % seed
        if key in cp['cells']:
            print(key, 'already done')
            continue
        d, n = 200, 800
        W_list = _core.make_heterogeneous_dags(d, 3, 1.0, 0.4, seed)
        X, _ = _core.sample_heterogeneous(W_list, n, 1.0, seed)
        true = _core.true_edge_set(W_list)
        labels = np.concatenate([np.full(n // 3 + (1 if k < n % 3 else 0), k)
                                 for k in range(3)])
        t0 = time.time()

        Wb = fit_official(X)
        eb = _core.W_to_edges(Wb.T, 0.3)
        f1b, _, _, _, _, _, cb = _core.score_undirected(eb, true)

        km = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(X)
        e_km = set()
        for c in range(3):
            mask = km.labels_ == c
            if mask.sum() < d + 1:
                continue
            Wc = fit_official(X[mask])
            e_km |= _core.W_to_edges(Wc.T, 0.3)
        f1p, _, _, _, _, _, cp_ = _core.score_undirected(e_km, true)

        e_or = set()
        for c in range(3):
            mask = labels == c
            if mask.sum() < d + 1:
                continue
            Wc = fit_official(X[mask])
            e_or |= _core.W_to_edges(Wc.T, 0.3)
        f1o, _, _, _, _, _, co = _core.score_undirected(e_or, true)

        cp['cells'][key] = {'d': d, 'seed': seed, 'n': n,
                            'baseline': {'f1': f1b, 'edges': cb},
                            'percluster': {'f1': f1p, 'edges': cp_},
                            'oracle': {'f1': f1o, 'edges': co},
                            'time_s': round(time.time() - t0, 1)}
        save_cp(cp)
        print('done %s (%.1fs) base=%.3f per=%.3f or=%.3f' %
              (key, time.time() - t0, f1b, f1p, f1o))
    # summary for d=200
    rows = [cp['cells']['d200_s%d' % s] for s in range(3)
            if 'd200_s%d' % s in cp['cells']]
    if len(rows) == 3:
        cp.setdefault('summary', {})
        cp['summary']['200'] = {m: round(float(np.mean([r[m]['f1'] for r in rows])), 3)
                                for m in ('baseline', 'percluster', 'oracle')}
        save_cp(cp)
        print('SUMMARY200:', json.dumps(cp['summary']['200']))


if __name__ == '__main__':
    main()
