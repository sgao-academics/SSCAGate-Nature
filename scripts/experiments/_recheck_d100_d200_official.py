"""Recheck Table S12(b) dimensionality scan with OFFICIAL NOTEARS (L-BFGS-B).

Config: heterogeneous K=3, s0_shared=1.0, s0_private=0.4, n/d=4, 3 seeds.
Methods: baseline (global), per-cluster (K-means + union), oracle (true labels + union).
d = 100 (baseline/per-cluster already run, oracle new) and d = 200 (all new).

Checkpointed (results/d_scan_official.json).  Run in background.
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


def run_cell(d, seed):
    n = 4 * d
    W_list = _core.make_heterogeneous_dags(d, 3, 1.0, 0.4, seed)
    X, _ = _core.sample_heterogeneous(W_list, n, 1.0, seed)
    true = _core.true_edge_set(W_list)
    labels = np.concatenate([np.full(n // 3 + (1 if k < n % 3 else 0), k) for k in range(3)])

    # baseline
    Wb = fit_official(X)
    eb = _core.W_to_edges(Wb.T, 0.3)
    f1b, _, _, _, _, _, cb = _core.score_undirected(eb, true)

    # per-cluster (K-means)
    km = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(X)
    e_km = set()
    for c in range(3):
        mask = km.labels_ == c
        if mask.sum() < d + 1:
            continue
        Wc = fit_official(X[mask])
        e_km |= _core.W_to_edges(Wc.T, 0.3)
    f1p, _, _, _, _, _, cp_ = _core.score_undirected(e_km, true)

    # oracle (true labels)
    e_or = set()
    for c in range(3):
        mask = labels == c
        if mask.sum() < d + 1:
            continue
        Wc = fit_official(X[mask])
        e_or |= _core.W_to_edges(Wc.T, 0.3)
    f1o, _, _, _, _, _, co = _core.score_undirected(e_or, true)

    return {'d': d, 'seed': seed, 'n': n,
            'baseline': {'f1': f1b, 'edges': cb},
            'percluster': {'f1': f1p, 'edges': cp_},
            'oracle': {'f1': f1o, 'edges': co}}


def main():
    cp = load_cp()
    cp.setdefault('cells', {})
    todo = []
    for d in (100, 200):
        for seed in range(3):
            todo.append((d, seed))
    for d, seed in todo:
        key = 'd%d_s%d' % (d, seed)
        if key in cp['cells']:
            continue
        t0 = time.time()
        cp['cells'][key] = run_cell(d, seed)
        cp['cells'][key]['time_s'] = round(time.time() - t0, 1)
        save_cp(cp)
        print('done %s (%.1fs)' % (key, time.time() - t0))
    # summary by d
    summ = {}
    for d in (100, 200):
        rows = [cp['cells']['d%d_s%d' % (d, s)] for s in range(3)]
        summ[str(d)] = {m: round(float(np.mean([r[m]['f1'] for r in rows])), 3)
                        for m in ('baseline', 'percluster', 'oracle')}
    cp['summary'] = summ
    save_cp(cp)
    print('SUMMARY:', json.dumps(summ))


if __name__ == '__main__':
    main()
