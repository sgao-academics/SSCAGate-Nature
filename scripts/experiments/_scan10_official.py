# -*- coding: utf-8 -*-
"""10-seed official NOTEARS dimensionality scan (Table S12b), resume-safe + fit cache.

Extends d_scan_official.json to N_SEEDS seeds per d (d=100, 200) using the OFFICIAL
L-BFGS-B implementation.  Every cell = baseline + per-cluster(K-means) + oracle
(true labels).  Two layers of durability:

  1. fit-level cache: results/d_scan_official_cache/d{d}_s{seed}_{role}_{part}.json
     stores the recovered edge list of a single fit, so a crash mid-cell never
     wastes completed fits;
  2. cell-level checkpoint: results/d_scan_official.json (same schema as before),
     so completed cells are never recomputed (d100_s0..s2 are reused directly).

Parallelism: the 7 fits of a cell are independent -> multiprocessing.Pool(4).
Deterministic: make_heterogeneous_dags(seed) yields the same DAG for a given seed,
so cells computed by the previous 3-seed script are bit-identical and reused.
"""
import sys, os, json, time
# ThreadPool instead of multiprocessing.Pool: scipy's L-BFGS-B (Fortran) releases
# the GIL while running, so threads give real parallelism -- and unlike a
# multiprocessing.Pool on Windows (which spawns a new console window per worker
# and flashes windows on every cell), a thread pool is silent.
from multiprocessing.pool import ThreadPool
import numpy as np
from sklearn.cluster import KMeans
import _core
import official_notears_linear as onl

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results')
CP = os.path.join(RESULTS, 'd_scan_official.json')
CACHE = os.path.join(RESULTS, 'd_scan_official_cache')
os.makedirs(CACHE, exist_ok=True)

N_SEEDS = 10
DIMS = (100, 200)
NWORKERS = 4


def fit_official(X):
    return onl.notears_linear(X, lambda1=0.1, loss_type='l2',
                              max_iter=100, h_tol=1e-8, w_threshold=0.3)


def cache_path(d, seed, role, part):
    return os.path.join(CACHE, 'd%d_s%d_%s_%d.json' % (d, seed, role, part))


def load_cp():
    return json.load(open(CP, encoding='utf-8')) if os.path.exists(CP) else {}


def save_cp(cp):
    tmp = CP + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cp, f)
    os.replace(tmp, CP)


def fit_and_cache(args):
    d, seed, role, part, X = args
    p = cache_path(d, seed, role, part)
    if os.path.exists(p):
        return p
    W = fit_official(X)
    e = _core.W_to_edges(W.T, 0.3)
    json.dump(sorted(e), open(p, 'w', encoding='utf-8'))
    return p


def read_edges(p):
    # cache files store edges as [[i, j], ...] (JSON cannot serialize tuples);
    # convert back to a set of tuples for hashing
    raw = json.load(open(p, encoding='utf-8'))
    return set(tuple(x) for x in raw)


def run_cell(d, seed):
    n = 4 * d
    W_list = _core.make_heterogeneous_dags(d, 3, 1.0, 0.4, seed)
    X, _ = _core.sample_heterogeneous(W_list, n, 1.0, seed)
    true = _core.true_edge_set(W_list)
    labels = np.concatenate([np.full(n // 3 + (1 if k < n % 3 else 0), k)
                             for k in range(3)])

    km = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(X)

    jobs = [(d, seed, 'base', 0, X)]
    for c in range(3):
        m = km.labels_ == c
        if m.sum() >= d + 1:
            jobs.append((d, seed, 'pc', c, X[m]))
    for c in range(3):
        m = labels == c
        if m.sum() >= d + 1:
            jobs.append((d, seed, 'or', c, X[m]))

    todo = [j for j in jobs if not os.path.exists(cache_path(*j[:4]))]
    if todo:
        with ThreadPool(NWORKERS) as pool:
            pool.map(fit_and_cache, todo)

    eb = read_edges(cache_path(d, seed, 'base', 0))
    f1b, _, _, _, _, _, cb = _core.score_undirected(eb, true)

    e_km = set()
    for c in range(3):
        m = km.labels_ == c
        if m.sum() >= d + 1:
            e_km |= read_edges(cache_path(d, seed, 'pc', c))
    f1p, _, _, _, _, _, cp_ = _core.score_undirected(e_km, true)

    e_or = set()
    for c in range(3):
        m = labels == c
        if m.sum() >= d + 1:
            e_or |= read_edges(cache_path(d, seed, 'or', c))
    f1o, _, _, _, _, _, co = _core.score_undirected(e_or, true)

    return {'d': d, 'seed': seed, 'n': n,
            'baseline': {'f1': f1b, 'edges': cb},
            'percluster': {'f1': f1p, 'edges': cp_},
            'oracle': {'f1': f1o, 'edges': co}}


def main():
    cp = load_cp()
    cp.setdefault('cells', {})
    for d in DIMS:
        for seed in range(N_SEEDS):
            key = 'd%d_s%d' % (d, seed)
            if key in cp['cells']:
                print(key, 'already done', flush=True)
                continue
            t0 = time.time()
            cp['cells'][key] = run_cell(d, seed)
            cp['cells'][key]['time_s'] = round(time.time() - t0, 1)
            save_cp(cp)
            print('done %s (%.1fs)' % (key, time.time() - t0), flush=True)
    cp.setdefault('summary', {})
    for d in DIMS:
        rows = [cp['cells']['d%d_s%d' % (d, s)] for s in range(N_SEEDS)]
        cp['summary'][str(d)] = {m: round(float(np.mean([r[m]['f1'] for r in rows])), 3)
                                 for m in ('baseline', 'percluster', 'oracle')}
        cp['summary'][str(d) + '_sd'] = {m: round(float(np.std([r[m]['f1'] for r in rows])), 3)
                                         for m in ('baseline', 'percluster', 'oracle')}
    save_cp(cp)
    print('SUMMARY:', json.dumps(cp['summary']), flush=True)


if __name__ == '__main__':
    main()
