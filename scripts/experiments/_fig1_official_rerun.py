# -*- coding: utf-8 -*-
"""Official L-BFGS-B re-run of the Fig.1 main experiment (Table S1/S7).

Figure 1's constructive conclusion (cluster decomposition improves F1 over a
global model) was produced with the AUTHOR's Adam implementation
(overnight_master.py run_notears_np), which the Supplementary Note 3/Table S12b
themselves flag as having convergence artifacts at high d.  To answer the
reviewer's sharpest objection ("the F1 gain may be an artifact of the author's
own solver"), we re-run Fig.1's grid with the OFFICIAL L-BFGS-B NOTEARS.

Grid: d=50, K=3, s0_shared=1.0
  s0_private in {0.2, 0.4, 0.8}   (heterogeneity strength)
  n         in {200, 300, 600}     (sample size)
  strategies: baseline / per-cluster(K-means) / oracle(true labels)
  seeds: 10

Durability (same pattern as _scan10_official.py):
  1. fit-level cache: results/fig1_official_cache/c{s0p}_n{n}_s{seed}_{role}_{part}.json
     stores a single fit's edge list, so a mid-cell crash never wastes fits;
  2. cell-level checkpoint: results/fig1_official.json
     (config, seed) -> {baseline, percluster, oracle}; completed cells skipped.
Parallelism: ThreadPool (scipy L-BFGS-B releases GIL; silent on Windows).
Deterministic: make_heterogeneous_dags / sample_heterogeneous are seeded.
"""
import sys, os, json, time
from multiprocessing.pool import ThreadPool
import numpy as np
from sklearn.cluster import KMeans
import _core
import official_notears_linear as onl

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'results')
CP = os.path.join(RESULTS, 'fig1_official.json')
CACHE = os.path.join(RESULTS, 'fig1_official_cache')
os.makedirs(CACHE, exist_ok=True)

D = 50
S0P = (0.2, 0.4, 0.8)
NS = (200, 300, 600)
N_SEEDS = 10
NWORKERS = 4
LAMBDA1 = 0.1      # same as official d-scan; matches sparsity used in main text
MAX_ITER = 100
W_THRESH = 0.3


def fit_official(X):
    return onl.notears_linear(X, lambda1=LAMBDA1, loss_type='l2',
                              max_iter=MAX_ITER, h_tol=1e-8, w_threshold=W_THRESH)


def cache_path(s0p, n, seed, role, part):
    return os.path.join(CACHE, 'c%s_n%d_s%d_%s_%d.json' % (s0p, n, seed, role, part))


def load_cp():
    return json.load(open(CP, encoding='utf-8')) if os.path.exists(CP) else {}


def save_cp(cp):
    tmp = CP + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cp, f)
    os.replace(tmp, CP)


def fit_and_cache(args):
    s0p, n, seed, role, part, X = args
    p = cache_path(s0p, n, seed, role, part)
    if os.path.exists(p):
        return p
    W = fit_official(X)
    e = _core.W_to_edges(W.T, W_THRESH)
    json.dump(sorted(e), open(p, 'w', encoding='utf-8'))
    return p


def read_edges(p):
    raw = json.load(open(p, encoding='utf-8'))
    return set(tuple(x) for x in raw)


def run_cell(s0p, n, seed):
    W_list = _core.make_heterogeneous_dags(D, 3, 1.0, s0p, seed)
    X, labels_true = _core.sample_heterogeneous(W_list, n, 1.0, seed)
    true = _core.true_edge_set(W_list)

    # true subgroup index per sample (to honor K=3 even split for oracle)
    labels = np.concatenate([np.full(n // 3 + (1 if k < n % 3 else 0), k)
                             for k in range(3)])
    km = KMeans(n_clusters=3, random_state=seed, n_init=10).fit(X)

    jobs = [(s0p, n, seed, 'base', 0, X)]
    for c in range(3):
        m = km.labels_ == c
        if m.sum() >= D + 1:
            jobs.append((s0p, n, seed, 'pc', c, X[m]))
    for c in range(3):
        m = labels == c
        if m.sum() >= D + 1:
            jobs.append((s0p, n, seed, 'or', c, X[m]))

    todo = [j for j in jobs if not os.path.exists(cache_path(*j[:5]))]
    if todo:
        with ThreadPool(NWORKERS) as pool:
            pool.map(fit_and_cache, todo)

    eb = read_edges(cache_path(s0p, n, seed, 'base', 0))
    f1b, pb, rb, tb, fpb, fnb, cb = _core.score_undirected(eb, true)

    e_km = set()
    for c in range(3):
        m = km.labels_ == c
        if m.sum() >= D + 1:
            e_km |= read_edges(cache_path(s0p, n, seed, 'pc', c))
    f1p, pp_, rp_, tp_, fpp_, fnp_, cp_ = _core.score_undirected(e_km, true)

    e_or = set()
    for c in range(3):
        m = labels == c
        if m.sum() >= D + 1:
            e_or |= read_edges(cache_path(s0p, n, seed, 'or', c))
    f1o, po, ro, to, fpo, fno, co = _core.score_undirected(e_or, true)

    return {'config': 's0p%s_n%d' % (s0p, n), 'd': D, 'seed': seed, 'n': n,
            's0p': s0p,
            'baseline': {'f1': f1b, 'precision': pb, 'recall': rb,
                         'tp': tb, 'fp': fpb, 'fn': fnb, 'edges': cb},
            'percluster': {'f1': f1p, 'precision': pp_, 'recall': rp_,
                           'tp': tp_, 'fp': fpp_, 'fn': fnp_, 'edges': cp_},
            'oracle': {'f1': f1o, 'precision': po, 'recall': ro,
                       'tp': to, 'fp': fpo, 'fn': fno, 'edges': co}}


def main():
    cp = load_cp()
    cp.setdefault('cells', {})
    total = len(S0P) * len(NS) * N_SEEDS
    done = 0
    for s0p in S0P:
        for n in NS:
            for seed in range(N_SEEDS):
                key = 'c%s_n%d_s%d' % (s0p, n, seed)
                if key in cp['cells']:
                    done += 1
                    continue
                t0 = time.time()
                cp['cells'][key] = run_cell(s0p, n, seed)
                cp['cells'][key]['time_s'] = round(time.time() - t0, 1)
                save_cp(cp)
                done += 1
                cb = cp['cells'][key]['baseline']
                cp_ = cp['cells'][key]['percluster']
                print('done %s/%d %s  base=%.3f per=%.3f gain=%+.3f (%.0fs)' % (
                    done, total, key, cb['f1'], cp_['f1'],
                    cp_['f1'] - cb['f1'], time.time() - t0), flush=True)

    # summarize per-config (mean + SD over seeds) and per-pooled across seeds
    cp.setdefault('summary', {})
    for s0p in S0P:
        for n in NS:
            cfg = 's0p%s_n%d' % (s0p, n)
            rows = [cp['cells']['c%s_n%d_s%d' % (s0p, n, s)] for s in range(N_SEEDS)]
            cp['summary'][cfg] = {
                m: {'f1': round(float(np.mean([r[m]['f1'] for r in rows])), 3),
                    'sd': round(float(np.std([r[m]['f1'] for r in rows])), 3),
                    'edges': round(float(np.mean([r[m]['edges'] for r in rows])), 1)}
                for m in ('baseline', 'percluster', 'oracle')}
            gains = [r['percluster']['f1'] - r['baseline']['f1'] for r in rows]
            cp['summary'][cfg]['gain'] = round(float(np.mean(gains)), 3)
            cp['summary'][cfg]['gain_min'] = round(float(min(gains)), 3)
            cp['summary'][cfg]['gain_max'] = round(float(max(gains)), 3)
            cp['summary'][cfg]['gain_sd'] = round(float(np.std(gains)), 3)
            cp['summary'][cfg]['n_pos'] = int(sum(1 for g in gains if g > 0))
    save_cp(cp)
    print('\n=== SUMMARY (official L-BFGS-B, d=50, 10 seeds) ===', flush=True)
    for s0p in S0P:
        for n in NS:
            s = cp['summary']['s0p%s_n%d' % (s0p, n)]
            print('s0p=%.1f n=%d: base=%.3f per=%.3f or=%.3f | gain=%+.3f (%+0.3f..%+0.3f) pos=%d/10'
                  % (s0p, n, s['baseline']['f1'], s['percluster']['f1'],
                     s['oracle']['f1'], s['gain'], s['gain_min'], s['gain_max'],
                     s['n_pos']), flush=True)
    print('SAVED', CP, flush=True)


if __name__ == '__main__':
    main()
