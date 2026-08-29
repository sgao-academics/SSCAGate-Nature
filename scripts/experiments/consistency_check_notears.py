"""P1-5: Consistency check of our self-implemented NOTEARS vs the official
Zheng et al. (2018) implementation (xunzheng/notears).

Question it answers (reviewer): "Is your NOTEARS implementation consistent with
the official one? Different optimizers (Adam vs L-BFGS-B) may give different
local optima, but the recovered structures should be close on identical data."

Protocol:
  - Data: _core.make_dag + sample_linear_sem (identical to the paper's pipeline)
  - Self: _core.fit(X, K=1, method='notears', gated='none')  (Adam, paper defaults)
  - Official: official_notears_linear.notears_linear (L-BFGS-B, default lambda1=0.1)
  - Compare: edge-set Jaccard, F1 vs true, edge counts
  - Grid: d in {20, 50, 100}, n/d in {1, 2}, 3 seeds each

Run: python consistency_check_notears.py
Output: results/consistency_check_notears.json
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _core
import official_notears_linear as onl

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results')
os.makedirs(RESULTS, exist_ok=True)


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_one(d, n, seed):
    W_true = _core.make_dag(d, s0=1.0, seed=seed)
    X = _core.sample_linear_sem(W_true, n, sigma=1.0, seed=seed)
    true = _core.true_edges(W_true)

    # --- self implementation (Adam, paper defaults: l1=0.01, outer=20, inner=100)
    t0 = time.time()
    W_self = _core.fit(X, 1, 'notears', 'none', outer=20, inner=100, seed=seed, l1=0.01)
    t_self = time.time() - t0
    e_self = _core.W_to_edges(W_self, 0.3)
    f1_s, p_s, r_s, tp_s, fp_s, fn_s, cnt_s = _core.score_undirected(e_self, true)

    # --- official implementation (L-BFGS-B, default lambda1=0.1, w_threshold=0.3)
    # NOTE: the official code parameterizes W as X @ W (column convention:
    # W[i,j] = weight of i->j), whereas _core uses row convention (W[j,i] =
    # weight of i->j).  W_off is therefore transposed before edge extraction.
    t0 = time.time()
    W_off = onl.notears_linear(X, lambda1=0.1, loss_type='l2',
                               max_iter=100, h_tol=1e-8, w_threshold=0.3)
    t_off = time.time() - t0
    e_off = _core.W_to_edges(W_off.T, 0.3)
    f1_o, p_o, r_o, tp_o, fp_o, fn_o, cnt_o = _core.score_undirected(e_off, true)

    return {
        'd': d, 'n': n, 'n_over_d': n / d, 'seed': seed,
        'true_edges': len(true),
        'self': {'edges': cnt_s, 'f1': f1_s, 'precision': p_s, 'recall': r_s,
                 'tp': tp_s, 'fp': fp_s, 'time_s': round(t_self, 2)},
        'official': {'edges': cnt_o, 'f1': f1_o, 'precision': p_o, 'recall': r_o,
                     'tp': tp_o, 'fp': fp_o, 'time_s': round(t_off, 2)},
        'agreement': {'jaccard': jaccard(e_self, e_off),
                      'f1_diff': f1_s - f1_o,
                      'edge_diff': cnt_s - cnt_o},
    }


def main():
    grid = []
    for d in (20, 50, 100):
        for nd in (1, 2):
            for seed in (0, 1, 2):
                grid.append((d, int(nd * d), seed))

    cp = _core.load_cp(os.path.join(RESULTS, 'consistency_check_notears.json'))
    cp.setdefault('cells', {})
    for d, n, seed in grid:
        key = f'd{d}_n{n}_s{seed}'
        if key in cp['cells']:
            continue
        cp['cells'][key] = run_one(d, n, seed)
        _core.save_cp(cp, os.path.join(RESULTS, 'consistency_check_notears.json'))
        print('done', key)

    # summary
    rows = list(cp['cells'].values())
    jac = [r['agreement']['jaccard'] for r in rows]
    f1s = [r['self']['f1'] for r in rows]
    f1o = [r['official']['f1'] for r in rows]
    fd = [r['agreement']['f1_diff'] for r in rows]
    ed = [r['agreement']['edge_diff'] for r in rows]
    cp['summary'] = {
        'n_cells': len(rows),
        'jaccard_mean': float(np.mean(jac)), 'jaccard_min': float(np.min(jac)),
        'self_f1_mean': float(np.mean(f1s)), 'official_f1_mean': float(np.mean(f1o)),
        'f1_diff_mean': float(np.mean(fd)), 'f1_diff_max_abs': float(np.max(np.abs(fd))),
        'edge_diff_mean': float(np.mean(ed)),
    }
    _core.save_cp(cp, os.path.join(RESULTS, 'consistency_check_notears.json'))
    print('SUMMARY:', json.dumps(cp['summary'], indent=2))


if __name__ == '__main__':
    main()
