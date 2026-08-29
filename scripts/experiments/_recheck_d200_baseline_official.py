# -*- coding: utf-8 -*-
"""d=200 official NOTEARS baseline single-point check.

Answers the question: does the OFFICIAL L-BFGS-B implementation of NOTEARS
go blind at d=200 (the "death line" observed with our Adam implementation)?

Runs ONE official baseline fit on heterogeneous data (K=3, s0_shared=1.0,
s0_private=0.4, n/d=4, seed=0), matching the Table S12(b) d=100 protocol,
so the d=200 row can be compared directly with the official d=100 rows.

Resume-safe: if results/d200_baseline_official.json exists, exits immediately.
Run in background; expected wall time ~30-90 min (single L-BFGS-B fit).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _core
import official_notears_linear as onl

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results')
OUT = os.path.join(RESULTS, 'd200_baseline_official.json')


def main():
    if os.path.exists(OUT):
        print('already done:', open(OUT, encoding='utf-8').read()[:300])
        return
    d, seed = 200, 0
    n = 4 * d
    W_list = _core.make_heterogeneous_dags(d, 3, 1.0, 0.4, seed)
    X, _ = _core.sample_heterogeneous(W_list, n, 1.0, seed)
    true = _core.true_edge_set(W_list)

    t0 = time.time()
    W = onl.notears_linear(X, lambda1=0.1, loss_type='l2',
                           max_iter=100, h_tol=1e-8, w_threshold=0.3)
    e = _core.W_to_edges(W.T, 0.3)
    f1, prec, rec, _, _, _, ce = _core.score_undirected(e, true)

    out = {'d': d, 'seed': seed, 'n': n,
           'baseline': {'f1': f1, 'edges': ce, 'precision': prec, 'recall': rec},
           'true_edges': len(true), 'time_s': round(time.time() - t0, 1)}
    tmp = OUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
