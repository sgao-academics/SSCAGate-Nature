"""
Experiment C: hyperparameter sensitivity. Is the edge-count pitfall robust to
the edge threshold and the sparsity coefficient (lambda)?

Two scans on a single DAG (d=50, n=50, i.e. n/d=1):
  1. edge threshold  in {0.2, 0.3, 0.4, 0.5}  (lambda fixed 0.01)
  2. sparsity lambda in {0.005, 0.01, 0.02}  (threshold fixed 0.3)

For each, baseline vs soft gating. The claim is robust if soft still inflates
false edges (more edges, lower F1) across the whole grid.

Checkpoint: ../../data/c_sensitivity_checkpoint.json
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core import (device, make_dag, sample_linear_sem, W_to_edges, true_edges,
                   score_undirected, fit, load_cp, save_cp)
import numpy as np

CHECKPOINT = r'../../data/c_sensitivity_checkpoint.json'


def run_unit(param_type, param_val, seed, method, outer, inner):
    d = 50
    W_true = make_dag(d, 1.0, seed=42)
    X = sample_linear_sem(W_true, d, 1.0, seed=1000 + seed)  # n = d
    te = true_edges(W_true)
    if param_type == 'threshold':
        W = fit(X, 3, 'notears', method, outer, inner, seed=seed, l1=0.01)
        edges = W_to_edges(W, thresh=param_val)
    else:  # lambda
        W = fit(X, 3, 'notears', method, outer, inner, seed=seed, l1=param_val)
        edges = W_to_edges(W, thresh=0.3)
    F1, P, R, tp, fp, fn, ne = score_undirected(edges, te)
    return dict(f1=round(F1, 4), precision=round(P, 4), recall=round(R, 4),
                tp=tp, fp=fp, fn=fn, n_edges=ne)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--outer', type=int, default=20)
    ap.add_argument('--inner', type=int, default=100)
    ap.add_argument('--thresh-list', type=str, default='0.2,0.3,0.4,0.5')
    ap.add_argument('--lambda-list', type=str, default='0.005,0.01,0.02')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    thresh_list = [float(x) for x in args.thresh_list.split(',')]
    lam_list = [float(x) for x in args.lambda_list.split(',')]
    methods = ['base', 'soft']

    cp = load_cp(CHECKPOINT)
    units = []
    for t in thresh_list:
        for s in range(args.seeds):
            for m in methods:
                units.append(('threshold', t, s, m))
    for l in lam_list:
        for s in range(args.seeds):
            for m in methods:
                units.append(('lambda', l, s, m))

    todo = [u for u in units if _key(*u) not in cp]
    print(f'sensitivity: {len(units)} units, {len(todo)} todo, device={device}', flush=True)

    done = 0
    for (pt, pv, s, m) in todo:
        if args.limit and done >= args.limit:
            print(f'[limit {args.limit}]', flush=True)
            break
        key = _key(pt, pv, s, m)
        t0 = time.time()
        r = run_unit(pt, pv, s, m, args.outer, args.inner)
        r.update(param_type=pt, param_val=pv, seed=s, method=m)
        cp[key] = r
        save_cp(cp, CHECKPOINT)
        done += 1
        print(f'[{done}/{len(todo)}] {pt:9s}={pv:>5} s={s} {m:4s} '
              f'F1={r["f1"]:.3f} edges={r["n_edges"]:>3} FP={r["fp"]:>3} | {time.time()-t0:.0f}s',
              flush=True)

    _summarize(cp, thresh_list, lam_list, methods, args.seeds)


def _key(pt, pv, s, m):
    return f'{pt}|{pv}|{s}|{m}'


def _summarize(cp, thresh_list, lam_list, methods, seeds):
    print('\n=== SUMMARY (mean over seeds) ===', flush=True)
    print('--- threshold scan (lambda=0.01) ---', flush=True)
    for t in thresh_list:
        line = f'thresh={t}: '
        for m in methods:
            keys = [_key('threshold', t, s, m) for s in range(seeds)]
            if any(k not in cp for k in keys):
                line += f'{m}=INCOMPLETE '
                continue
            f1s = [cp[k]['f1'] for k in keys]
            es = [cp[k]['n_edges'] for k in keys]
            line += f'{m}={np.mean(f1s):.3f}(e{np.mean(es):.0f}) '
        print(line, flush=True)
    print('--- lambda scan (threshold=0.3) ---', flush=True)
    for l in lam_list:
        line = f'lambda={l}: '
        for m in methods:
            keys = [_key('lambda', l, s, m) for s in range(seeds)]
            if any(k not in cp for k in keys):
                line += f'{m}=INCOMPLETE '
                continue
            f1s = [cp[k]['f1'] for k in keys]
            es = [cp[k]['n_edges'] for k in keys]
            line += f'{m}={np.mean(f1s):.3f}(e{np.mean(es):.0f}) '
        print(line, flush=True)


if __name__ == '__main__':
    t0 = time.time(); main(); print(f'Total {time.time()-t0:.0f}s', flush=True)
