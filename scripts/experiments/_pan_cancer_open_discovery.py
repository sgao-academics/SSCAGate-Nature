# -*- coding: utf-8 -*-
"""P0-b: open (no-implantation) pan-cancer cluster decomposition + DepMap
external validation.

Goal (reviewer-facing): show that the edges recovered ONLY by cluster
decomposition (per-cluster union minus global baseline) on real TCGA expression
are not noise -- they carry a stronger co-dependency signal in an INDEPENDENT
perturbation assay (DepMap CRISPR gene effect) than random gene pairs do.

Design (all 33 TCGA cancers, open discovery, no implanted edges):
  - load TCGA expression, take top d=100 variable genes, standardize
  - K-means (K=3) partition, then:
      baseline     : official NOTEARS (L-BFGS-B) on the full mixture
      per-cluster  : official NOTEARS within each cluster, union edges
  - "cluster-only edges" = per-cluster edges NOT in baseline
  - DepMap anchor: for each recovered directed pair (i->j), the CRISPR gene
    effect of gene i and gene j across cell lines are correlated; a genuine
    regulatory edge should show higher |co-essentiality| than a random pair.
  - statistic: compare the mean |co-essentiality| of cluster-only edges vs.
    (a) baseline edges, (b) random gene pairs (permutation), per cancer and pooled.

Durability (same skeleton as _scan10_official.py):
  - fit-level cache  : results/pan_open_cache/{cancer}_{role}_{part}.json  (edge list)
  - cell-level check : results/pan_open.json  ({cancer} -> edges, skipped if done)
  - deterministic: top-d gene selection is deterministic; NOTEARS is deterministic;
    K-means seeded. Seeds control K-means init only (fit is deterministic).
Parallelism: ThreadPool(NWORKERS) -- scipy L-BFGS-B releases GIL, silent on Windows.
"""
import sys, os, json, time
from multiprocessing.pool import ThreadPool
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import _core
import official_notears_linear as onl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # SSCAGate-Nature/
RESULTS = os.path.join(ROOT, 'results')
CP = os.path.join(RESULTS, 'pan_open.json')
CACHE = os.path.join(RESULTS, 'pan_open_cache')
os.makedirs(CACHE, exist_ok=True)

DATA_DIR = r'../../data'
DEPMAP = os.path.join(DATA_DIR, 'depmap', 'CRISPRGeneEffect.csv')

CANCERS = ['ACC','BLCA','BRCA','CESC','CHOL','COAD','DLBC','ESCA','GBM','HNSC',
           'KICH','KIRC','KIRP','LAML','LGG','LIHC','LUAD','LUSC','MESO','OV',
           'PAAD','PCPG','PRAD','READ','SARC','SKCM','STAD','TGCT','THCA','THYM',
           'UCEC','UCS','UVM']
D = 50
K = 3
N_SEEDS = 3
NWORKERS = 4
LAMBDA1 = 0.1
MAX_ITER = 100
W_THRESH = 0.3
MIN_SAMPLES = 3 * (D + 1)   # need n >= 3(d+1) for K=3 per-cluster identifiability


def load_tcga(cancer, d=D):
    p = os.path.join(DATA_DIR, 'TCGA_%s_HiSeqV2.tsv' % cancer)
    df = pd.read_csv(p, sep='\t', index_col=0)
    # rows = genes (index), cols = samples; transpose to samples x genes
    X = df.values.T.astype(np.float32)
    var = np.var(X, axis=0)
    top = np.argsort(var)[-d:]
    X = X[:, top]
    genes = np.array(df.index[top])
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, genes


def fit_official(X):
    return onl.notears_linear(X, lambda1=LAMBDA1, loss_type='l2',
                              max_iter=MAX_ITER, h_tol=1e-8, w_threshold=W_THRESH)


def cache_path(cancer, seed, role, part):
    return os.path.join(CACHE, '%s_s%d_%s_%d.json' % (cancer, seed, role, part))


def load_cp():
    return json.load(open(CP, encoding='utf-8')) if os.path.exists(CP) else {}


def save_cp(cp):
    tmp = CP + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cp, f)
    os.replace(tmp, CP)


def fit_and_cache(args):
    cancer, seed, role, part, X = args
    p = cache_path(cancer, seed, role, part)
    if os.path.exists(p):
        return p
    W = fit_official(X)
    e = _core.W_to_edges(W.T, W_THRESH)
    json.dump(sorted(e), open(p, 'w', encoding='utf-8'))
    return p


def read_edges(p):
    raw = json.load(open(p, encoding='utf-8'))
    return set(tuple(x) for x in raw)


def run_cancer(cancer, seed):
    X, genes = load_tcga(cancer)
    if X.shape[0] < MIN_SAMPLES:
        return None
    # subsample deterministically to a fixed n so K-means/noteears are stable
    rng = np.random.RandomState(1000 + seed)
    n = min(X.shape[0], 300)
    idx = rng.choice(X.shape[0], n, replace=False)
    Xs = X[idx]

    km = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(Xs)

    jobs = [(cancer, seed, 'base', 0, Xs)]
    for c in range(K):
        m = km.labels_ == c
        if m.sum() >= D + 1:
            jobs.append((cancer, seed, 'pc', c, Xs[m]))
    todo = [j for j in jobs if not os.path.exists(cache_path(*j[:4]))]
    if todo:
        with ThreadPool(NWORKERS) as pool:
            pool.map(fit_and_cache, todo)

    e_base = read_edges(cache_path(cancer, seed, 'base', 0))
    e_pc = set()
    for c in range(K):
        m = km.labels_ == c
        if m.sum() >= D + 1:
            e_pc |= read_edges(cache_path(cancer, seed, 'pc', c))
    e_cluster_only = e_pc - e_base
    return {
        'cancer': cancer, 'seed': seed, 'n': n,
        'n_genes': len(genes),
        'baseline_edges': sorted(e_base),
        'percluster_edges': sorted(e_pc),
        'cluster_only_edges': sorted(e_cluster_only),
        'genes': [str(g) for g in genes],
    }


def main():
    cp = load_cp()
    cp.setdefault('cells', {})
    total = len(CANCERS) * N_SEEDS
    done = 0
    t_start = time.time()
    for cancer in CANCERS:
        for seed in range(N_SEEDS):
            key = '%s_s%d' % (cancer, seed)
            if key in cp['cells']:
                done += 1
                continue
            t0 = time.time()
            r = run_cancer(cancer, seed)
            if r is None:
                cp['cells'][key] = {'cancer': cancer, 'seed': seed, 'skipped': True}
            else:
                r['time_s'] = round(time.time() - t0, 1)
                cp['cells'][key] = r
            save_cp(cp)
            done += 1
            if r is None:
                print('done %d/%d %s s%d SKIP (n<min)' % (done, total, cancer, seed), flush=True)
            else:
                print('done %d/%d %s s%d  base=%d per=%d only=%d (%.0fs)' % (
                    done, total, cancer, seed, len(r['baseline_edges']),
                    len(r['percluster_edges']), len(r['cluster_only_edges']),
                    time.time() - t0), flush=True)
    print('\nALL CELLS DONE. total %.0fs' % (time.time() - t_start), flush=True)


if __name__ == '__main__':
    main()
