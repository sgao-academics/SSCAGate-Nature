"""

E-A: TCGA per-cluster validation.



Does cluster decomposition (per-cluster NOTEARS) recover CLUSTER-SPECIFIC

implanted edges on REAL TCGA expression data, where the heterogeneity is a set

of edges present in only one subgroup?



Design:

  1. Load TCGA (BRCA), top-d variable genes, subsample n.

  2. K-means (K=3) on the ORIGINAL expression -> "ground-truth" subgroups.

  3. In each subgroup k, implant m_per_cluster edges present ONLY in subgroup k

     (X[idx_k, j] = beta * X[idx_k, i] + eps). Ground truth = union of all

     cluster-specific edges.

  4. Methods (all share the same NOTEARS core):

       baseline   : single global NOTEARS on the full mixture

       per-cluster: K-means re-clustering of the IMPLANTED data, then NOTEARS

                    within each cluster, union

       oracle     : NOTEARS within each TRUE subgroup, union (upper bound)



Metric: F1 against the union of implanted cluster-specific edges.



Usage:

  python tcga_percluster_test.py --cancer BRCA --n 300 --m-per-cluster 8 --seeds 3

"""

import os, sys, json, time, warnings, argparse

warnings.filterwarnings('ignore')

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import numpy as np

import pandas as pd

import torch

from sklearn.cluster import KMeans

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ['OMP_NUM_THREADS'] = '2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



DATA_DIR = r'../../data'



# ---------- NOTEARS core (same as cagate_grid.py) ----------

def run_notears_np(X, outer, inner, seed=0):

    torch.manual_seed(seed); np.random.seed(seed)

    d = X.shape[1]

    Xg = torch.tensor(X, dtype=torch.float32, device=device)

    W = torch.zeros(d, d, device=device, requires_grad=True)

    rho, alpha = 1.0, 0.0

    opt = torch.optim.Adam([W], lr=0.002)

    for o in range(outer):

        for _ in range(inner):

            opt.zero_grad()

            M = torch.eye(d, device=device) - W

            sq = (Xg @ M.T).pow(2)

            loss_d = sq.mean()

            h = torch.trace(torch.linalg.matrix_exp(W * W)) - d

            loss = loss_d + 0.5 * rho * h * h + alpha * h + 0.01 * torch.sum(torch.abs(W))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(W, 10.0)

            opt.step()

        with torch.no_grad():

            hv = (torch.trace(torch.linalg.matrix_exp(W * W)) - d).item()

        if abs(hv) < 1e-8: break

        if abs(hv) > 1e-6: alpha = alpha + rho * hv

        rho = min(5 * rho, 1e12)

        if torch.isnan(W).any() or torch.isinf(W).any(): break

    return W.detach().cpu().numpy()



def W_to_edges(W, thresh=0.3):

    d = W.shape[0]

    return {(i, j) for j in range(d) for i in range(j) if abs(W[j, i]) > thresh}



def score_edges(pred, true):

    pred = set(pred); true = set(true)

    tp = len(pred & true); fp = len(pred - true); fn = len(true - pred)

    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    return F1, P, R, tp, fp, len(pred)



def cluster_union_edges(X, labels, outer, inner, seed):

    d = X.shape[1]

    edges = set()

    for c in np.unique(labels):

        mask = labels == c

        if mask.sum() < d + 1:

            continue

        Wc = run_notears_np(X[mask], outer, inner, seed + int(c) * 1000)

        edges |= W_to_edges(Wc)

    return edges



# ---------- data ----------

def load_tcga(cancer, d=100):

    path = os.path.join(DATA_DIR, f'TCGA_{cancer}_HiSeqV2.tsv')

    df = pd.read_csv(path, sep='\t', index_col=0)  # genes x samples

    X = df.values.T.astype(np.float32)  # samples x genes

    var = np.var(X, axis=0)

    top_idx = np.argsort(var)[-d:]

    X = X[:, top_idx]

    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    return X



def implant_cluster_specific_edges(X, labels, m_per_cluster, seed,

                                   beta_lo=1.0, beta_hi=1.5, sigma_eps=0.3):

    """Implant m_per_cluster edges present ONLY in each subgroup k."""

    rng = np.random.RandomState(seed)

    n, d = X.shape

    X = X.copy()

    edges = []

    for c in np.unique(labels):

        idx_c = np.where(labels == c)[0]

        used = set()

        placed = 0

        tries = 0

        while placed < m_per_cluster and tries < m_per_cluster * 50:

            tries += 1

            i = rng.randint(0, d); j = rng.randint(0, d)

            if i == j or (i, j) in used or (j, i) in used:

                continue

            used.add((i, j))

            beta = rng.uniform(beta_lo, beta_hi) * rng.choice([-1.0, 1.0])

            X[idx_c, j] = beta * X[idx_c, i] + sigma_eps * rng.randn(len(idx_c))

            edges.append((i, j))

            placed += 1

    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    return X, edges



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('--cancer', type=str, default='BRCA')

    ap.add_argument('--d', type=int, default=100)

    ap.add_argument('--n', type=int, default=300)

    ap.add_argument('--K', type=int, default=3)

    ap.add_argument('--m-per-cluster', type=int, default=8)

    ap.add_argument('--seeds', type=int, default=3)

    ap.add_argument('--outer', type=int, default=20)

    ap.add_argument('--inner', type=int, default=100)

    args = ap.parse_args()



    X_full = load_tcga(args.cancer, d=args.d)

    n_full = X_full.shape[0]

    print(f'{args.cancer}: n_full={n_full}, d={args.d}, K={args.K}, '

          f'm_per_cluster={args.m_per_cluster}, test n={args.n}, seeds={args.seeds}')



    rows = {'base': [], 'pc': [], 'oracle': []}

    for s in range(args.seeds):

        rng = np.random.RandomState(1000 + s)

        idx = rng.choice(n_full, min(args.n, n_full), replace=False)

        X_sub = X_full[idx]



        # ground-truth subgroups from ORIGINAL expression

        labels_true = KMeans(n_clusters=args.K, random_state=s, n_init=10).fit(X_sub).labels_



        # implant cluster-specific edges

        X_imp, true_edges = implant_cluster_specific_edges(

            X_sub, labels_true, args.m_per_cluster, seed=2000 + s)



        # baseline: global NOTEARS

        F1, P, R, tp, fp, npe = score_edges(

            W_to_edges(run_notears_np(X_imp, args.outer, args.inner, s)), true_edges)

        rows['base'].append((F1, P, R, tp, fp, npe))



        # per-cluster: K-means on IMPLANTED data

        labels_km = KMeans(n_clusters=args.K, random_state=s, n_init=10).fit(X_imp).labels_

        F1, P, R, tp, fp, npe = score_edges(

            cluster_union_edges(X_imp, labels_km, args.outer, args.inner, s), true_edges)

        rows['pc'].append((F1, P, R, tp, fp, npe))



        # oracle: true subgroup labels

        F1, P, R, tp, fp, npe = score_edges(

            cluster_union_edges(X_imp, labels_true, args.outer, args.inner, s), true_edges)

        rows['oracle'].append((F1, P, R, tp, fp, npe))



    print(f'\n{"method":<10}{"F1":>7}{"Prec":>7}{"Rec":>7}{"TP":>5}{"FP":>5}{"pred":>6}')

    out = {}

    for name in ['base', 'pc', 'oracle']:

        F1 = np.mean([x[0] for x in rows[name]]); P = np.mean([x[1] for x in rows[name]])

        R = np.mean([x[2] for x in rows[name]]); tp = np.mean([x[3] for x in rows[name]])

        fp = np.mean([x[4] for x in rows[name]]); npe = np.mean([x[5] for x in rows[name]])

        out[name] = {'f1': float(F1), 'precision': float(P), 'recall': float(R),

                     'tp': float(tp), 'fp': float(fp), 'pred_edges': float(npe)}

        print(f'{name:<10}{F1:>7.3f}{P:>7.3f}{R:>7.3f}{tp:>5.1f}{fp:>5.1f}{npe:>6.1f}')



    out_dir = r'../../data'

    path = os.path.join(out_dir, f'tcga_percluster_{args.cancer}_n{args.n}.json')

    with open(path, 'w') as f:

        json.dump({'cancer': args.cancer, 'n': args.n, 'd': args.d, 'K': args.K,

                   'm_per_cluster': args.m_per_cluster, 'results': out}, f, indent=2)

    print(f'\nSaved {path}')



if __name__ == '__main__':

    t0 = time.time(); main(); print(f'Total {time.time()-t0:.0f}s')

