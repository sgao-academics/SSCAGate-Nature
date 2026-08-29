"""

E-B: Does better clustering close the gap to the oracle?



The per-cluster F1 is bottlenecked by partitioning accuracy. We compare three

partitioners (K-means, spectral clustering, Gaussian mixture) for the

per-cluster NOTEARS strategy, against the oracle (true labels), on synthetic

heterogeneous data where the K-means-oracle gap is large.



Fixed config: d=50, K=3, s0_shared=1.0, s0_private=0.4, n=300, seeds=3.

"""

import os, sys, json, time, warnings, argparse

warnings.filterwarnings('ignore')

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import numpy as np

import torch

from sklearn.cluster import KMeans, SpectralClustering

from sklearn.mixture import GaussianMixture

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ['OMP_NUM_THREADS'] = '2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ---------- data generation (same as cagate_grid.py) ----------

def make_dag(d, s0, seed):

    rng = np.random.RandomState(seed)

    W = np.zeros((d, d))

    for j in range(d):

        for i in range(j):

            if rng.rand() < s0 / max(d - 1, 1):

                W[j, i] = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])

    return W



def make_heterogeneous_dags(d, K_true, s0_shared, s0_private, seed):

    rng = np.random.RandomState(seed)

    W_shared = make_dag(d, s0_shared, seed)

    W_list = []

    for k in range(K_true):

        W_k = W_shared.copy()

        for j in range(d):

            for i in range(j):

                if W_shared[j, i] == 0 and rng.rand() < s0_private / max(d - 1, 1):

                    W_k[j, i] = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])

        W_list.append(W_k)

    return W_list



def sample_sem(W, n, sigma, seed):

    d = W.shape[0]

    rng = np.random.RandomState(seed)

    X = np.zeros((n, d))

    eps = rng.randn(n, d) * sigma

    for j in range(d):

        X[:, j] = X @ W[j, :] + eps[:, j]

    return X



def sample_heterogeneous(W_list, n_total, sigma, seed):

    K = len(W_list)

    Xs, labels = [], []

    for k, W_k in enumerate(W_list):

        nk = n_total // K + (1 if k < n_total % K else 0)

        Xs.append(sample_sem(W_k, nk, sigma, seed + k * 1000))

        labels.extend([k] * nk)

    X = np.vstack(Xs).astype(np.float32)

    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    return X, np.array(labels)



def true_edge_set(W_list):

    d = W_list[0].shape[0]

    edges = set()

    for W in W_list:

        for j in range(d):

            for i in range(j):

                if W[j, i] != 0:

                    edges.add((i, j))

    return edges



# ---------- NOTEARS core ----------

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



def F1_of(pred, true):

    tp = len(pred & true); fp = len(pred - true); fn = len(true - pred)

    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    return F1, P, R



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



def partition(X, method, K, seed):

    if method == 'kmeans':

        return KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X).labels_

    if method == 'spectral':

        return SpectralClustering(n_clusters=K, random_state=seed,

                                  affinity='rbf', assign_labels='kmeans').fit(X).labels_

    if method == 'gmm':

        return GaussianMixture(n_components=K, random_state=seed,

                               covariance_type='full', reg_covar=1e-3).fit(X).predict(X)

    raise ValueError(method)



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('--d', type=int, default=50)

    ap.add_argument('--K', type=int, default=3)

    ap.add_argument('--s0-shared', type=float, default=1.0)

    ap.add_argument('--s0-private', type=float, default=0.4)

    ap.add_argument('--n', type=int, default=300)

    ap.add_argument('--seeds', type=int, default=3)

    ap.add_argument('--outer', type=int, default=15)

    ap.add_argument('--inner', type=int, default=100)

    args = ap.parse_args()



    methods = ['kmeans', 'spectral', 'gmm']

    rows = {'base': [], 'oracle': []}

    for m in methods:

        rows[m] = []



    for s in range(args.seeds):

        W_list = make_heterogeneous_dags(args.d, args.K, args.s0_shared, args.s0_private, seed=42 + s)

        X, labels_true = sample_heterogeneous(W_list, args.n, 1.0, seed=1000 + s)

        true_edges = true_edge_set(W_list)



        F1, P, R = F1_of(W_to_edges(run_notears_np(X, args.outer, args.inner, s)), true_edges)

        rows['base'].append((F1, P, R))



        for m in methods:

            labels = partition(X, m, args.K, s)

            F1, P, R = F1_of(cluster_union_edges(X, labels, args.outer, args.inner, s), true_edges)

            rows[m].append((F1, P, R))



        F1, P, R = F1_of(cluster_union_edges(X, labels_true, args.outer, args.inner, s), true_edges)

        rows['oracle'].append((F1, P, R))



    print(f'\nd={args.d} K={args.K} s0_priv={args.s0_private} n={args.n} seeds={args.seeds}')

    print(f'{"method":<12}{"F1":>7}{"Prec":>7}{"Rec":>7}')

    out = {}

    for name in ['base'] + methods + ['oracle']:

        F1 = np.mean([x[0] for x in rows[name]])

        P = np.mean([x[1] for x in rows[name]])

        R = np.mean([x[2] for x in rows[name]])

        out[name] = {'f1': float(F1), 'precision': float(P), 'recall': float(R)}

        print(f'{name:<12}{F1:>7.3f}{P:>7.3f}{R:>7.3f}')



    out_dir = r'../../data'

    path = os.path.join(out_dir, 'cluster_method_comparison.json')

    with open(path, 'w') as f:

        json.dump({'d': args.d, 'K': args.K, 's0_private': args.s0_private,

                   'n': args.n, 'results': out}, f, indent=2)

    print(f'\nSaved {path}')



if __name__ == '__main__':

    t0 = time.time(); main(); print(f'Total {time.time()-t0:.0f}s')

