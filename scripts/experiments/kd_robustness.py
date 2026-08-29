"""

E-G: Robustness of the per-cluster F1 gain across K and d.



Verify that the per-cluster advantage over baseline is not specific to K=3,

d=50. We scan (d, K) pairs subject to the per-cluster feasibility constraint

that each cluster has at least d+1 samples (n/K > d).



Configs (s0_private = 0.4, the "sweet spot"):

  (d=50, K=2, n=300), (d=50, K=5, n=300), (d=100, K=2, n=400)

Methods: baseline / per-cluster (K-means) / oracle (true labels). seeds=3.

"""

import os, sys, json, time, warnings, argparse

warnings.filterwarnings('ignore')

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import numpy as np

import torch

from sklearn.cluster import KMeans

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ['OMP_NUM_THREADS'] = '2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('--s0-private', type=float, default=0.4)

    ap.add_argument('--s0-shared', type=float, default=1.0)

    ap.add_argument('--seeds', type=int, default=3)

    ap.add_argument('--outer', type=int, default=15)

    ap.add_argument('--inner', type=int, default=100)

    args = ap.parse_args()



    configs = [(50, 2, 300), (50, 5, 300), (100, 2, 400)]

    results = {}



    for d, K, n in configs:

        base_f1, pc_f1, or_f1 = [], [], []

        for s in range(args.seeds):

            W_list = make_heterogeneous_dags(d, K, args.s0_shared, args.s0_private, seed=42 + s)

            X, labels_true = sample_heterogeneous(W_list, n, 1.0, seed=1000 + s)

            true_edges = true_edge_set(W_list)

            F1, *_ = F1_of(W_to_edges(run_notears_np(X, args.outer, args.inner, s)), true_edges)

            base_f1.append(F1)

            labels_km = KMeans(n_clusters=K, random_state=s, n_init=10).fit(X).labels_

            F1, *_ = F1_of(cluster_union_edges(X, labels_km, args.outer, args.inner, s), true_edges)

            pc_f1.append(F1)

            F1, *_ = F1_of(cluster_union_edges(X, labels_true, args.outer, args.inner, s), true_edges)

            or_f1.append(F1)

        key = f'd{d}_K{K}'

        results[key] = {'d': d, 'K': K, 'n': n,

                        'baseline_f1': float(np.mean(base_f1)),

                        'percluster_f1': float(np.mean(pc_f1)),

                        'oracle_f1': float(np.mean(or_f1)),

                        'gain': float(np.mean(pc_f1) - np.mean(base_f1))}

        print(f'd={d} K={K} n={n}: base F1={np.mean(base_f1):.3f} | percluster F1={np.mean(pc_f1):.3f} '

              f'| oracle F1={np.mean(or_f1):.3f} | gain={np.mean(pc_f1)-np.mean(base_f1):+.3f}')



    path = r'../../data/kd_robustness.json'

    with open(path, 'w') as f:

        json.dump({'s0_private': args.s0_private, 'results': results}, f, indent=2)

    print(f'\nSaved {path}')



if __name__ == '__main__':

    t0 = time.time(); main(); print(f'Total {time.time()-t0:.0f}s')

