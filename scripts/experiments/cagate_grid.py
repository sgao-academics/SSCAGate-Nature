"""

CAGate heterogeneity grid: how does the per-cluster NOTEARS F1 gain over

baseline vary with (heterogeneity strength s0_private) x (sample size n)?



Grid: s0_private in {0.2, 0.4, 0.8}  x  n in {200, 300, 600}

Methods: baseline / per-cluster(K-means) / oracle(true labels)

All share the same NOTEARS core. d=50, K_true=3, seeds=3.

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

    return F1, P, R, tp, fp, len(pred)



def _cluster_union(X, labels, outer, inner, seed):

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

    ap.add_argument('--d', type=int, default=50)

    ap.add_argument('--K', type=int, default=3)

    ap.add_argument('--s0-shared', type=float, default=1.0)

    ap.add_argument('--seeds', type=int, default=3)

    ap.add_argument('--outer', type=int, default=15)

    ap.add_argument('--inner', type=int, default=100)

    args = ap.parse_args()



    s0_priv_list = [0.2, 0.4, 0.8]

    n_list = [200, 300, 600]

    results = {}

    print(f'd={args.d}, K={args.K}, seeds={args.seeds}, grid: s0_priv x n')



    for s0p in s0_priv_list:

        for n in n_list:

            base_f1, pc_f1, or_f1 = [], [], []

            base_p, pc_p, or_p = [], [], []

            base_r, pc_r, or_r = [], [], []

            for s in range(args.seeds):

                W_list = make_heterogeneous_dags(args.d, args.K, args.s0_shared, s0p, seed=42 + s)

                X, labels_true = sample_heterogeneous(W_list, n, 1.0, seed=1000 + s)

                true_edges = true_edge_set(W_list)

                # baseline

                F1, P, R, *_ = F1_of(W_to_edges(run_notears_np(X, args.outer, args.inner, s)), true_edges)

                base_f1.append(F1); base_p.append(P); base_r.append(R)

                # per-cluster

                labels_km = KMeans(n_clusters=args.K, random_state=s, n_init=10).fit(X).labels_

                F1, P, R, *_ = F1_of(_cluster_union(X, labels_km, args.outer, args.inner, s), true_edges)

                pc_f1.append(F1); pc_p.append(P); pc_r.append(R)

                # oracle

                F1, P, R, *_ = F1_of(_cluster_union(X, labels_true, args.outer, args.inner, s), true_edges)

                or_f1.append(F1); or_p.append(P); or_r.append(R)



            key = f's0p{s0p}_n{n}'

            results[key] = {

                's0_private': s0p, 'n': n,

                'baseline_f1': float(np.mean(base_f1)), 'baseline_p': float(np.mean(base_p)), 'baseline_r': float(np.mean(base_r)),

                'percluster_f1': float(np.mean(pc_f1)), 'percluster_p': float(np.mean(pc_p)), 'percluster_r': float(np.mean(pc_r)),

                'oracle_f1': float(np.mean(or_f1)), 'oracle_p': float(np.mean(or_p)), 'oracle_r': float(np.mean(or_r)),

                'gain': float(np.mean(pc_f1) - np.mean(base_f1)),

            }

            print(f's0_priv={s0p} n={n:>4}: base F1={np.mean(base_f1):.3f} | percluster F1={np.mean(pc_f1):.3f} | oracle F1={np.mean(or_f1):.3f} | gain={np.mean(pc_f1)-np.mean(base_f1):+.3f}', flush=True)



    path = r'../../data/cagate_grid.json'

    with open(path, 'w') as f:

        json.dump(results, f, indent=2)

    print(f'\nSaved {path}')

    # summary

    print('\n=== gain (percluster - baseline F1) by s0_private x n ===')

    for s0p in s0_priv_list:

        row = []

        for n in n_list:

            row.append(f'{results[f"s0p{s0p}_n{n}"]["gain"]:+.3f}')

        print(f's0_private={s0p}: ' + '  '.join(row))



if __name__ == '__main__':

    t0 = time.time(); main(); print(f'Total {time.time()-t0:.0f}s')

