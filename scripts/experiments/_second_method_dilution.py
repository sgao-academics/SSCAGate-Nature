"""
Empirical test of the Theorem's method-family claim (v2, stable small-d setting).

Runs BOTH NOTEARS and a second global reconstruction method (DAGMA-lite, the
log-determinant acyclicity method of Bello et al. 2022) on the SAME synthetic
heterogeneous data (mixture of K subpopulation SEMs), and compares the
private-edge recovery of a single GLOBAL fit versus CLUSTER DECOMPOSITION
(partition-then-fit, using the true subpopulation labels so the test isolates the
dilution effect from clustering accuracy).

If the theorem's Remark is correct, the second method should exhibit the SAME
failure: a global fit drops most private edges while partitioning recovers them.
A second method that does NOT fail would falsify the "not NOTEARS-specific"
claim; a second method that does fail supports it.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys, json, math, time, statistics
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'd': 20,
    'K': 3,
    'n_per_kl': 200,     # per subpopulation -> total n = 600
    's0': 1.0,           # shared skeleton density
    's0_priv': 0.4,      # private edge density per subpopulation
    'w_min': 0.5, 'w_max': 1.0,
    'theta': 0.3,
    'seeds': [0, 1, 2],
    'lr': 0.001,
    'lam': 0.01,
    'notears_outer': 15, 'notears_inner': 400,
    'dagma_iter': 3000, 'dagma_clip': 0.8,   # clip W to keep I-W nonsingular
}


def gen_mixture(d, K, n_per_kl, s0, s0_priv, w_min, w_max, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)

    positions = [(i, j) for i in range(d) for j in range(i + 1, d)]
    rng.shuffle(positions)

    W_shared = np.zeros((d, d))
    for (i, j) in positions[:int(s0 * d)]:
        W_shared[i, j] = rng.uniform(w_min, w_max) * rng.choice([-1, 1])

    priv_masks = [np.zeros((d, d), dtype=bool) for _ in range(K)]
    for sub in range(K):
        free = [(i, j) for (i, j) in positions if W_shared[i, j] == 0]
        rng.shuffle(free)
        for (i, j) in free[:int(s0_priv * d)]:
            priv_masks[sub][i, j] = True

    private_edges = [(i, j) for (i, j) in positions
                     if any(priv_masks[s][i, j] for s in range(K))]

    X_list, cid_list = [], []
    for sub in range(K):
        A = W_shared.copy()
        mask_idx = np.argwhere(priv_masks[sub])
        for (i, j) in mask_idx:
            A[i, j] = rng.uniform(w_min, w_max) * rng.choice([-1, 1])
        M = np.eye(d) - A.T
        try:
            inv_M = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            inv_M = np.linalg.pinv(M)
        n = n_per_kl
        Xc = np.random.randn(n, d) @ inv_M.T
        Xc = Xc + (0.3 + 2.0 * sub / max(K - 1, 1)) * np.random.randn(n, d)
        X_list.append(Xc)
        cid_list.append(np.full(n, sub, dtype=int))

    X = np.vstack(X_list)
    cid = np.concatenate(cid_list)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return torch.tensor(X, dtype=torch.float32), W_shared, priv_masks, private_edges, torch.tensor(cid, dtype=torch.long)


def fit_notears(X, lr=0.001, lam=0.01, outer=15, inner=400):
    d = X.shape[1]
    Xg = X.to(device)
    W = torch.zeros(d, d, requires_grad=True, device=device)
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=lr)
    for _ in range(outer):
        for _ in range(inner):
            opt.zero_grad()
            M = torch.eye(d, device=device) - W
            sq = (Xg @ M.T) ** 2
            h = torch.trace(torch.linalg.matrix_exp(W * W)) - d
            loss = sq.mean() + 0.5 * rho * h ** 2 + alpha * h + lam * torch.sum(torch.abs(W))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(W, 10.0)
            opt.step()
        with torch.no_grad():
            hv = (torch.trace(torch.linalg.matrix_exp(W * W)) - d).item()
        if abs(hv) > 1e-6:
            alpha = alpha + rho * hv
        rho = min(5 * rho, 1e12)
        if torch.isnan(W).any() or torch.isinf(W).any() or abs(hv) < 1e-6:
            break
    return W.detach().cpu()


def fit_dagma(X, lr=0.001, lam=0.01, n_iter=3000, clip=0.8):
    d = X.shape[1]
    Xg = X.to(device)
    W = torch.zeros(d, d, requires_grad=True, device=device)
    opt = torch.optim.Adam([W], lr=lr)
    for it in range(n_iter):
        opt.zero_grad()
        # keep I-W safely nonsingular by clamping W magnitude
        with torch.no_grad():
            W.clamp_(-clip, clip)
        M = torch.eye(d, device=device) - W
        loss_mse = (Xg @ M.T).pow(2).mean()
        s = torch.linalg.slogdet(M)
        h = -s.logabsdet
        loss = loss_mse + 0.5 * h + lam * torch.sum(torch.abs(W))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(W, 10.0)
        opt.step()
        if torch.isnan(W).any() or torch.isinf(W).any():
            break
    return W.detach().cpu()


def edges_at(W, theta=0.3):
    return (torch.abs(W) > theta).float()


def private_recall(We, private_edges):
    present = 0
    for (i, j) in private_edges:
        if We[i, j] > 0 or We[j, i] > 0:
            present += 1
    return present / len(private_edges) if private_edges else 0.0


def cluster_decomp(X, cid, fit_fn, theta=0.3):
    d = X.shape[1]
    We = torch.zeros(d, d)
    for sub in torch.unique(cid):
        mask = cid == sub
        Wsub = fit_fn(X[mask])
        We = torch.maximum(We, (torch.abs(Wsub) > theta).float())
    return We


def main():
    theta = CONFIG['theta']
    results = []
    for seed in CONFIG['seeds']:
        X, W_shared, priv_masks, private_edges, cid = gen_mixture(
            CONFIG['d'], CONFIG['K'], CONFIG['n_per_kl'], CONFIG['s0'], CONFIG['s0_priv'],
            CONFIG['w_min'], CONFIG['w_max'], seed)

        t0 = time.time()
        Wn_g = fit_notears(X, CONFIG['lr'], CONFIG['lam'], CONFIG['notears_outer'], CONFIG['notears_inner'])
        Wn_c = cluster_decomp(X, cid, lambda x: fit_notears(x, CONFIG['lr'], CONFIG['lam'], CONFIG['notears_outer'], CONFIG['notears_inner']))
        t_n = time.time() - t0

        t0 = time.time()
        Wd_g = fit_dagma(X, CONFIG['lr'], CONFIG['lam'], CONFIG['dagma_iter'], CONFIG['dagma_clip'])
        Wd_c = cluster_decomp(X, cid, lambda x: fit_dagma(x, CONFIG['lr'], CONFIG['lam'], CONFIG['dagma_iter'], CONFIG['dagma_clip']))
        t_d = time.time() - t0

        row = {
            'seed': seed,
            'private_edges': len(private_edges),
            'notears_global': round(private_recall(edges_at(Wn_g, theta), private_edges), 3),
            'notears_cluster': round(private_recall(Wn_c, private_edges), 3),
            'dagma_global': round(private_recall(edges_at(Wd_g, theta), private_edges), 3),
            'dagma_cluster': round(private_recall(Wd_c, private_edges), 3),
            'notears_s': round(t_n, 1),
            'dagma_s': round(t_d, 1),
        }
        results.append(row)
        print(row, flush=True)

    print('\n=== SUMMARY ===', flush=True)
    n_avg = lambda k: round(statistics.mean(r[k] for r in results), 3)
    print('NOTEARS global private-recall:', n_avg('notears_global'), flush=True)
    print('NOTEARS cluster private-recall:', n_avg('notears_cluster'), flush=True)
    print('DAGMA   global private-recall:', n_avg('dagma_global'), flush=True)
    print('DAGMA   cluster private-recall:', n_avg('dagma_cluster'), flush=True)

    print('=== DELTA (cluster - global) ===', flush=True)
    print('NOTEARS:', round(n_avg('notears_cluster') - n_avg('notears_global'), 3), flush=True)
    print('DAGMA  :', round(n_avg('dagma_cluster') - n_avg('dagma_global'), 3), flush=True)

    out = {'config': CONFIG, 'device': str(device), 'results': results,
           'mean': {k: n_avg(k) for k in ['notears_global', 'notears_cluster', 'dagma_global', 'dagma_cluster']},
           'delta': {'notears': round(n_avg('notears_cluster') - n_avg('notears_global'), 3),
                     'dagma': round(n_avg('dagma_cluster') - n_avg('dagma_global'), 3)}}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_second_method_dilution.json'),
              'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print('written _second_method_dilution.json', flush=True)


if __name__ == '__main__':
    main()
