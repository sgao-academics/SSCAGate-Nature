"""E10: hard-gating trajectory — does the edge-count inflation track gate concentration too?

Hard gating = K-means hard assignment (fixed labels) + residual-dispersion gate.
Track, per outer iteration, edge_count / gate_std / n_eff, and compare to baseline.
Complements the soft-gating trajectory (experiment F). Uses TCGA BRCA semi-synthetic.
"""
import os, sys, json, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = r'../../data'
CHECKPOINT = r'../../data/e10_checkpoint.json'

def load_tcga(cancer, d=100):
    path = os.path.join(DATA_DIR, f'TCGA_{cancer}_HiSeqV2.tsv')
    df = pd.read_csv(path, sep='\t', index_col=0)
    X = df.values.T.astype(np.float32)
    var = np.var(X, axis=0)
    top_idx = np.argsort(var)[-d:]
    X = X[:, top_idx]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X

def implant_edges(X, m, seed, beta_lo=1.0, beta_hi=1.5, sigma_eps=0.3):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    X = X.copy()
    edges = []
    used = set()
    while len(edges) < m:
        i = rng.randint(0, d); j = rng.randint(0, d)
        if i == j or (i, j) in used or (j, i) in used: continue
        used.add((i, j))
        beta = rng.uniform(beta_lo, beta_hi) * rng.choice([-1.0, 1.0])
        X[:, j] = beta * X[:, i] + sigma_eps * rng.randn(n)
        edges.append((i, j))
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, edges

def edge_count(W, thresh=0.3):
    d = W.shape[0]
    return int((np.abs(W) > thresh).sum() / 2)

def kish_neff(g):
    g = np.asarray(g, dtype=np.float64)
    s = g.sum()
    return (s * s) / (g * g).sum() if s > 0 else len(g)

def run_hard_trajectory(X, K, outer, inner, seed):
    n, d = X.shape
    torch.manual_seed(seed)
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    labels = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X).labels_  # hard, fixed
    W = torch.zeros(d, d, device=device, requires_grad=True)
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=0.002)
    traj = []
    for o in range(outer):
        for _ in range(inner):
            opt.zero_grad()
            M = torch.eye(d, device=device) - W
            sq = (Xg @ M.T).pow(2)
            res = sq.mean(dim=1)
            cstd = []
            for c in range(K):
                mask = torch.tensor(labels == c, device=device)
                cr = res[mask]
                cstd.append(cr.std().item() if len(cr) > 1 else 1.0)
            cstd = torch.tensor(cstd, device=device)
            smed = torch.median(cstd)
            raw = torch.sigmoid(0.5 * (smed / cstd.clamp(min=1e-8) - 1))
            gates = torch.tensor([raw[labels[i]].item() for i in range(n)], device=device)
            loss_d = (gates * sq.sum(dim=1)).sum() / gates.sum().clamp(min=1)
            h = torch.trace(torch.linalg.matrix_exp(W * W)) - d
            loss = loss_d + 0.5 * rho * h * h + alpha * h + 0.01 * torch.sum(torch.abs(W))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(W, 10.0)
            opt.step()
        Wn = W.detach().cpu().numpy()
        gn = gates.detach().cpu().numpy()
        traj.append({'iter': o + 1, 'edge_count': edge_count(Wn),
                     'gate_std': float(gn.std()), 'n_eff': float(kish_neff(gn))})
        with torch.no_grad():
            hv = (torch.trace(torch.linalg.matrix_exp(W * W)) - d).item()
        if abs(hv) < 1e-8: break
        if abs(hv) > 1e-6: alpha = alpha + rho * hv
        rho = min(5 * rho, 1e12)
        if torch.isnan(W).any() or torch.isinf(W).any(): break
    return traj

def run_baseline_trajectory(X, outer, inner):
    d = X.shape[1]
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    W = torch.zeros(d, d, device=device, requires_grad=True)
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=0.002)
    traj = []
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
        Wn = W.detach().cpu().numpy()
        traj.append({'iter': o + 1, 'edge_count': edge_count(Wn)})
        with torch.no_grad():
            hv = (torch.trace(torch.linalg.matrix_exp(W * W)) - d).item()
        if abs(hv) < 1e-8: break
        if abs(hv) > 1e-6: alpha = alpha + rho * hv
        rho = min(5 * rho, 1e12)
        if torch.isnan(W).any() or torch.isinf(W).any(): break
    return traj

def load_cp():
    return json.load(open(CHECKPOINT, encoding='utf-8')) if os.path.exists(CHECKPOINT) else {}

def save_cp(cp):
    tmp = CHECKPOINT + '.tmp'
    json.dump(cp, open(tmp, 'w', encoding='utf-8'))
    os.replace(tmp, CHECKPOINT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cancer', type=str, default='BRCA')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--outer', type=int, default=20)
    ap.add_argument('--inner', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    cp = load_cp()
    units = [(s, m) for s in range(args.seeds) for m in ['hard', 'baseline']]
    if args.limit > 0:
        units = units[:args.limit]
    todo = [u for u in units if f'{u[0]}/{u[1]}' not in cp]
    print(f'total: {len(units)}, todo: {len(todo)}, device={device}')

    X_full = load_tcga(args.cancer, d=100)
    n_full = X_full.shape[0]

    for (s, method) in todo:
        key = f'{s}/{method}'
        t0 = time.time()
        rng = np.random.RandomState(1000 + s)
        idx = rng.choice(n_full, min(args.n, n_full), replace=False)
        X_imp, _ = implant_edges(X_full[idx], args.m, seed=2000 + s)
        if method == 'hard':
            traj = run_hard_trajectory(X_imp, args.K, args.outer, args.inner, seed=s)
        else:
            traj = run_baseline_trajectory(X_imp, args.outer, args.inner)
        cp[key] = dict(seed=s, method=method, trajectory=traj)
        save_cp(cp)
        print(f'{key}: {len(traj)} iters | {time.time()-t0:.0f}s', flush=True)

    # summary: mean trajectory
    print('\n=== E10 hard-gating trajectory (mean over seeds) ===')
    print(f'{"iter":>5}{"base_edges":>12}{"hard_edges":>12}{"gate_std":>10}{"n_eff":>9}')
    max_iter = args.outer
    be = np.zeros(max_iter); he = np.zeros(max_iter); gs = np.zeros(max_iter); ne = np.zeros(max_iter); cnt = np.zeros(max_iter)
    for s in range(args.seeds):
        h = cp.get(f'{s}/hard'); b = cp.get(f'{s}/baseline')
        if h:
            for rec in h['trajectory']:
                i = rec['iter'] - 1; he[i] += rec['edge_count']; gs[i] += rec['gate_std']; ne[i] += rec['n_eff']; cnt[i] += 1
        if b:
            for rec in b['trajectory']:
                i = rec['iter'] - 1; be[i] += rec['edge_count']
    cnt[cnt == 0] = 1
    he /= cnt; gs /= cnt; ne /= cnt; be /= cnt
    for i in range(max_iter):
        if cnt[i] > 0 or he[i] > 0 or be[i] > 0:
            print(f'{i+1:>5}{be[i]:>12.1f}{he[i]:>12.1f}{gs[i]:>10.4f}{ne[i]:>9.1f}')

if __name__ == '__main__':
    main()
