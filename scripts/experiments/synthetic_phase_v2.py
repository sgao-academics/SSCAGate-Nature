"""
Phase-transition synthetic experiment V2 (parameterized).

Same methods as v1, but:
  - d, n-ratios, seeds, iterations configurable via CLI
  - reports BOTH F1 (correctness) and edge-count (the paper's metric)
    side by side, to expose whether gating recovers true edges or inflates
    false edges.

Usage:
  python _synthetic_phase_v2.py --d 50 --n-ratios 1,2,4 --seeds 2 --outer 20 --inner 100
"""
import os, sys, json, time, warnings, argparse
warnings.filterwarnings('ignore')
sys.path.insert(0, r'[conda site-packages]')
import numpy as np
import torch
from sklearn.cluster import KMeans
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- data gen ----------
def make_dag(d, s0, seed):
    rng = np.random.RandomState(seed)
    W = np.zeros((d, d))
    for j in range(d):
        for i in range(j):
            if rng.rand() < s0 / max(d - 1, 1):
                W[j, i] = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])
    return W

def sample_sem(W, n, sigma, seed):
    d = W.shape[0]
    rng = np.random.RandomState(seed)
    X = np.zeros((n, d))
    eps = rng.randn(n, d) * sigma
    for j in range(d):
        X[:, j] = X @ W[j, :] + eps[:, j]
    return X

def make_heterogeneous(W_base, K_true, n_total, sigma, seed, het=1.0):
    rng = np.random.RandomState(seed)
    d = W_base.shape[0]
    Xs = []
    for k in range(K_true):
        scale_k = rng.uniform(0.5, 1.5) if het > 0 else 1.0
        nk = n_total // K_true + (1 if k < n_total % K_true else 0)
        Xs.append(sample_sem(W_base * scale_k, nk, sigma, seed + k * 1000))
    X = np.vstack(Xs).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X

# ---------- metrics ----------
def score(W_np, W_true, thresh=0.3):
    d = W_true.shape[0]
    pred = set(); true = set()
    for j in range(d):
        for i in range(j):
            if abs(W_np[j, i]) > thresh: pred.add((i, j))
            if W_true[j, i] != 0: true.add((i, j))
    tp = len(pred & true); fp = len(pred - true); fn = len(true - pred)
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return F1, P, R, len(pred), len(true)

# ---------- methods (shared NOTEARS core) ----------
def _notears_loop(Xg, W, outer, inner, lr=0.002, extra_loss=None, opt_W=None):
    d = W.shape[0]
    rho, alpha = 1.0, 0.0
    if opt_W is None:
        opt_W = torch.optim.Adam([W], lr=lr)
    for o in range(outer):
        for _ in range(inner):
            opt_W.zero_grad()
            M = torch.eye(d, device=device) - W
            sq = (Xg @ M.T).pow(2)
            loss_d = sq.mean()
            if extra_loss is not None:
                loss_d = extra_loss(sq, W)
            h = torch.trace(torch.linalg.matrix_exp(W * W)) - d
            loss = loss_d + 0.5 * rho * h * h + alpha * h + 0.01 * torch.sum(torch.abs(W))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(W, 10.0)
            opt_W.step()
        with torch.no_grad():
            hv = (torch.trace(torch.linalg.matrix_exp(W * W)) - d).item()
        if abs(hv) < 1e-8: break
        if abs(hv) > 1e-6: alpha = alpha + rho * hv
        rho = min(5 * rho, 1e12)
        if torch.isnan(W).any() or torch.isinf(W).any(): break
    return W

def run_baseline(X, outer, inner):
    d = X.shape[1]
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    W = torch.zeros(d, d, device=device, requires_grad=True)
    _notears_loop(Xg, W, outer, inner)
    return W.detach().cpu().numpy()

def _gate_hard(res, cid, alpha=0.5):
    gates = torch.ones_like(res)
    for c in torch.unique(cid):
        mask = (cid == c)
        if mask.sum() < 3: continue
        r_c = res[mask]; med = torch.median(r_c); mad = torch.median(torch.abs(r_c - med))
        if mad < 1e-8: continue
        gates[mask] = torch.sigmoid(-alpha * (r_c - med) / mad)
    return gates

def run_hard(X, K, outer, inner, seed=0):
    d = X.shape[1]
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    cid = torch.tensor(KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X).labels_, device=device)
    W = torch.zeros(d, d, device=device, requires_grad=True)
    def extra(sq, W):
        res = sq.mean(dim=1)
        gates = _gate_hard(res, cid)
        return (gates * sq.sum(dim=1)).sum() / gates.sum().clamp(min=1)
    _notears_loop(Xg, W, outer, inner, extra_loss=extra)
    return W.detach().cpu().numpy()

def run_soft(X, K, outer, inner, seed=0, beta_entropy=0.05):
    n, d = X.shape
    torch.manual_seed(seed)
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    W = torch.zeros(d, d, device=device, requires_grad=True)
    P_logits = (torch.randn(n, K, device=device) * 0.1).requires_grad_(True)
    opt_W = torch.optim.Adam([W], lr=0.002)
    opt_P = torch.optim.Adam([P_logits], lr=0.01)
    rho, alpha = 1.0, 0.0
    for o in range(outer):
        for _ in range(inner):
            opt_W.zero_grad(); opt_P.zero_grad()
            M = torch.eye(d, device=device) - W
            sq = (Xg @ M.T).pow(2)
            res = sq.mean(dim=1)
            P_soft = torch.softmax(P_logits, dim=1)
            cw = P_soft.sum(0)
            wm = (P_soft * res.unsqueeze(1)).sum(0) / cw.clamp(min=1e-8)
            wv = (P_soft * (res.unsqueeze(1) - wm.unsqueeze(0)) ** 2).sum(0) / cw.clamp(min=1e-8)
            cstd = torch.sqrt(wv.clamp(min=1e-8))
            smed = torch.median(cstd)
            raw = torch.sigmoid(0.5 * (smed / cstd.clamp(min=1e-8) - 1))
            gates = (P_soft * raw.unsqueeze(0)).sum(1)
            loss_d = (gates * sq.sum(dim=1)).sum() / gates.sum().clamp(min=1)
            h = torch.trace(torch.linalg.matrix_exp(W * W)) - d
            logP = torch.log(P_soft.clamp(min=1e-8))
            ent = -(P_soft * logP).sum(1).mean()
            loss = loss_d + 0.5 * rho * h * h + alpha * h + 0.01 * torch.sum(torch.abs(W)) - beta_entropy * ent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(W, 10.0)
            torch.nn.utils.clip_grad_norm_(P_logits, 5.0)
            opt_W.step(); opt_P.step()
        with torch.no_grad():
            hv = (torch.trace(torch.linalg.matrix_exp(W * W)) - d).item()
        if abs(hv) < 1e-8: break
        if abs(hv) > 1e-6: alpha = alpha + rho * hv
        rho = min(5 * rho, 1e12)
        if torch.isnan(W).any() or torch.isinf(W).any(): break
    return W.detach().cpu().numpy()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=50)
    ap.add_argument('--n-ratios', type=str, default='1,2,4')
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--outer', type=int, default=20)
    ap.add_argument('--inner', type=int, default=100)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--s0', type=float, default=2.0)
    ap.add_argument('--het', type=float, default=1.0, help='1.0=weight heterogeneity, 0=no heterogeneity')
    args = ap.parse_args()

    n_ratios = [float(x) for x in args.n_ratios.split(',')]
    W_base = make_dag(args.d, args.s0, seed=42)
    n_true = int((W_base != 0).sum())
    print(f'd={args.d}, true edges={n_true}, K={args.K}, het={args.het}, ratios={n_ratios}, seeds={args.seeds}, iter={args.outer}x{args.inner}')

    results = {}
    for nd in n_ratios:
        n = int(nd * args.d)
        rows = {'base': [], 'hard': [], 'soft': []}
        for s in range(args.seeds):
            X = make_heterogeneous(W_base, args.K, n, 1.0, seed=1000 + s, het=args.het)
            for name, fn in [('base', lambda: run_baseline(X, args.outer, args.inner)),
                             ('hard', lambda: run_hard(X, args.K, args.outer, args.inner, seed=s)),
                             ('soft', lambda: run_soft(X, args.K, args.outer, args.inner, seed=s))]:
                Wp = fn()
                F1, P, R, npred, _ = score(Wp, W_base)
                rows[name].append((F1, P, R, npred))
        line = f'n/d={nd:>4.1f} (n={n:>4})'
        summary = {}
        for name in ['base', 'hard', 'soft']:
            f1 = np.mean([x[0] for x in rows[name]]); p = np.mean([x[1] for x in rows[name]])
            r = np.mean([x[2] for x in rows[name]]); e = np.mean([x[3] for x in rows[name]])
            summary[name] = {'f1': float(f1), 'precision': float(p), 'recall': float(r), 'edges': float(e)}
            line += f' | {name}: F1={f1:.3f} edges={e:.0f}'
        summary['hard_minus_soft_f1'] = summary['hard']['f1'] - summary['soft']['f1']
        results[str(nd)] = {'n': n, 'nd': nd, **summary}
        print(line, flush=True)

    out = rf'../../data/synth_v2_d{args.d}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'d': args.d, 'true_edges': n_true, 'results': results}, f, indent=2)
    print(f'Saved {out}')

if __name__ == '__main__':
    t0 = time.time(); main(); print(f'Total {time.time()-t0:.0f}s')
