"""E4: gate fixed vs joint optimization — does the positive feedback require co-optimization?

Three conditions on the SAME data:
  baseline  : no gating (uniform weight)
  fixed     : gate computed once from initial residuals, then FROZEN while W is optimized
  joint     : gate and W co-optimized (the soft gating of the main text)

If fixed-gate false edges ~ joint false edges, the selective weighting alone (not the
feedback) drives inflation. If fixed-gate stays near baseline while joint explodes,
the co-optimization feedback is the causal driver.
"""
import os, sys, json, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np
import torch
from sklearn.cluster import KMeans
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT = r'../../data/e4_checkpoint.json'

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
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X.astype(np.float32)

def W_to_edges(W, thresh=0.3):
    d = W.shape[0]
    return {(i, j) for j in range(d) for i in range(j) if abs(W[j, i]) > thresh}

def true_edges(W):
    d = W.shape[0]
    return {(i, j) for j in range(d) for i in range(j) if W[j, i] != 0}

def score(pred, true):
    pred = set(pred); true = set(true)
    tp = len(pred & true); fp = len(pred - true); fn = len(true - pred)
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return F1, P, tp, fp, len(pred)

def _dag_step(W, Xg, gates, d, rho, alpha, opt):
    opt.zero_grad()
    M = torch.eye(d, device=device) - W
    sq = (Xg @ M.T).pow(2)
    if gates is None:
        loss_d = sq.mean()
    else:
        loss_d = (gates * sq.sum(dim=1)).sum() / gates.sum().clamp(min=1)
    h = torch.trace(torch.linalg.matrix_exp(W * W)) - d
    loss = loss_d + 0.5 * rho * h * h + alpha * h + 0.01 * torch.sum(torch.abs(W))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(W, 10.0)
    opt.step()

def _hval(W, d):
    with torch.no_grad():
        return (torch.trace(torch.linalg.matrix_exp(W * W)) - d).item()

def run_baseline(X, outer, inner):
    d = X.shape[1]
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    W = torch.zeros(d, d, device=device, requires_grad=True)
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=0.002)
    for o in range(outer):
        for _ in range(inner):
            _dag_step(W, Xg, None, d, rho, alpha, opt)
        hv = _hval(W, d)
        if abs(hv) < 1e-8: break
        if abs(hv) > 1e-6: alpha = alpha + rho * hv
        rho = min(5 * rho, 1e12)
        if torch.isnan(W).any() or torch.isinf(W).any(): break
    return W.detach().cpu().numpy()

def compute_frozen_gates(X, K, seed):
    """Compute gates from K-means residual dispersion of an initial (short) baseline fit, then return FROZEN gates."""
    n, d = X.shape
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    Winit = torch.tensor(run_baseline(X, 5, inner=100), dtype=torch.float32, device=device)
    M = torch.eye(d, device=device) - Winit
    sq = (Xg @ M.T).pow(2)
    res = sq.mean(dim=1)
    labels = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X).labels_
    cstd = []
    for c in range(K):
        mask = torch.tensor(labels == c, device=device)
        cr = res[mask]
        cstd.append(cr.std().item() if len(cr) > 1 else 1.0)
    cstd = torch.tensor(cstd, device=device)
    smed = torch.median(cstd)
    raw = torch.sigmoid(0.5 * (smed / cstd.clamp(min=1e-8) - 1))
    gates = torch.tensor([raw[labels[i]].item() for i in range(n)], device=device)
    return gates

def run_fixed_gate(X, K, outer, inner, seed):
    d = X.shape[1]
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    gates = compute_frozen_gates(X, K, seed)  # frozen, not updated
    W = torch.zeros(d, d, device=device, requires_grad=True)
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=0.002)
    for o in range(outer):
        for _ in range(inner):
            _dag_step(W, Xg, gates, d, rho, alpha, opt)
        hv = _hval(W, d)
        if abs(hv) < 1e-8: break
        if abs(hv) > 1e-6: alpha = alpha + rho * hv
        rho = min(5 * rho, 1e12)
        if torch.isnan(W).any() or torch.isinf(W).any(): break
    return W.detach().cpu().numpy()

def run_joint(X, K, outer, inner, seed=0, beta_entropy=0.05):
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
        hv = _hval(W, d)
        if abs(hv) < 1e-8: break
        if abs(hv) > 1e-6: alpha = alpha + rho * hv
        rho = min(5 * rho, 1e12)
        if torch.isnan(W).any() or torch.isinf(W).any(): break
    return W.detach().cpu().numpy()

def load_cp():
    return json.load(open(CHECKPOINT, encoding='utf-8')) if os.path.exists(CHECKPOINT) else {}

def save_cp(cp):
    tmp = CHECKPOINT + '.tmp'
    json.dump(cp, open(tmp, 'w', encoding='utf-8'))
    os.replace(tmp, CHECKPOINT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=50)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--outer', type=int, default=20)
    ap.add_argument('--inner', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    cp = load_cp()
    units = []
    for s in range(args.seeds):
        for m in ['baseline', 'fixed', 'joint']:
            units.append((s, m))
    if args.limit > 0:
        units = units[:args.limit]

    todo = [u for u in units if f'{u[0]}/{u[1]}' not in cp]
    print(f'total: {len(units)}, todo: {len(todo)}, device={device}')

    W_true = make_dag(args.d, 1.0, seed=42)
    te = true_edges(W_true)

    for (s, method) in todo:
        key = f'{s}/{method}'
        t0 = time.time()
        X = sample_sem(W_true, args.n, 1.0, seed=1000 + s)
        if method == 'baseline':
            W = run_baseline(X, args.outer, args.inner)
        elif method == 'fixed':
            W = run_fixed_gate(X, args.K, args.outer, args.inner, seed=s)
        else:
            W = run_joint(X, args.K, args.outer, args.inner, seed=s)
        F1, P, tp, fp, ne = score(W_to_edges(W), te)
        cp[key] = dict(seed=s, method=method, f1=round(F1,4), precision=round(P,4), tp=tp, fp=fp, n_edges=ne)
        save_cp(cp)
        print(f'{key}: F1={F1:.3f} P={P:.3f} TP={tp} FP={fp} edges={ne} | {time.time()-t0:.0f}s', flush=True)

    # summary
    print('\n=== E4 summary (mean over seeds) ===')
    for m in ['baseline', 'fixed', 'joint']:
        vals = [cp[k] for k in cp if k.endswith('/' + m)]
        if vals:
            print(f'{m}: F1={np.mean([v["f1"] for v in vals]):.3f} | FP={np.mean([v["fp"] for v in vals]):.1f} | edges={np.mean([v["n_edges"] for v in vals]):.1f}')

if __name__ == '__main__':
    main()
