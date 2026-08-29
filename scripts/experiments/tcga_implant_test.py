"""
E8: TCGA semi-synthetic validation.

On REAL TCGA expression data, implant a known causal subgraph (m edges with
strong coefficients), then measure whether cluster-gated methods (hard/soft)
recover the IMPLANTED edges (TP) or instead inflate FALSE edges (FP) relative
to baseline NOTEARS.

Metric: F1 against the implanted edge set (ground truth). This closes the loop:
the synthetic result (gating inflates false edges) is reproduced on real
transcriptomic covariance structure.

Usage:
  python _tcga_implant_test.py --cancer BRCA --n 200 --m 20 --seeds 3
"""
import os, sys, json, time, warnings, argparse
warnings.filterwarnings('ignore')
sys.path.insert(0, r'[conda site-packages]')
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = r'../../data'

# ---------- methods (shared NOTEARS core, same as v2) ----------
def score_edges(pred_edges, true_edges):
    pred = set(pred_edges); true = set(true_edges)
    tp = len(pred & true); fp = len(pred - true); fn = len(true - pred)
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return F1, P, R, tp, fp, len(pred), len(true)

def _notears_loop(Xg, W, outer, inner, extra_loss=None):
    d = W.shape[0]
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=0.002)
    for o in range(outer):
        for _ in range(inner):
            opt.zero_grad()
            M = torch.eye(d, device=device) - W
            sq = (Xg @ M.T).pow(2)
            loss_d = sq.mean() if extra_loss is None else extra_loss(sq, W)
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

def W_to_edges(W, thresh=0.3):
    d = W.shape[0]
    edges = set()
    for j in range(d):
        for i in range(j):
            if abs(W[j, i]) > thresh:
                edges.add((i, j))
    return edges

# ---------- data: TCGA + implant ----------
def load_tcga(cancer, d=100):
    path = os.path.join(DATA_DIR, f'TCGA_{cancer}_HiSeqV2.tsv')
    df = pd.read_csv(path, sep='\t', index_col=0)  # genes x samples
    X = df.values.T.astype(np.float32)  # samples x genes
    var = np.var(X, axis=0)
    top_idx = np.argsort(var)[-d:]
    X = X[:, top_idx]
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X

def implant_edges(X, m, seed, beta_lo=1.0, beta_hi=1.5, sigma_eps=0.3):
    """Implant m strong causal edges by REPLACING X[:,j] with a strong
    dependence on X[:,i]: X_j = beta*X_i + eps. This gives corr ~0.96,
    clearly above TCGA background co-expression (~0.1-0.5), so implanted
    edges are recoverable and TP is discriminative."""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cancer', type=str, default='BRCA')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--outer', type=int, default=20)
    ap.add_argument('--inner', type=int, default=100)
    args = ap.parse_args()

    X_full = load_tcga(args.cancer, d=100)
    n_full = X_full.shape[0]
    print(f'{args.cancer}: n_full={n_full}, d=100, implant m={args.m}, test n={args.n}, seeds={args.seeds}')

    rows = {'base': [], 'hard': [], 'soft': []}
    for s in range(args.seeds):
        rng = np.random.RandomState(1000 + s)
        idx = rng.choice(n_full, min(args.n, n_full), replace=False)
        X_sub = X_full[idx]
        X_imp, true_edges = implant_edges(X_sub, args.m, seed=2000 + s)
        for name, fn in [('base', lambda: run_baseline(X_imp, args.outer, args.inner)),
                         ('hard', lambda: run_hard(X_imp, args.K, args.outer, args.inner, seed=s)),
                         ('soft', lambda: run_soft(X_imp, args.K, args.outer, args.inner, seed=s))]:
            W = fn()
            pe = W_to_edges(W)
            F1, P, R, tp, fp, npe, nte = score_edges(pe, true_edges)
            rows[name].append((F1, P, R, tp, fp, npe, nte))

    print(f'\n{"method":<9}{"F1":>7}{"Prec":>7}{"Rec":>7}{"TP":>5}{"FP":>5}{"pred_edges":>12}{"true":>5}')
    out = {}
    for name in ['base', 'hard', 'soft']:
        F1 = np.mean([x[0] for x in rows[name]]); P = np.mean([x[1] for x in rows[name]])
        R = np.mean([x[2] for x in rows[name]]); tp = np.mean([x[3] for x in rows[name]])
        fp = np.mean([x[4] for x in rows[name]]); npe = np.mean([x[5] for x in rows[name]])
        nte = np.mean([x[6] for x in rows[name]])
        out[name] = {'f1': float(F1), 'precision': float(P), 'recall': float(R),
                     'tp': float(tp), 'fp': float(fp), 'pred_edges': float(npe), 'true_edges': float(nte)}
        print(f'{name:<9}{F1:>7.3f}{P:>7.3f}{R:>7.3f}{tp:>5.1f}{fp:>5.1f}{npe:>12.1f}{nte:>5.1f}')

    out_dir = r'../../data'
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'tcga_implant_{args.cancer}_n{args.n}.json')
    with open(path, 'w') as f:
        json.dump({'cancer': args.cancer, 'n': args.n, 'm': args.m, 'results': out}, f, indent=2)
    print(f'\nSaved {path}')

if __name__ == '__main__':
    t0 = time.time(); main(); print(f'Total {time.time()-t0:.0f}s')
