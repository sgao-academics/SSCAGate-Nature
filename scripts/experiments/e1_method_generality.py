"""
E1: Is the edge-count pitfall specific to NOTEARS?

The edge-count pitfall (soft gating inflates false edges while F1 collapses)
was established with NOTEARS (trace-exp acyclicity + MSE loss). A reviewer
will ask: is this an artifact of NOTEARS, or a general property of the gating
method? We answer by swapping the acyclicity constraint and the loss while
keeping the soft-gating mechanism identical, and asking whether false-edge
inflation persists.

Three method variants (all torch, all reuse the verified implementations):
  - notears : trace-exp acyclicity  h = trace(exp(W*W)) - d   + MSE loss
  - dagma   : log-det acyclicity    h = -logdet(sI - W*W) + d*log(s)  + MSE
  - golem   : log-det acyclicity    + Gaussian likelihood loss
              (0.5*d*log(RSS)) instead of MSE (GOLEM-EV, Ng et al. 2020)

For each variant we run baseline (no gating) vs soft residual gating (K=3),
across n/d in {0.5, 1, 2}, with ground-truth F1 and edge count. If soft gating
still inflates false edges under dagma and golem, the pitfall is a property of
the gating mechanism, not of NOTEARS.

Checkpoint: ../../data/e1_checkpoint.json (unit-granular, resumable).
"""
import os, sys, json, time, warnings, argparse
warnings.filterwarnings('ignore')
import numpy as np
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT = r'../../data/e1_checkpoint.json'


# ---------------------------------------------------------------- data gen
def make_dag(d, s0, seed):
    rng = np.random.RandomState(seed)
    W = np.zeros((d, d))
    for j in range(d):
        for i in range(j):
            if rng.rand() < s0 / max(d - 1, 1):
                W[j, i] = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])
    return W


def sample_linear_sem(W, n, sigma, seed):
    d = W.shape[0]
    rng = np.random.RandomState(seed)
    X = np.zeros((n, d))
    for j in range(d):
        X[:, j] = X @ W[j, :] + rng.randn(n) * sigma
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
    return F1, P, len(pred)


# ---------------------------------------------------------------- acyclicity
def h_acyc(W, method, s=1.0):
    """DAG constraint h(W) >= 0, =0 iff acyclic."""
    d = W.shape[0]
    if method == 'notears':
        return torch.trace(torch.linalg.matrix_exp(W * W)) - d
    else:  # dagma / golem: log-det acyclicity (Bello et al. 2022)
        # slogdet (not cholesky) so the constraint is defined and differentiable
        # for every W, including cyclic ones.
        M = s * torch.eye(d, device=device) - W * W
        sign, logabsdet = torch.linalg.slogdet(M)
        return -logabsdet + d * np.log(s)


def recon_loss(Xg, W, method, gates=None):
    """Reconstruction / likelihood loss (optionally per-sample gated).

    NOTE: the gated MSE path uses sq.sum(dim=1), NOT sq.mean(dim=1), to match
    the verified soft-gating implementation in spurious_phase_reproduction.py.
    Using mean would shrink loss_d by d and let the entropy term suppress gate
    differentiation (the gating then collapses to uniform and no false edges
    are manufactured).
    """
    n, d = Xg.shape
    M = torch.eye(d, device=device) - W
    sq = (Xg @ M.T).pow(2)           # (n, d)
    if method == 'golem':
        # Gaussian likelihood (GOLEM-EV): n/2 * log(RSS/n), RSS = ||X - XW||_F^2.
        if gates is None:
            rss = sq.sum()            # ||X - XW||_F^2
            return 0.5 * n * torch.log(rss.clamp(min=1e-20) / n)
        else:
            rss_w = (gates * sq.sum(dim=1)).sum() / gates.sum().clamp(min=1.0)
            return 0.5 * n * torch.log(rss_w.clamp(min=1e-20) / n)
    else:
        # MSE loss (notears / dagma)
        if gates is None:
            return sq.mean()
        else:
            return (gates * sq.sum(dim=1)).sum() / gates.sum().clamp(min=1.0)


# ---------------------------------------------------------------- fit
def fit(X, K, method, gated, outer, inner, seed=0, beta_entropy=0.05, l1=0.01):
    n, d = X.shape
    torch.manual_seed(seed)
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    W = torch.zeros(d, d, device=device, requires_grad=True)
    P_logits = None
    opt_P = None
    if gated:
        P_logits = (torch.randn(n, K, device=device) * 0.1).requires_grad_(True)
        opt_P = torch.optim.Adam([P_logits], lr=0.01)
    opt_W = torch.optim.Adam([W], lr=0.002)
    rho, alpha = 1.0, 0.0
    for o in range(outer):
        for _ in range(inner):
            opt_W.zero_grad()
            if gated:
                opt_P.zero_grad()
            gates = None
            if gated:
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
            loss_d = recon_loss(Xg, W, method, gates)
            h = h_acyc(W, method)
            loss = loss_d + 0.5 * rho * h * h + alpha * h + l1 * torch.sum(torch.abs(W))
            if gated:
                logP = torch.log(torch.softmax(P_logits, dim=1).clamp(min=1e-8))
                ent = -(torch.softmax(P_logits, dim=1) * logP).sum(1).mean()
                loss = loss - beta_entropy * ent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(W, 10.0)
            if gated:
                torch.nn.utils.clip_grad_norm_(P_logits, 5.0)
            opt_W.step()
            if gated:
                opt_P.step()
        with torch.no_grad():
            hv = h_acyc(W, method).item()
        if abs(hv) < 1e-8:
            break
        if abs(hv) > 1e-6:
            alpha = alpha + rho * hv
        rho = min(5 * rho, 1e12)
        if torch.isnan(W).any() or torch.isinf(W).any():
            break
    return W.detach().cpu().numpy()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=50)
    ap.add_argument('--s0', type=float, default=1.0)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--outer', type=int, default=20)
    ap.add_argument('--inner', type=int, default=100)
    ap.add_argument('--limit', type=int, default=0, help='cap units (smoke test)')
    args = ap.parse_args()

    methods = ['notears', 'dagma', 'golem']
    nd_list = [0.5, 1.0, 2.0]

    # load / init checkpoint
    ckpt = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'r', encoding='utf-8') as f:
            ckpt = json.load(f)

    W_true = make_dag(args.d, args.s0, seed=42)
    te = true_edges(W_true)
    print(f'E1 method-generality, d={args.d}, true edges={len(te)}, K={args.K}, '
          f'seeds={args.seeds}, device={device}', flush=True)

    units_done = 0
    for method in methods:
        for nd in nd_list:
            n = int(nd * args.d)
            for s in range(args.seeds):
                for gated in [False, True]:
                    key = f'{method}|nd{nd}|s{s}|{"soft" if gated else "base"}'
                    if key in ckpt:
                        continue
                    if args.limit and units_done >= args.limit:
                        print(f'[limit {args.limit} reached, stop]', flush=True)
                        return
                    X = sample_linear_sem(W_true, n, 1.0, seed=1000 + s)
                    W = fit(X, args.K, method, gated, args.outer, args.inner, seed=s)
                    F1, P, e = score(W_to_edges(W), te)
                    ckpt[key] = {'method': method, 'nd': nd, 'seed': s,
                                 'gated': gated, 'f1': F1, 'edges': e}
                    units_done += 1
                    # atomic write
                    tmp = CHECKPOINT + '.tmp'
                    with open(tmp, 'w', encoding='utf-8') as f:
                        json.dump(ckpt, f)
                    os.replace(tmp, CHECKPOINT)
                    print(f'[{units_done}] {method:7s} nd={nd:>3} s={s} '
                          f'{"soft" if gated else "base":4s} -> F1={F1:.3f} edges={e:.1f}',
                          flush=True)

    # summarize
    print('\n=== SUMMARY (mean over seeds) ===', flush=True)
    for method in methods:
        for nd in nd_list:
            bf = [ckpt[f'{method}|nd{nd}|s{s}|base']['f1'] for s in range(args.seeds)]
            sf = [ckpt[f'{method}|nd{nd}|s{s}|soft']['f1'] for s in range(args.seeds)]
            be = [ckpt[f'{method}|nd{nd}|s{s}|base']['edges'] for s in range(args.seeds)]
            se = [ckpt[f'{method}|nd{nd}|s{s}|soft']['edges'] for s in range(args.seeds)]
            print(f'{method:7s} nd={nd:>3}: base_F1={np.mean(bf):.3f} soft_F1={np.mean(sf):.3f} '
                  f'F1_delta={np.mean(sf)-np.mean(bf):+.3f} | base_e={np.mean(be):5.1f} '
                  f'soft_e={np.mean(se):5.1f} edge_delta={np.mean(se)-np.mean(be):+6.1f}',
                  flush=True)
    print(f'\nDone. checkpoint: {CHECKPOINT}', flush=True)


if __name__ == '__main__':
    t0 = time.time(); main(); print(f'Total {time.time()-t0:.0f}s', flush=True)
