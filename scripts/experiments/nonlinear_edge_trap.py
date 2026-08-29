"""

E-E: Does the edge-count trap persist under nonlinear data generation?



If the trap were an artifact of linear-Gaussian data, it would disappear once

the data are generated from a nonlinear SEM. Here we generate data from a

nonlinear SEM (x_j = tanh(sum_i W[j,i] x_i) + eps), fit a (misspecified) linear

NOTEARS, and ask whether residual gating still inflates false edges relative to

baseline. If it does, the trap is a property of the gating method, not of the

data-generating process.



Design: d=50, single DAG, nonlinear sampling. n/d in {2, 4}. baseline vs soft

residual gating (K=3). Metric: F1 and edge count against the true edge set.

"""

import os, sys, json, time, warnings, argparse

warnings.filterwarnings('ignore')

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import numpy as np

import torch

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



def sample_nonlinear_sem(W, n, sigma, seed):

    """x_j = tanh(sum_i W[j,i] x_i) + eps_j  (nonlinear in the parents)."""

    d = W.shape[0]

    rng = np.random.RandomState(seed)

    X = np.zeros((n, d))

    eps = rng.randn(n, d) * sigma

    for j in range(d):

        X[:, j] = np.tanh(X @ W[j, :]) + eps[:, j]

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



def run_baseline(X, outer, inner):

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



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('--d', type=int, default=50)

    ap.add_argument('--s0', type=float, default=1.0)

    ap.add_argument('--K', type=int, default=3)

    ap.add_argument('--seeds', type=int, default=3)

    ap.add_argument('--outer', type=int, default=20)

    ap.add_argument('--inner', type=int, default=100)

    args = ap.parse_args()



    nd_list = [2.0, 4.0]

    results = {}



    W_true = make_dag(args.d, args.s0, seed=42)

    te = true_edges(W_true)

    print(f'nonlinear SEM, d={args.d}, true edges={len(te)}, K={args.K}, seeds={args.seeds}')



    for nd in nd_list:

        n = int(nd * args.d)

        base_f1, base_e, soft_f1, soft_e = [], [], [], []

        for s in range(args.seeds):

            X = sample_nonlinear_sem(W_true, n, 1.0, seed=1000 + s)

            F1b, Pb, eb = score(W_to_edges(run_baseline(X, args.outer, args.inner)), te)

            F1s, Ps, es = score(W_to_edges(run_soft(X, args.K, args.outer, args.inner, seed=s)), te)

            base_f1.append(F1b); base_e.append(eb)

            soft_f1.append(F1s); soft_e.append(es)

        results[str(nd)] = {

            'n': n,

            'baseline_edges': float(np.mean(base_e)), 'soft_edges': float(np.mean(soft_e)),

            'edge_delta': float(np.mean(soft_e) - np.mean(base_e)),

            'baseline_f1': float(np.mean(base_f1)), 'soft_f1': float(np.mean(soft_f1)),

            'f1_delta': float(np.mean(soft_f1) - np.mean(base_f1)),

        }

        print(f'n/d={nd:>4}: base_edges={np.mean(base_e):6.1f} soft_edges={np.mean(soft_e):6.1f} '

              f'edge_delta={np.mean(soft_e)-np.mean(base_e):+6.1f} | base_F1={np.mean(base_f1):.3f} '

              f'soft_F1={np.mean(soft_f1):.3f} F1_delta={np.mean(soft_f1)-np.mean(base_f1):+.3f}')



    path = r'../../data/nonlinear_edge_trap.json'

    with open(path, 'w') as f:

        json.dump({'d': args.d, 'K': args.K, 'nonlinear': True, 'results': results}, f, indent=2)

    print(f'\nSaved {path}')



if __name__ == '__main__':

    t0 = time.time(); main(); print(f'Total {time.time()-t0:.0f}s')

