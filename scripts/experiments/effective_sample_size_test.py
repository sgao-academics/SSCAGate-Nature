"""

E-C: Measure the effective sample size of residual gating.



Residual gating re-weights each sample by a gate g_i in (0,1). The Kish

effective sample size is n_eff = (sum g_i)^2 / sum g_i^2 <= n. We measure the

final gate distribution of the soft-gated NOTEARS on TCGA semi-synthetic data

and show that n_eff collapses far below n, which is what relaxes the

reconstruction loss relative to the L1/DAG regularizers and inflates false edges.



Design (matches tcga_implant_BRCA_n200):

  TCGA BRCA, d=100, n=200 subsample, m=20 implanted edges.

  baseline : no gate, n_eff = n = 200

  soft     : learned gate, n_eff = (sum g)^2 / sum g^2 << n

Report n_eff, gate-weight summary, edge count, and F1 for both.

"""

import os, sys, json, time, warnings, argparse

warnings.filterwarnings('ignore')

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import numpy as np

import pandas as pd

import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ['OMP_NUM_THREADS'] = '2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



DATA_DIR = r'../../data'



def score_edges(pred_edges, true_edges):

    pred = set(pred_edges); true = set(true_edges)

    tp = len(pred & true); fp = len(pred - true); fn = len(true - pred)

    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    return F1, P, R, tp, fp, len(pred)



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



def run_soft_with_gates(X, K, outer, inner, seed=0, beta_entropy=0.05):

    n, d = X.shape

    torch.manual_seed(seed)

    Xg = torch.tensor(X, dtype=torch.float32, device=device)

    W = torch.zeros(d, d, device=device, requires_grad=True)

    P_logits = (torch.randn(n, K, device=device) * 0.1).requires_grad_(True)

    opt_W = torch.optim.Adam([W], lr=0.002)

    opt_P = torch.optim.Adam([P_logits], lr=0.01)

    rho, alpha = 1.0, 0.0

    final_gates = None

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

        final_gates = gates.detach().clone()

        with torch.no_grad():

            hv = (torch.trace(torch.linalg.matrix_exp(W * W)) - d).item()

        if abs(hv) < 1e-8: break

        if abs(hv) > 1e-6: alpha = alpha + rho * hv

        rho = min(5 * rho, 1e12)

        if torch.isnan(W).any() or torch.isinf(W).any(): break

    return W.detach().cpu().numpy(), final_gates.detach().cpu().numpy()



def W_to_edges(W, thresh=0.3):

    d = W.shape[0]

    return {(i, j) for j in range(d) for i in range(j) if abs(W[j, i]) > thresh}



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



def kish_neff(g):

    g = np.asarray(g, dtype=np.float64)

    s = g.sum()

    return (s * s) / (g * g).sum()



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

    print(f'{args.cancer}: n_full={n_full}, d=100, m={args.m}, n={args.n}, seeds={args.seeds}')



    base_rows, soft_rows = [], []

    neff_list = []

    gate_mean_list, gate_min_list = [], []



    for s in range(args.seeds):

        rng = np.random.RandomState(1000 + s)

        idx = rng.choice(n_full, min(args.n, n_full), replace=False)

        X_sub = X_full[idx]

        X_imp, true_edges = implant_edges(X_sub, args.m, seed=2000 + s)



        Wb = run_baseline(X_imp, args.outer, args.inner)

        base_rows.append(score_edges(W_to_edges(Wb), true_edges))



        Ws, gates = run_soft_with_gates(X_imp, args.K, args.outer, args.inner, seed=s)

        soft_rows.append(score_edges(W_to_edges(Ws), true_edges))

        neff = kish_neff(gates)

        neff_list.append(neff)

        gate_mean_list.append(float(gates.mean()))

        gate_min_list.append(float(gates.min()))



    def agg(rows):

        return {

            'f1': float(np.mean([x[0] for x in rows])),

            'precision': float(np.mean([x[1] for x in rows])),

            'recall': float(np.mean([x[2] for x in rows])),

            'tp': float(np.mean([x[3] for x in rows])),

            'fp': float(np.mean([x[4] for x in rows])),

            'pred_edges': float(np.mean([x[5] for x in rows])),

        }



    base = agg(base_rows); soft = agg(soft_rows)

    n = args.n

    print(f'\n{"method":<10}{"n_eff":>7}{"F1":>7}{"Prec":>7}{"FP":>6}{"pred":>6}')

    print(f'{"baseline":<10}{n:>7.0f}{base["f1"]:>7.3f}{base["precision"]:>7.3f}'

          f'{base["fp"]:>6.1f}{base["pred_edges"]:>6.1f}')

    print(f'{"soft":<10}{np.mean(neff_list):>7.1f}{soft["f1"]:>7.3f}{soft["precision"]:>7.3f}'

          f'{soft["fp"]:>6.1f}{soft["pred_edges"]:>6.1f}')

    print(f'\ngate weight: mean={np.mean(gate_mean_list):.3f}, '

          f'min={np.mean(gate_min_list):.3f}, n_eff per seed={[round(x,1) for x in neff_list]}')

    print(f'n_eff / n = {np.mean(neff_list)/n:.3f}  (Kish: n_eff=(sum g)^2/sum g^2)')



    out = {

        'cancer': args.cancer, 'n': n, 'm': args.m,

        'baseline': base, 'soft': soft,

        'soft_n_eff_mean': float(np.mean(neff_list)),

        'soft_n_eff_per_seed': [float(x) for x in neff_list],

        'gate_mean': float(np.mean(gate_mean_list)),

        'gate_min': float(np.mean(gate_min_list)),

    }

    path = os.path.join(r'../../data', f'neff_{args.cancer}_n{args.n}.json')

    with open(path, 'w') as f:

        json.dump(out, f, indent=2)

    print(f'\nSaved {path}')



if __name__ == '__main__':

    t0 = time.time(); main(); print(f'Total {time.time()-t0:.0f}s')

