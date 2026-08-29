"""

E-F: Direct evidence for the gate-W positive feedback.



Track, during soft-gating training, how the number of edges in W and the

concentration of the gate weights co-evolve. The positive-feedback mechanism

predicts that, as W acquires edges (many of them false), the gate weights become

more concentrated (larger std, smaller Kish n_eff) --- the two reinforce each

other. We compare against the baseline (no gate), whose edge count stays stable.



Two outputs per outer iteration:

  edge_count = number of edges |W| > 0.3

  gate_std   = std of the per-sample gate weights (concentration)

  n_eff      = Kish effective sample size



Design: TCGA semi-synthetic (BRCA, n=200, d=100, m=20 implanted edges), seeds=3.

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

    return int((np.abs(W) > thresh).sum() / 2)  # count (i,j) once (upper triangle)



def kish_neff(g):

    g = np.asarray(g, dtype=np.float64)

    s = g.sum()

    return (s * s) / (g * g).sum()



def run_soft_trajectory(X, K, outer, inner, seed=0, beta_entropy=0.05):

    n, d = X.shape

    torch.manual_seed(seed)

    Xg = torch.tensor(X, dtype=torch.float32, device=device)

    W = torch.zeros(d, d, device=device, requires_grad=True)

    P_logits = (torch.randn(n, K, device=device) * 0.1).requires_grad_(True)

    opt_W = torch.optim.Adam([W], lr=0.002)

    opt_P = torch.optim.Adam([P_logits], lr=0.01)

    rho, alpha = 1.0, 0.0

    traj = []

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

        # record at end of each outer iteration

        Wn = W.detach().cpu().numpy()

        gn = gates.detach().cpu().numpy()

        traj.append({

            'iter': o + 1,

            'edge_count': edge_count(Wn),

            'gate_std': float(gn.std()),

            'n_eff': float(kish_neff(gn)),

        })

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

    print(f'{args.cancer}: n={args.n}, m={args.m}, K={args.K}, seeds={args.seeds}, outer={args.outer}')



    # aggregate trajectories over seeds

    max_iter = args.outer

    soft_edge = np.zeros(max_iter); soft_std = np.zeros(max_iter); soft_neff = np.zeros(max_iter)

    base_edge = np.zeros(max_iter)

    count = np.zeros(max_iter)

    all_soft_traj = []



    for s in range(args.seeds):

        rng = np.random.RandomState(1000 + s)

        idx = rng.choice(n_full, min(args.n, n_full), replace=False)

        X_sub = X_full[idx]

        X_imp, _ = implant_edges(X_sub, args.m, seed=2000 + s)



        st = run_soft_trajectory(X_imp, args.K, args.outer, args.inner, seed=s)

        bt = run_baseline_trajectory(X_imp, args.outer, args.inner)

        all_soft_traj.append(st)

        for rec in st:

            i = rec['iter'] - 1

            soft_edge[i] += rec['edge_count']; soft_std[i] += rec['gate_std']; soft_neff[i] += rec['n_eff']

            count[i] += 1

        for rec in bt:

            i = rec['iter'] - 1

            base_edge[i] += rec['edge_count']



    count[count == 0] = 1

    soft_edge /= count; soft_std /= count; soft_neff /= count

    base_edge /= count



    print(f'\n{"iter":>5}{"base_edges":>12}{"soft_edges":>12}{"gate_std":>10}{"n_eff":>9}')

    out_traj = []

    for i in range(max_iter):

        print(f'{i+1:>5}{base_edge[i]:>12.1f}{soft_edge[i]:>12.1f}{soft_std[i]:>10.4f}{soft_neff[i]:>9.1f}')

        out_traj.append({'iter': i+1, 'baseline_edges': float(base_edge[i]),

                         'soft_edges': float(soft_edge[i]), 'gate_std': float(soft_std[i]),

                         'n_eff': float(soft_neff[i])})



    path = r'../../data/gate_feedback_trajectory.json'

    with open(path, 'w') as f:

        json.dump({'cancer': args.cancer, 'n': args.n, 'trajectory': out_traj,

                   'soft_full_trajectories': all_soft_traj}, f, indent=2)

    print(f'\nSaved {path}')



if __name__ == '__main__':

    t0 = time.time(); main(); print(f'Total {time.time()-t0:.0f}s')

