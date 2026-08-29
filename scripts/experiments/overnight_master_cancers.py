"""

Overnight full-scale pan-cancer experiment (33 cancers x 20 seeds).



This is the truly long overnight job. For each of the 33 TCGA cancers we run:

  - gating   : baseline / hard / soft on implanted edges (the "break" side)

  - per-cluster (only if n_full >= 3*(d+1)) : baseline / per-cluster / oracle

               on cluster-specific implanted edges (the "build" side)



Checkpoint & resume: independent checkpoint file overnight_checkpoint_cancers.json,

written atomically after every (cancer, seed, method) unit. Re-running resumes.



Usage:

  python overnight_master_cancers.py --limit 3   # smoke test

  python overnight_master_cancers.py             # full overnight (long)

"""

import os, sys, json, time, warnings, argparse

warnings.filterwarnings('ignore')

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import numpy as np

import pandas as pd

import torch

from sklearn.cluster import KMeans

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ['OMP_NUM_THREADS'] = '2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



DATA_DIR = r'../../data'

OUT_DIR = r'../../data'

CHECKPOINT = os.path.join(OUT_DIR, 'overnight_checkpoint_cancers.json')



CANCERS = ['ACC','BLCA','BRCA','CESC','CHOL','COAD','DLBC','ESCA','GBM','HNSC',

           'KICH','KIRC','KIRP','LAML','LGG','LIHC','LUAD','LUSC','MESO','OV',

           'PAAD','PCPG','PRAD','READ','SARC','SKCM','STAD','TGCT','THCA','THYM',

           'UCEC','UCS','UVM']



def load_checkpoint():

    if os.path.exists(CHECKPOINT):

        try:

            with open(CHECKPOINT, 'r', encoding='utf-8') as f:

                return json.load(f)

        except Exception:

            return {}

    return {}



def save_checkpoint(cp):

    tmp = CHECKPOINT + '.tmp'

    with open(tmp, 'w', encoding='utf-8') as f:

        json.dump(cp, f, indent=2)

    os.replace(tmp, CHECKPOINT)



# ---- NOTEARS core ----

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



def run_hard(X, K, outer, inner, seed=0):

    d = X.shape[1]

    Xg = torch.tensor(X, dtype=torch.float32, device=device)

    cid = torch.tensor(KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X).labels_, device=device)

    W = torch.zeros(d, d, device=device, requires_grad=True)

    rho, alpha = 1.0, 0.0

    opt = torch.optim.Adam([W], lr=0.002)

    def _gate(res, cid, ag=0.5):

        gates = torch.ones_like(res)

        for c in torch.unique(cid):

            mask = (cid == c)

            if mask.sum() < 3: continue

            r_c = res[mask]; med = torch.median(r_c); mad = torch.median(torch.abs(r_c - med))

            if mad < 1e-8: continue

            gates[mask] = torch.sigmoid(-ag * (r_c - med) / mad)

        return gates

    for o in range(outer):

        for _ in range(inner):

            opt.zero_grad()

            M = torch.eye(d, device=device) - W

            sq = (Xg @ M.T).pow(2)

            res = sq.mean(dim=1)

            gates = _gate(res, cid)

            loss_d = (gates * sq.sum(dim=1)).sum() / gates.sum().clamp(min=1)

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



def W_to_edges(W, thresh=0.3):

    d = W.shape[0]

    return {(i, j) for j in range(d) for i in range(j) if abs(W[j, i]) > thresh}



def F1_of(pred, true):

    tp = len(pred & true); fp = len(pred - true); fn = len(true - pred)

    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    return {'f1': float(F1), 'precision': float(P), 'recall': float(R),

            'tp': float(tp), 'fp': float(fp), 'n_edges': float(len(pred))}



def cluster_union_edges(X, labels, outer, inner, seed):

    d = X.shape[1]

    edges = set()

    for c in np.unique(labels):

        mask = labels == c

        if mask.sum() < d + 1:

            continue

        Wc = run_notears_np(X[mask], outer, inner, seed + int(c) * 1000)

        edges |= W_to_edges(Wc)

    return edges



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

    edges = set()

    used = set()

    while len(edges) < m:

        i = rng.randint(0, d); j = rng.randint(0, d)

        if i == j or (i, j) in used or (j, i) in used: continue

        used.add((i, j))

        beta = rng.uniform(beta_lo, beta_hi) * rng.choice([-1.0, 1.0])

        X[:, j] = beta * X[:, i] + sigma_eps * rng.randn(n)

        edges.add((i, j))

    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    return X, edges



def implant_cluster_specific_edges(X, labels, m_per_cluster, seed,

                                   beta_lo=1.0, beta_hi=1.5, sigma_eps=0.3):

    rng = np.random.RandomState(seed)

    n, d = X.shape

    X = X.copy()

    edges = set()

    for c in np.unique(labels):

        idx_c = np.where(labels == c)[0]

        used = set()

        placed = 0

        tries = 0

        while placed < m_per_cluster and tries < m_per_cluster * 50:

            tries += 1

            i = rng.randint(0, d); j = rng.randint(0, d)

            if i == j or (i, j) in used or (j, i) in used:

                continue

            used.add((i, j))

            beta = rng.uniform(beta_lo, beta_hi) * rng.choice([-1.0, 1.0])

            X[idx_c, j] = beta * X[idx_c, i] + sigma_eps * rng.randn(len(idx_c))

            edges.add((i, j))

            placed += 1

    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    return X, edges



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument('--seeds', type=int, default=20)

    ap.add_argument('--outer', type=int, default=15)

    ap.add_argument('--inner', type=int, default=100)

    ap.add_argument('--limit', type=int, default=None)

    ap.add_argument('--d', type=int, default=100)

    ap.add_argument('--K', type=int, default=3)

    ap.add_argument('--n', type=int, default=300)

    args = ap.parse_args()



    cp = load_checkpoint()

    print(f'checkpoint loaded: {len(cp)} units; device={device}')



    # build unit list: gating (all cancers) + per-cluster (n_full >= 3*(d+1))

    units = []

    n_full_map = {}

    for cancer in CANCERS:

        Xf = load_tcga(cancer, d=args.d)

        nf = Xf.shape[0]

        n_full_map[cancer] = nf

        for s in range(args.seeds):

            for m in ['baseline', 'hard', 'soft']:

                units.append(('gating', cancer, s, m))

            if nf >= 3 * (args.d + 1):

                for m in ['baseline', 'percluster', 'oracle']:

                    units.append(('percluster', cancer, s, m))

    total = len(units)

    pc_cancers = sorted(c for c in CANCERS if n_full_map[c] >= 3 * (args.d + 1))

    print(f'total units: {total} ({"+".join(pc_cancers)} have per-cluster)')



    data_cache = {}

    run_count = 0

    t0 = time.time()

    for exp, cancer, s, method in units:

        key = f'{exp}/{cancer}/seed{s}/{method}'

        if key in cp:

            continue

        dcache_key = (exp, cancer, s)

        if dcache_key not in data_cache:

            X_full = load_tcga(cancer, d=args.d)

            n_full = X_full.shape[0]

            rng = np.random.RandomState(1000 + s)

            n_use = min(args.n, n_full)

            idx = rng.choice(n_full, n_use, replace=False)

            X_sub = X_full[idx]

            if exp == 'gating':

                X_imp, te = implant_edges(X_sub, 20, seed=2000 + s)

                labels = None

            else:

                labels_true = KMeans(n_clusters=args.K, random_state=s, n_init=10).fit(X_sub).labels_

                X_imp, te = implant_cluster_specific_edges(X_sub, labels_true, 8, seed=2000 + s)

                labels = labels_true

            data_cache[dcache_key] = (X_imp, te, labels)

        X_imp, te, labels = data_cache[dcache_key]



        if exp == 'gating':

            if method == 'baseline':

                res = F1_of(W_to_edges(run_notears_np(X_imp, args.outer, args.inner, s)), te)

            elif method == 'hard':

                res = F1_of(W_to_edges(run_hard(X_imp, args.K, args.outer, args.inner, seed=s)), te)

            else:

                res = F1_of(W_to_edges(run_soft(X_imp, args.K, args.outer, args.inner, seed=s)), te)

        else:

            if method == 'baseline':

                res = F1_of(W_to_edges(run_notears_np(X_imp, args.outer, args.inner, s)), te)

            elif method == 'percluster':

                labels_km = KMeans(n_clusters=args.K, random_state=s, n_init=10).fit(X_imp).labels_

                res = F1_of(cluster_union_edges(X_imp, labels_km, args.outer, args.inner, s), te)

            else:

                res = F1_of(cluster_union_edges(X_imp, labels, args.outer, args.inner, s), te)



        cp[key] = {'exp': exp, 'cancer': cancer, 'seed': s, 'method': method, **res}

        run_count += 1

        save_checkpoint(cp)

        elapsed = time.time() - t0

        print(f'[{run_count}/{total}] {key}: F1={res["f1"]:.3f} P={res["precision"]:.3f} '

              f'edges={res["n_edges"]:.0f} | elapsed {elapsed:.0f}s', flush=True)

        if args.limit is not None and run_count >= args.limit:

            print(f'[limit reached] stopped after {run_count} units')

            break



    print(f'\nDone. ran {run_count} units this session; checkpoint has {len(cp)} units.')



if __name__ == '__main__':

    t0 = time.time(); main(); print(f'\nTotal wall {time.time()-t0:.0f}s')

