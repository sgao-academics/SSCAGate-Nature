"""
Overnight full-scale experiment with checkpoint/resume and perfect reproducibility.

Experiments (all 10 seeds):
  1. cagate_grid    : 9 configs (s0_private x n) x {baseline, per-cluster, oracle}
  2. tcga_percluster: BRCA n=300, m=8 per subgroup x {baseline, per-cluster, oracle}
  3. tcga_gating    : BRCA n=200, m=20 x {baseline, hard, soft}

Checkpoint & resume:
  Every (experiment, config, seed, method) unit is written to
  overnight_checkpoint.json IMMEDIATELY after completion. On restart, completed
  units are skipped. All randomness is seeded, so results are perfectly
  reproducible: re-running resumes and never recomputes a finished unit.

Usage:
  python overnight_master.py --limit 5      # smoke test: only first 5 units
  python overnight_master.py                # full overnight run
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
OUT_DIR = r'../../data'
CHECKPOINT = os.path.join(OUT_DIR, 'overnight_checkpoint.json')

# ================= checkpoint management =================
def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        try:
            with open(CHECKPOINT, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_checkpoint(cp):
    # atomic write: write temp then replace, so a crash mid-write never corrupts
    tmp = CHECKPOINT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cp, f, indent=2)
    os.replace(tmp, CHECKPOINT)

# ================= data generation (cagate_grid, verified) =================
def make_dag(d, s0, seed):
    rng = np.random.RandomState(seed)
    W = np.zeros((d, d))
    for j in range(d):
        for i in range(j):
            if rng.rand() < s0 / max(d - 1, 1):
                W[j, i] = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])
    return W

def make_heterogeneous_dags(d, K_true, s0_shared, s0_private, seed):
    rng = np.random.RandomState(seed)
    W_shared = make_dag(d, s0_shared, seed)
    W_list = []
    for k in range(K_true):
        W_k = W_shared.copy()
        for j in range(d):
            for i in range(j):
                if W_shared[j, i] == 0 and rng.rand() < s0_private / max(d - 1, 1):
                    W_k[j, i] = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])
        W_list.append(W_k)
    return W_list

def sample_sem(W, n, sigma, seed):
    d = W.shape[0]
    rng = np.random.RandomState(seed)
    X = np.zeros((n, d))
    eps = rng.randn(n, d) * sigma
    for j in range(d):
        X[:, j] = X @ W[j, :] + eps[:, j]
    return X

def sample_heterogeneous(W_list, n_total, sigma, seed):
    K = len(W_list)
    Xs, labels = [], []
    for k, W_k in enumerate(W_list):
        nk = n_total // K + (1 if k < n_total % K else 0)
        Xs.append(sample_sem(W_k, nk, sigma, seed + k * 1000))
        labels.extend([k] * nk)
    X = np.vstack(Xs).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, np.array(labels)

def true_edge_set(W_list):
    d = W_list[0].shape[0]
    edges = set()
    for W in W_list:
        for j in range(d):
            for i in range(j):
                if W[j, i] != 0:
                    edges.add((i, j))
    return edges

# ================= NOTEARS core (verified) =================
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

# ================= residual gating (tcga_implant_test, verified) =================
def run_hard(X, K, outer, inner, seed=0):
    d = X.shape[1]
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    cid = torch.tensor(KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X).labels_, device=device)
    W = torch.zeros(d, d, device=device, requires_grad=True)
    rho, alpha = 1.0, 0.0
    opt = torch.optim.Adam([W], lr=0.002)
    def _gate_hard(res, cid, alpha_g=0.5):
        gates = torch.ones_like(res)
        for c in torch.unique(cid):
            mask = (cid == c)
            if mask.sum() < 3: continue
            r_c = res[mask]; med = torch.median(r_c); mad = torch.median(torch.abs(r_c - med))
            if mad < 1e-8: continue
            gates[mask] = torch.sigmoid(-alpha_g * (r_c - med) / mad)
        return gates
    for o in range(outer):
        for _ in range(inner):
            opt.zero_grad()
            M = torch.eye(d, device=device) - W
            sq = (Xg @ M.T).pow(2)
            res = sq.mean(dim=1)
            gates = _gate_hard(res, cid)
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

# ================= TCGA data =================
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
    return X, set(edges)

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

# ================= main orchestration =================
def paired_t_test(a, b):
    d = np.array(a) - np.array(b)
    n = len(d)
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else float('inf')
    t = mean_d / se if se > 0 else 0.0
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        p = float('nan')
    return mean_d, sd, t, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--outer', type=int, default=15)
    ap.add_argument('--inner', type=int, default=100)
    ap.add_argument('--limit', type=int, default=None, help='max units to run (smoke test)')
    args = ap.parse_args()

    cp = load_checkpoint()
    n_done = len(cp)
    print(f'checkpoint loaded: {n_done} units already done; device={device}')

    # build unit list
    units = []  # (exp, config_key, seed, method)
    s0_priv_list = [0.2, 0.4, 0.8]
    n_list = [200, 300, 600]
    for s0p in s0_priv_list:
        for n in n_list:
            cfg = f's0p{s0p}_n{n}'
            for s in range(args.seeds):
                for m in ['baseline', 'percluster', 'oracle']:
                    units.append(('grid', cfg, s, m))
    for s in range(args.seeds):
        for m in ['baseline', 'percluster', 'oracle']:
            units.append(('tcga_pc', 'BRCA_n300', s, m))
    for s in range(args.seeds):
        for m in ['baseline', 'hard', 'soft']:
            units.append(('tcga_gating', 'BRCA_n200', s, m))

    total = len(units)
    print(f'total units: {total}')

    # cache for data (avoid regenerating the same data across methods)
    grid_cache = {}
    tcga_pc_cache = {}
    tcga_gating_cache = {}

    run_count = 0
    t0 = time.time()
    for exp, cfg, s, method in units:
        key = f'{exp}/{cfg}/seed{s}/{method}'
        if key in cp:
            continue
        # ---- generate data (cached per (exp, cfg, seed)) ----
        if exp == 'grid':
            dcache_key = (cfg, s)
            if dcache_key not in grid_cache:
                s0p = float(cfg.split('_')[0].replace('s0p', ''))
                n = int(cfg.split('_n')[1])
                W_list = make_heterogeneous_dags(50, 3, 1.0, s0p, seed=42 + s)
                X, labels_true = sample_heterogeneous(W_list, n, 1.0, seed=1000 + s)
                te = true_edge_set(W_list)
                grid_cache[dcache_key] = (X, labels_true, te)
            X, labels_true, te = grid_cache[dcache_key]
            if method == 'baseline':
                res = F1_of(W_to_edges(run_notears_np(X, args.outer, args.inner, s)), te)
            elif method == 'percluster':
                labels_km = KMeans(n_clusters=3, random_state=s, n_init=10).fit(X).labels_
                res = F1_of(cluster_union_edges(X, labels_km, args.outer, args.inner, s), te)
            else:
                res = F1_of(cluster_union_edges(X, labels_true, args.outer, args.inner, s), te)
        elif exp == 'tcga_pc':
            dcache_key = s
            if dcache_key not in tcga_pc_cache:
                X_full = load_tcga('BRCA', d=100)
                rng = np.random.RandomState(1000 + s)
                idx = rng.choice(X_full.shape[0], 300, replace=False)
                X_sub = X_full[idx]
                labels_true = KMeans(n_clusters=3, random_state=s, n_init=10).fit(X_sub).labels_
                X_imp, te = implant_cluster_specific_edges(X_sub, labels_true, 8, seed=2000 + s)
                tcga_pc_cache[dcache_key] = (X_imp, labels_true, te)
            X_imp, labels_true, te = tcga_pc_cache[dcache_key]
            if method == 'baseline':
                res = F1_of(W_to_edges(run_notears_np(X_imp, args.outer, args.inner, s)), te)
            elif method == 'percluster':
                labels_km = KMeans(n_clusters=3, random_state=s, n_init=10).fit(X_imp).labels_
                res = F1_of(cluster_union_edges(X_imp, labels_km, args.outer, args.inner, s), te)
            else:
                res = F1_of(cluster_union_edges(X_imp, labels_true, args.outer, args.inner, s), te)
        else:  # tcga_gating
            dcache_key = s
            if dcache_key not in tcga_gating_cache:
                X_full = load_tcga('BRCA', d=100)
                rng = np.random.RandomState(1000 + s)
                idx = rng.choice(X_full.shape[0], 200, replace=False)
                X_sub = X_full[idx]
                X_imp, te = implant_edges(X_sub, 20, seed=2000 + s)
                tcga_gating_cache[dcache_key] = (X_imp, te)
            X_imp, te = tcga_gating_cache[dcache_key]
            if method == 'baseline':
                res = F1_of(W_to_edges(run_notears_np(X_imp, args.outer, args.inner, s)), te)
            elif method == 'hard':
                res = F1_of(W_to_edges(run_hard(X_imp, 3, args.outer, args.inner, seed=s)), te)
            else:
                res = F1_of(W_to_edges(run_soft(X_imp, 3, args.outer, args.inner, seed=s)), te)

        cp[key] = {'exp': exp, 'config': cfg, 'seed': s, 'method': method, **res}
        run_count += 1
        save_checkpoint(cp)
        elapsed = time.time() - t0
        print(f'[{run_count}/{total}] {key}: F1={res["f1"]:.3f} P={res["precision"]:.3f} '
              f'edges={res["n_edges"]:.0f} | elapsed {elapsed:.0f}s', flush=True)

        if args.limit is not None and run_count >= args.limit:
            print(f'[limit reached] stopped after {run_count} units')
            break

    print(f'\nDone. ran {run_count} units this session, checkpoint now has {len(cp)} units.')

    # ---- compute paired t-test for grid per-cluster vs baseline (if complete) ----
    print('\n=== paired t-test (per-cluster vs baseline), grid configs ===')
    for s0p in s0_priv_list:
        for n in n_list:
            cfg = f's0p{s0p}_n{n}'
            base, pc = [], []
            complete = True
            for s in range(args.seeds):
                bk = f'grid/{cfg}/seed{s}/baseline'
                pk = f'grid/{cfg}/seed{s}/percluster'
                if bk not in cp or pk not in cp:
                    complete = False
                    break
                base.append(cp[bk]['f1']); pc.append(cp[pk]['f1'])
            if complete:
                mean_d, sd, t, p = paired_t_test(pc, base)
                print(f'{cfg}: gain={mean_d:+.3f} +/- {sd:.3f}, t={t:.3f}, p={p:.4g}')
            else:
                print(f'{cfg}: incomplete ({len(pc)}/{args.seeds} seeds done)')

if __name__ == '__main__':
    t0 = time.time(); main(); print(f'\nTotal wall {time.time()-t0:.0f}s')
