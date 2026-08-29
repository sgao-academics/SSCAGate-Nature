"""
Shared core for SSCAGate supplementary experiments.

One source of truth for data generation, NOTEARS-family fits, gating variants,
and evaluation, so every experiment is reproducible and identical to the
verified implementations used in the main paper.

Verified-implementation provenance (do NOT deviate):
  - NOTEARS trace-exp + MSE      : e1_method_generality.py (frozen, verified)
  - DAGMA  log-det (slogdet!)    : e1_method_generality.py (cholesky BUG fixed)
  - GOLEM  log-det + Gaussian LL : e1_method_generality.py (n-scaling, not d)
  - soft gating (sq.sum dim=1!)  : e1_method_generality.py (mean BUG fixed)
  - frozen gate                  : e4_gate_fixed_vs_joint.py
  - hard gating trajectory       : e10_hard_gating_trajectory.py
  - heterogeneous data gen       : cagate_grid.py

Key bugs already fixed and MUST stay fixed:
  1. log-det acyclicity uses torch.linalg.slogdet (NOT cholesky).
  2. gated MSE uses sq.sum(dim=1) (NOT sq.mean(dim=1)).
  3. GOLEM likelihood uses n-scaling n/2*log(RSS/n) (NOT d-scaling).
  4. No sys.path pollution (do NOT insert [conda site-packages]).
"""
import os, json, time, warnings
import numpy as np
import torch
from sklearn.cluster import KMeans

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================ data gen
def make_dag(d, s0, seed):
    """Random DAG, edges oriented low-index -> high-index (topological order)."""
    rng = np.random.RandomState(seed)
    W = np.zeros((d, d))
    for j in range(d):
        for i in range(j):
            if rng.rand() < s0 / max(d - 1, 1):
                W[j, i] = rng.uniform(0.5, 1.0) * rng.choice([-1.0, 1.0])
    return W


def make_heterogeneous_dags(d, K_true, s0_shared, s0_private, seed):
    """K subpopulations sharing a skeleton plus subpopulation-private edges."""
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


def sample_linear_sem(W, n, sigma, seed):
    """Linear SEM: x_j = sum_i W[j,i] x_i + eps_j, standardized."""
    d = W.shape[0]
    rng = np.random.RandomState(seed)
    X = np.zeros((n, d))
    for j in range(d):
        X[:, j] = X @ W[j, :] + rng.randn(n) * sigma
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X.astype(np.float32)


def sample_nonlinear_sem(W, n, sigma, seed):
    """Nonlinear SEM: x_j = tanh(sum_i W[j,i] x_i) + eps_j (misspecification test)."""
    d = W.shape[0]
    rng = np.random.RandomState(seed)
    X = np.zeros((n, d))
    for j in range(d):
        X[:, j] = np.tanh(X @ W[j, :]) + rng.randn(n) * sigma
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X.astype(np.float32)


def sample_heterogeneous(W_list, n_total, sigma, seed):
    """Sample from K subpopulations, evenly split, standardized."""
    K = len(W_list)
    Xs, labels = [], []
    for k, W_k in enumerate(W_list):
        nk = n_total // K + (1 if k < n_total % K else 0)
        Xs.append(sample_linear_sem(W_k, nk, sigma, seed + k * 1000))
        labels.extend([k] * nk)
    X = np.vstack(Xs).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, np.array(labels)


def sample_heterogeneous_nonlinear(W_list, n_total, sigma, seed):
    """Nonlinear (tanh) heterogeneous sampling, for the misspecification test."""
    K = len(W_list)
    Xs, labels = [], []
    for k, W_k in enumerate(W_list):
        nk = n_total // K + (1 if k < n_total % K else 0)
        Xs.append(sample_nonlinear_sem(W_k, nk, sigma, seed + k * 1000))
        labels.extend([k] * nk)
    X = np.vstack(Xs).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, np.array(labels)


# ================================================================ evaluation
def W_to_edges(W, thresh=0.3):
    """Directed edges (i->j, i<j) implied by lower-triangular W."""
    d = W.shape[0]
    return {(i, j) for j in range(d) for i in range(j) if abs(W[j, i]) > thresh}


def true_edges(W):
    d = W.shape[0]
    return {(i, j) for j in range(d) for i in range(j) if W[j, i] != 0}


def true_edge_set(W_list):
    """Union of subpopulation edge sets."""
    edges = set()
    for W in W_list:
        edges |= true_edges(W)
    return edges


def score_undirected(pred, true):
    """Edge-existence F1 (direction ignored), plus precision/recall/counts."""
    pred = set(pred); true = set(true)
    tp = len(pred & true); fp = len(pred - true); fn = len(true - pred)
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return F1, P, R, tp, fp, fn, len(pred)


def score_directed(pred, true):
    """Structural Hamming Distance (direction-sensitive)."""
    pred = set(pred); true = set(true)
    missing = len(true - pred)                      # true edge not recovered
    extra = len(pred - true)                        # spurious edge
    reversed_ = sum(1 for (i, j) in pred if (j, i) in true)
    shd = missing + extra + reversed_
    return shd, missing, extra, reversed_


# ================================================================ acyclicity / loss
def h_acyc(W, method, s=1.0):
    d = W.shape[0]
    if method == 'notears':
        return torch.trace(torch.linalg.matrix_exp(W * W)) - d
    else:
        M = s * torch.eye(d, device=device) - W * W
        sign, logabsdet = torch.linalg.slogdet(M)
        return -logabsdet + d * np.log(s)


def recon_loss(Xg, W, method, gates=None):
    n, d = Xg.shape
    M = torch.eye(d, device=device) - W
    sq = (Xg @ M.T).pow(2)                       # (n, d)
    if method == 'golem':
        if gates is None:
            rss = sq.sum()
            return 0.5 * n * torch.log(rss.clamp(min=1e-20) / n)
        else:
            rss_w = (gates * sq.sum(dim=1)).sum() / gates.sum().clamp(min=1.0)
            return 0.5 * n * torch.log(rss_w.clamp(min=1e-20) / n)
    else:
        if gates is None:
            return sq.mean()
        else:
            return (gates * sq.sum(dim=1)).sum() / gates.sum().clamp(min=1.0)


# ================================================================ gating helpers
def soft_gates_from_logits(Xg, W, P_logits, K):
    """Soft-assignment gates from within-cluster residual dispersion."""
    d = W.shape[0]
    M = torch.eye(d, device=device) - W
    sq = (Xg @ M.T).pow(2)
    res = sq.mean(dim=1)                          # per-sample mean residual
    P_soft = torch.softmax(P_logits, dim=1)
    cw = P_soft.sum(0)
    wm = (P_soft * res.unsqueeze(1)).sum(0) / cw.clamp(min=1e-8)
    wv = (P_soft * (res.unsqueeze(1) - wm.unsqueeze(0)) ** 2).sum(0) / cw.clamp(min=1e-8)
    cstd = torch.sqrt(wv.clamp(min=1e-8))
    smed = torch.median(cstd)
    raw = torch.sigmoid(0.5 * (smed / cstd.clamp(min=1e-8) - 1))
    return (P_soft * raw.unsqueeze(0)).sum(1)


def hard_gates_from_labels(Xg, W, labels, K):
    """Hard-assignment gates: per-sample gate from its cluster's residual dispersion."""
    d = W.shape[0]
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
    return torch.tensor([raw[labels[i]].item() for i in range(Xg.shape[0])], device=device)


# ================================================================ fits
def fit(X, K, method='notears', gated='none', outer=20, inner=100, seed=0,
        beta_entropy=0.05, l1=0.01, lr=0.002):
    """
    Fit a single weight matrix.
    gated in {'none', 'soft', 'hard', 'fixed'}:
      none  -> uniform (baseline)
      soft  -> learned softmax assignment, jointly optimized with W
      hard  -> K-means hard labels, gate recomputed each step from residuals
      fixed -> gate frozen from an initial short baseline fit (E4)
    Returns W (numpy d x d).
    """
    n, d = X.shape
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    W = torch.zeros(d, d, device=device, requires_grad=True)
    P_logits = None
    opt_P = None
    frozen_gates = None
    hard_labels = None

    if gated in ('soft',):
        P_logits = (torch.randn(n, K, device=device) * 0.1).requires_grad_(True)
        opt_P = torch.optim.Adam([P_logits], lr=0.01)
    elif gated == 'hard':
        hard_labels = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X).labels_
    elif gated == 'fixed':
        frozen_gates = _compute_frozen_gates(X, K, seed)

    opt_W = torch.optim.Adam([W], lr=lr)
    rho, alpha = 1.0, 0.0
    for o in range(outer):
        for _ in range(inner):
            opt_W.zero_grad()
            if opt_P is not None:
                opt_P.zero_grad()
            if gated == 'soft':
                gates = soft_gates_from_logits(Xg, W, P_logits, K)
            elif gated == 'hard':
                gates = hard_gates_from_labels(Xg, W, hard_labels, K)
            elif gated == 'fixed':
                gates = frozen_gates
            else:
                gates = None
            loss_d = recon_loss(Xg, W, method, gates)
            h = h_acyc(W, method)
            loss = loss_d + 0.5 * rho * h * h + alpha * h + l1 * torch.sum(torch.abs(W))
            if gated == 'soft':
                P_soft = torch.softmax(P_logits, dim=1)
                logP = torch.log(P_soft.clamp(min=1e-8))
                ent = -(P_soft * logP).sum(1).mean()
                loss = loss - beta_entropy * ent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(W, 10.0)
            if opt_P is not None:
                torch.nn.utils.clip_grad_norm_(P_logits, 5.0)
            opt_W.step()
            if opt_P is not None:
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


def _compute_frozen_gates(X, K, seed):
    """Gate from a short baseline fit, then frozen (E4)."""
    n, d = X.shape
    Xg = torch.tensor(X, dtype=torch.float32, device=device)
    Winit = torch.tensor(fit(X, K, 'notears', 'none', outer=5, inner=100, seed=seed),
                         dtype=torch.float32, device=device)
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
    return torch.tensor([raw[labels[i]].item() for i in range(n)], device=device)


def per_cluster_union(X, labels, K, outer, inner, seed, method='notears'):
    """Run NOTEARS within each cluster and union the edges."""
    d = X.shape[1]
    edges = set()
    for c in np.unique(labels):
        mask = labels == c
        if mask.sum() < d + 1:
            continue
        Wc = fit(X[mask], K, method, 'none', outer, inner, seed=seed + int(c) * 1000)
        edges |= W_to_edges(Wc)
    return edges


# ================================================================ checkpoint
def load_cp(path):
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}


def save_cp(cp, path):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cp, f)
    os.replace(tmp, path)
