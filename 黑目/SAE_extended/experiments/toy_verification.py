"""Toy-model numerical verification of the scaling-law predictions in
main.tex § 6.7. Implements the protocol in § 7."""

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_ground_truth(N, M, seed):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((M, N))
    G /= np.linalg.norm(G, axis=0, keepdims=True)
    return torch.tensor(G, dtype=torch.float32, device=DEVICE)


def sample_batch(G, k, batch_size, noise_std, rng):
    N = G.shape[1]
    Z = torch.zeros(batch_size, N, device=DEVICE)
    for b in range(batch_size):
        idx = rng.choice(N, size=k, replace=False)
        Z[b, idx] = torch.tensor(rng.uniform(0.0, 1.0, size=k), dtype=torch.float32)
    X = Z @ G.T
    if noise_std > 0:
        X = X + noise_std * torch.randn_like(X)
    return X, Z


class TopKSAE(nn.Module):
    def __init__(self, M, L, k_topk):
        super().__init__()
        self.W_e = nn.Parameter(torch.randn(L, M) * 0.1)
        self.b_e = nn.Parameter(torch.zeros(L))
        self.W_d = nn.Parameter(torch.randn(M, L) * 0.1)
        self.b_d = nn.Parameter(torch.zeros(M))
        self.k_topk = k_topk

    def forward(self, X):
        pre = X @ self.W_e.T + self.b_e
        vals, idx = torch.topk(pre.abs(), self.k_topk, dim=1)
        mask = torch.zeros_like(pre)
        mask.scatter_(1, idx, 1.0)
        Zhat = F.relu(pre) * mask
        Xhat = Zhat @ self.W_d.T + self.b_d
        return Xhat, Zhat

    def dictionary(self):
        with torch.no_grad():
            W = self.W_d.detach().clone()
            W = W / (W.norm(dim=0, keepdim=True) + 1e-8)
            return W.cpu().numpy()


def train_sae(G, k_data, k_sae, L, steps, batch_size, noise_std, lr, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    M = G.shape[0]
    sae = TopKSAE(M, L, k_sae).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for step in range(steps):
        X, _ = sample_batch(G, k_data, batch_size, noise_std, rng)
        Xhat, _ = sae(X)
        loss = F.mse_loss(Xhat, X)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return sae.dictionary()


def overlap_rate(D1, D2, tau):
    """Greedy bipartite matching by cosine similarity. Returns fraction
    of pairs with cosine >= tau."""
    L = D1.shape[1]
    sim = D1.T @ D2  # already normalized
    matched = 0
    used = set()
    pairs = []
    flat = [(sim[i, j], i, j) for i in range(L) for j in range(L)]
    flat.sort(reverse=True)
    used_i = set()
    used_j = set()
    for s, i, j in flat:
        if s < tau:
            break
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matched += 1
        if matched >= L:
            break
    return matched / L


def coherence(D):
    """Max off-diagonal absolute cosine of dictionary D (M x L)."""
    G = D.T @ D
    np.fill_diagonal(G, 0)
    return float(np.max(np.abs(G)))


def interference_spectral(D):
    """||W_d^T W_d - I||_2."""
    L = D.shape[1]
    G = D.T @ D - np.eye(L)
    return float(np.linalg.norm(G, ord=2))


def welch_bound(L, M):
    if L <= M:
        return 0.0
    return float(np.sqrt((L - M) / (M * (L - 1))))


def delta_mu(L, M, k):
    w = welch_bound(L, M)
    thr = 1.0 / (2 * k - 1)
    return max(0.0, w - thr)


# ----- Experiment 1: rho vs L (prediction 6.7.1) ----------------------
def sweep_L(M=16, N=128, k_data=4, k_sae=4, n_seeds=5, steps=1200,
            batch_size=128, noise_std=0.01, lr=1e-3, taus=(0.5, 0.7, 0.9)):
    G = make_ground_truth(N, M, seed=42)
    L_values = [N, 2 * N, 4 * N]
    rows = []
    for L in L_values:
        dicts = []
        t0 = time.time()
        for s in range(n_seeds):
            D = train_sae(G, k_data, k_sae, L, steps, batch_size,
                          noise_std, lr, seed=1000 + s)
            dicts.append(D)
        t1 = time.time()
        rhos = {tau: [] for tau in taus}
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                for tau in taus:
                    rhos[tau].append(overlap_rate(dicts[i], dicts[j], tau))
        mu_vals = [coherence(D) for D in dicts]
        i_vals = [interference_spectral(D) for D in dicts]
        for tau in taus:
            rows.append({
                "experiment": "L_sweep",
                "L": L, "M": M, "k": k_data, "tau": tau,
                "rho_mean": float(np.mean(rhos[tau])),
                "rho_std": float(np.std(rhos[tau])),
                "mu_mean": float(np.mean(mu_vals)),
                "I_norm_mean": float(np.mean(i_vals)),
                "welch_bound": welch_bound(L, M),
                "delta_mu_predicted": delta_mu(L, M, k_data),
                "n_seeds": n_seeds,
                "train_time_s": round(t1 - t0, 1),
            })
        print(f"L={L:>5}  rho@0.7={np.mean(rhos[0.7]):.3f}  "
              f"mu={np.mean(mu_vals):.3f}  ||I||={np.mean(i_vals):.2f}  "
              f"({(t1-t0):.1f}s)")
    return rows


# ----- Experiment 2: rho vs k (prediction 6.7.2) ----------------------
def sweep_k(M=16, N=128, L=256, n_seeds=5, steps=1200,
            batch_size=128, noise_std=0.01, lr=1e-3, tau=0.7):
    G = make_ground_truth(N, M, seed=42)
    k_values = [2, 4, 8]
    rows = []
    for k in k_values:
        dicts = []
        t0 = time.time()
        for s in range(n_seeds):
            D = train_sae(G, k, k, L, steps, batch_size,
                          noise_std, lr, seed=1000 + s)
            dicts.append(D)
        t1 = time.time()
        rho_list = []
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                rho_list.append(overlap_rate(dicts[i], dicts[j], tau))
        rows.append({
            "experiment": "k_sweep",
            "L": L, "M": M, "k": k, "tau": tau,
            "rho_mean": float(np.mean(rho_list)),
            "rho_std": float(np.std(rho_list)),
            "delta_mu_predicted": delta_mu(L, M, k),
            "n_seeds": n_seeds,
            "train_time_s": round(t1 - t0, 1),
        })
        print(f"k={k:>3}  rho@0.7={np.mean(rho_list):.3f}  "
              f"delta_mu={delta_mu(L, M, k):.3f}  ({(t1-t0):.1f}s)")
    return rows


# ----- Experiment 3: rho vs M (prediction 6.7.3) ----------------------
def sweep_M(N=128, L=256, k=4, n_seeds=5, steps=1200,
            batch_size=128, noise_std=0.01, lr=1e-3, tau=0.7):
    M_values = [8, 16, 32]
    rows = []
    for M in M_values:
        G = make_ground_truth(N, M, seed=42)
        dicts = []
        t0 = time.time()
        for s in range(n_seeds):
            D = train_sae(G, k, k, L, steps, batch_size,
                          noise_std, lr, seed=1000 + s)
            dicts.append(D)
        t1 = time.time()
        rho_list = []
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                rho_list.append(overlap_rate(dicts[i], dicts[j], tau))
        rows.append({
            "experiment": "M_sweep",
            "L": L, "M": M, "k": k, "tau": tau,
            "rho_mean": float(np.mean(rho_list)),
            "rho_std": float(np.std(rho_list)),
            "delta_mu_predicted": delta_mu(L, M, k),
            "n_seeds": n_seeds,
            "train_time_s": round(t1 - t0, 1),
        })
        print(f"M={M:>3}  rho@0.7={np.mean(rho_list):.3f}  "
              f"delta_mu={delta_mu(L, M, k):.3f}  ({(t1-t0):.1f}s)")
    return rows


def main():
    print("=" * 60)
    print("Toy-model verification of SAE-uniqueness scaling laws")
    print("=" * 60)
    all_rows = []
    print("\n[1/3] Sweep over L (prediction 6.7.1: rho should DECREASE)")
    all_rows.extend(sweep_L())
    print("\n[2/3] Sweep over k (prediction 6.7.2: rho should DECREASE)")
    all_rows.extend(sweep_k())
    print("\n[3/3] Sweep over M (prediction 6.7.3: rho should INCREASE)")
    all_rows.extend(sweep_M())
    out_path = RESULTS_DIR / "scaling_law_results.json"
    out_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
