"""
MMR-based reading corpus selection.

Pipeline:
  1. Load rules_jes.json, apply to filter ~31k → ~6k papers
  2. Compute FLD weights from intact/absent centroids + covariance matrix
  3. Run MMR: iteratively select papers maximizing
       λ * fld(d) - (1-λ) * max_sim(d, already_selected)
     where sim uses Mahalanobis distance in feature space
  4. Write selected papers to OUTPUT_FILE (jsonl)

Usage:
  python 03_filter/mmr_select.py [--n 1500] [--lam 0.5] [--out path]
"""

import argparse, json, pathlib
import numpy as np
from numpy.linalg import pinv
from joblib import Parallel, delayed

# ── config ───────────────────────────────────────────────────────────────────
DIMS       = ["mr", "tn", "md", "ar", "er", "ei", "sg"]
RULES_FILE = pathlib.Path("03_filter/rules/rules_jes.json")
DATA_FILE  = pathlib.Path("data/full/output.jsonl")
OUTPUT_FILE = pathlib.Path("03_filter/mmr_corpus.jsonl")

# ── rules engine ─────────────────────────────────────────────────────────────
def _get_val(p, field):
    if field in ("mk_f", "hr_f"): return p.get(field)
    if field == "marketing": return p.get("mk_f")
    if field == "human_review": return p.get("hr_f")
    if field == "integrity": return p.get("ig") or p.get("integrity")
    v = p.get(field)
    if isinstance(v, (int, float)): return v
    if isinstance(v, dict): return v.get("s") or v.get("score")
    return None

def _eval_cond(p, c):
    v = _get_val(p, c["field"])
    if v is None: return None
    op, th = c["op"], c["value"]
    if op == "lt":  return v < th
    if op == "lte": return v <= th
    if op == "gt":  return v > th
    if op == "gte": return v >= th
    if op == "eq":  return v == th
    if op == "neq": return v != th
    if op == "in":  return v in (th if isinstance(th, list) else [th])
    return None

def _eval_rule(p, rule):
    results = [_eval_cond(p, c) for c in rule.get("conditions", [])]
    logic = rule.get("internal_logic", "AND")
    if logic == "AND":
        # None = field missing = condition not satisfied = breaks AND
        resolved = [False if r is None else r for r in results]
    else:
        # OR: None = field missing = doesn't contribute
        resolved = [r for r in results if r is not None]
    if not resolved: return False
    hit = all(resolved) if logic == "AND" else any(resolved)
    return (not hit) if rule.get("negate") else hit

def apply_rules(papers, cfg):
    excl_rules   = cfg.get("rules", [])
    rescue_rules = cfg.get("rescue_rules", [])
    kept = []
    for p in papers:
        excluded = any(_eval_rule(p, r) for r in excl_rules)
        if cfg.get("force_keep_hr") and p.get("hr_f"):
            excluded = False
        if excluded:
            for r in rescue_rules:
                if _eval_rule(p, r):
                    excluded = False
                    break
        if not excluded:
            kept.append(p)
    return kept

# ── feature extraction ────────────────────────────────────────────────────────
def build_matrix(papers, dims, means=None):
    """Return (N x D) matrix; impute missing values with column means."""
    raw = []
    for p in papers:
        row = []
        for d in dims:
            v = p.get(d)
            if isinstance(v, (int, float)):
                row.append(float(v))
            else:
                row.append(np.nan)
        raw.append(row)
    X = np.array(raw)
    if means is None:
        means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = means[j]
    return X, means

# ── FLD score ─────────────────────────────────────────────────────────────────
def compute_fld(X, papers, dims):
    """
    Fisher Linear Discriminant weights from full population.
    w = Σ⁻¹ × (μ_intact - μ_absent)
    Should be called on ALL papers (not filtered) for unbiased weights.
    Returns raw score array (no cs), weight vector w, and Sigma.
    """
    intact_mask = np.array([p.get("ig") == "intact" for p in papers])
    absent_mask = np.array([p.get("ig") == "absent"  for p in papers])

    mu_intact = X[intact_mask].mean(axis=0)
    mu_absent = X[absent_mask].mean(axis=0)

    Sigma = np.cov(X.T)
    w = pinv(Sigma) @ (mu_intact - mu_absent)
    scores = X @ w
    return scores, w, Sigma

# ── MMR (single run) ──────────────────────────────────────────────────────────
def mmr_select(X, fld_scores, Sigma_inv, lam, target_n, verbose=False):
    """
    Single greedy MMR run.
    relevance  = normalized FLD score
    similarity = exp(-mahala_dist²/2) in Mahalanobis space (RBF kernel)
    """
    n = len(fld_scores)
    lo, hi = fld_scores.min(), fld_scores.max()
    rel = (fld_scores - lo) / (hi - lo + 1e-9)

    selected  = []
    remaining = list(range(n))
    max_sim   = np.zeros(n)

    for step in range(target_n):
        scores   = lam * rel[remaining] - (1 - lam) * max_sim[remaining]
        best_loc = int(np.argmax(scores))
        best_idx = remaining[best_loc]
        selected.append(best_idx)
        remaining.pop(best_loc)

        if verbose and (step + 1) % 100 == 0:
            print(f"  selected {step+1}/{target_n} ...")

        xb = X[best_idx]
        for idx in remaining:
            diff      = X[idx] - xb
            mahala_sq = float(diff @ Sigma_inv @ diff)
            sim       = np.exp(-mahala_sq / 2.0)
            if sim > max_sim[idx]:
                max_sim[idx] = sim

    return selected


# ── Monte Carlo ensemble ───────────────────────────────────────────────────────
def _single_run(seed, X, fld_scores, Sigma_inv, lam, target_n, noise_scale):
    rng = np.random.default_rng(seed)
    noisy = fld_scores + rng.standard_normal(len(fld_scores)) * noise_scale
    return mmr_select(X, noisy, Sigma_inv, lam, target_n)


def mmr_ensemble(X, fld_scores, Sigma_inv, lam, target_n, n_runs=100, n_jobs=-1):
    """
    Monte Carlo MMR: average over N noisy runs to reduce path-dependence.
    Each run adds Gaussian noise (σ = fld.std() * 0.1) to FLD scores.
    Final ranking = argsort of mean rank across runs (Borda count).
    """
    noise_scale = fld_scores.std() * 0.1
    n = len(fld_scores)

    print(f"Running {n_runs} MMR ensemble runs (n_jobs={n_jobs}) ...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_run)(seed, X, fld_scores, Sigma_inv, lam, target_n, noise_scale)
        for seed in range(n_runs)
    )

    # accumulate ranks: unselected papers get rank target_n + 1
    rank_sum = np.full(n, (target_n + 1) * n_runs, dtype=np.float64)
    for run_idx, selected in enumerate(results):
        for rank, paper_idx in enumerate(selected):
            rank_sum[paper_idx] -= (target_n + 1 - (rank + 1))

    mean_rank = rank_sum / n_runs
    return list(np.argsort(mean_rank)[:target_n])

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int,   default=1500, help="target corpus size")
    parser.add_argument("--lam",    type=float, default=0.5,  help="lambda (relevance weight)")
    parser.add_argument("--runs",   type=int,   default=100,  help="Monte Carlo ensemble runs")
    parser.add_argument("--jobs",   type=int,   default=-1,   help="parallel jobs (-1=all cores)")
    parser.add_argument("--out",    type=str,   default=str(OUTPUT_FILE))
    parser.add_argument("--rules",  type=str,   default=str(RULES_FILE))
    args = parser.parse_args()

    print("Loading papers ...")
    all_papers = []
    with open(DATA_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r.get("parsed"):
                p = r["parsed"]
                p["_title"] = r.get("title", "")
                p["_venue"] = r.get("venue", "")
                p["_year"]  = r.get("year")
                all_papers.append(p)

    cfg = json.loads(pathlib.Path(args.rules).read_text())
    print("Applying rules ...")
    papers = apply_rules(all_papers, cfg)
    print(f"  {len(all_papers)} → {len(papers)} papers after rules")

    print("Building feature matrix (all papers for FLD) ...")
    X_all, means = build_matrix(all_papers, DIMS)
    _, w, Sigma = compute_fld(X_all, all_papers, DIMS)
    Sigma_inv = pinv(Sigma)

    print("Scoring filtered papers ...")
    X, _ = build_matrix(papers, DIMS, means)
    cs_vals = np.array([
        float(p.get("cs", 7)) if isinstance(p.get("cs"), (int, float)) else 7.0
        for p in papers
    ])
    fld_scores = (X @ w) * np.sqrt(cs_vals / 10.0)

    print("\nFLD weights:")
    for d, wi in zip(DIMS, w):
        print(f"  {d:>4}: {wi:+.4f}")

    selected_idx = mmr_ensemble(X, fld_scores, Sigma_inv, args.lam, args.n,
                                n_runs=args.runs, n_jobs=args.jobs)

    out_path = pathlib.Path(args.out)
    print(f"\nWriting {len(selected_idx)} papers to {out_path} ...")
    with open(out_path, "w") as f:
        for rank, idx in enumerate(selected_idx):
            p = papers[idx]
            rec = {
                "rank":       rank + 1,
                "paper_id":   p.get("id") or p.get("paper_id", ""),
                "title":      p.get("_title", ""),
                "venue":      p.get("_venue", ""),
                "year":       p.get("_year"),
                "ig":         p.get("ig"),
                "fld_score":  round(float(fld_scores[idx]), 4),
                "mr": p.get("mr"), "tn": p.get("tn"), "md": p.get("md"),
                "ar": p.get("ar"), "er": p.get("er"), "ei": p.get("ei"),
                "sg": p.get("sg"), "cs": p.get("cs"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"\nTop 10 by selection order:")
    print(f"{'rank':>5} {'ig':>8} {'fld':>6} {'mr':>4} {'tn':>4} {'er':>4}  title[:60]")
    for rank, idx in enumerate(selected_idx[:10]):
        p = papers[idx]
        print(f"{rank+1:>5} {p.get('ig',''):>8} {fld_scores[idx]:>6.2f} "
              f"{p.get('mr',0):>4.1f} {p.get('tn',0):>4.1f} {p.get('er',0):>4.1f}  "
              f"{p.get('_title','')[:58]}")

if __name__ == "__main__":
    main()
