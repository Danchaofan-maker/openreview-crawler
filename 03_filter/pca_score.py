"""
PCA-based corpus selection.

Problem with FLD: defined by intact/absent centroids → dominated by mr (weight 0.70),
giving mr/tn/md triple weight despite them being 0.75–0.81 correlated.

PCA on filtered subset shows:
  PC1 (57.6%): math rigor axis — mr/tn/md/sg/ei together (redundant trio)
  PC2 (13.4%): ar alone (fully independent)
  PC3 (10.7%): ei alone (independent)
  PC4 ( 6.7%): sg alone

Score = variance-weighted sum of PCs, sign-corrected so "good paper" = high score:
  -PC1 × √var1  (negate: high PC1 = high er, low mr = bad paper)
  +PC2 × √var2  (ar: realistic assumptions, independent signal)
  +PC3 × √var3  (ei: theoretical intent, independent signal)
  +PC4 × √var4  (sg: scope generality)

Then MMR ensemble for diversity, same as mmr_select.py.

Usage:
  python 03_filter/pca_score.py [--n 1500] [--lam 0.5] [--out path]
"""

import argparse, json, pathlib, sys
import numpy as np
from numpy.linalg import pinv
from joblib import Parallel, delayed

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mmr_select import apply_rules, build_matrix, mmr_ensemble

DIMS        = ["mr", "tn", "md", "ar", "er", "ei", "sg"]
RULES_FILE  = pathlib.Path("03_filter/rules/rules_jes.json")
DATA_FILE   = pathlib.Path("data/full/output.jsonl")
OUTPUT_FILE = pathlib.Path("03_filter/pca_corpus.jsonl")

N_PCS = 4   # PC1–PC4 cover 88.4% of variance


def compute_pca_scores(X):
    """
    Standardize X, run SVD, return PC scores and variance explained.
    Sign-correct so that high score = good paper (math-rigorous, low-er).
    """
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    _, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    var_explained = S**2 / (S**2).sum()

    pc_scores = X_std @ Vt.T   # (N, D)

    # Sign-correct each PC based on its dominant dimension
    # PC1: mr/tn/md have negative loadings → high PC1 = bad → negate
    # PC2: ar has positive loading (+0.90) → high ar = good → keep
    # PC3: ei has positive loading (+0.82) → high ei = good → keep
    # PC4: sg has positive loading (+0.76) → high sg = good → keep
    signs = []
    for pc_idx in range(N_PCS):
        loadings = Vt[pc_idx]
        # dominant dim = highest abs loading
        dom = int(np.argmax(np.abs(loadings)))
        # "good direction": mr/tn/md/ei/sg should be positive, er negative
        dim_name = DIMS[dom]
        natural_positive = dim_name not in ("er",)
        sign = +1 if (loadings[dom] > 0) == natural_positive else -1
        signs.append(sign)

    quality = np.zeros(len(X))
    for i in range(N_PCS):
        quality += signs[i] * pc_scores[:, i] * np.sqrt(var_explained[i])

    return quality, var_explained, Vt, signs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int,   default=1500)
    parser.add_argument("--lam",  type=float, default=0.5)
    parser.add_argument("--runs", type=int,   default=100)
    parser.add_argument("--jobs", type=int,   default=-1)
    parser.add_argument("--out",  type=str,   default=str(OUTPUT_FILE))
    parser.add_argument("--rules",type=str,   default=str(RULES_FILE))
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

    print("Building feature matrix ...")
    X, _ = build_matrix(papers, DIMS)

    print("Computing PCA scores ...")
    pca_scores, var_explained, Vt, signs = compute_pca_scores(X)

    print("\nPCA decomposition (filtered subset):")
    print(f"  {'PC':>4}  {'var%':>6}  {'cumul%':>7}  {'sign':>5}  dominant dim")
    cumul = 0.0
    for i in range(N_PCS):
        cumul += var_explained[i]
        dom = int(np.argmax(np.abs(Vt[i])))
        print(f"  PC{i+1:>2}  {var_explained[i]*100:>5.1f}%  {cumul*100:>6.1f}%"
              f"  {'neg' if signs[i]==-1 else 'pos':>5}  {DIMS[dom]}")

    print("\nLoadings matrix:")
    print("      " + "".join(f"  PC{i+1}" for i in range(N_PCS)))
    for j, d in enumerate(DIMS):
        row = "".join(f"  {Vt[i,j]:>+5.2f}" for i in range(N_PCS))
        print(f"  {d:>4}{row}")

    print(f"\nPCA score stats:  min={pca_scores.min():.2f}  "
          f"mean={pca_scores.mean():.2f}  max={pca_scores.max():.2f}")

    # Covariance matrix of filtered papers for Mahalanobis similarity in MMR
    Sigma     = np.cov(X.T)
    Sigma_inv = pinv(Sigma)

    selected_idx = mmr_ensemble(X, pca_scores, Sigma_inv, args.lam, args.n,
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
                "pca_score":  round(float(pca_scores[idx]), 4),
                "mr": p.get("mr"), "tn": p.get("tn"), "md": p.get("md"),
                "ar": p.get("ar"), "er": p.get("er"), "ei": p.get("ei"),
                "sg": p.get("sg"), "cs": p.get("cs"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"\nTop 10:")
    print(f"{'rank':>5} {'ig':>8} {'pca':>6} {'mr':>4} {'tn':>4} {'ar':>4} {'ei':>4} {'er':>4}  title[:55]")
    for rank, idx in enumerate(selected_idx[:10]):
        p = papers[idx]
        print(f"{rank+1:>5} {p.get('ig',''):>8} {pca_scores[idx]:>6.2f} "
              f"{p.get('mr',0):>4.1f} {p.get('tn',0):>4.1f} "
              f"{p.get('ar',0):>4.1f} {p.get('ei',0):>4.1f} {p.get('er',0):>4.1f}  "
              f"{p.get('_title','')[:53]}")


if __name__ == "__main__":
    main()
