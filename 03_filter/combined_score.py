"""
Combined FLD + PCA corpus selection.

FLD captures math rigor (intact/absent separation, mr-dominant).
PCA captures independent axes FLD underweights (ar, ei, sg).

combined = α × norm(fld) + (1-α) × norm(pca)

Usage:
  python 03_filter/combined_score.py [--n 1500] [--lam 0.5] [--alpha 0.7] [--out path]
"""

import argparse, json, pathlib, sys
import numpy as np
from numpy.linalg import pinv
from joblib import Parallel, delayed

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mmr_select import apply_rules, build_matrix, compute_fld, mmr_ensemble
from pca_score import compute_pca_scores

DIMS        = ["mr", "tn", "md", "ar", "er", "ei", "sg"]
RULES_FILE  = pathlib.Path("03_filter/rules/rules_jes.json")
DATA_FILE   = pathlib.Path("data/full/output.jsonl")
OUTPUT_FILE = pathlib.Path("03_filter/combined_corpus.jsonl")


def normalize(scores):
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",     type=int,   default=1500)
    parser.add_argument("--lam",   type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.7, help="FLD weight (1-alpha = PCA weight)")
    parser.add_argument("--runs",  type=int,   default=100)
    parser.add_argument("--jobs",  type=int,   default=-1)
    parser.add_argument("--out",   type=str,   default=str(OUTPUT_FILE))
    parser.add_argument("--rules", type=str,   default=str(RULES_FILE))
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

    print("Building feature matrices ...")
    X_all, means = build_matrix(all_papers, DIMS)
    X, _         = build_matrix(papers, DIMS, means)

    print("Computing FLD scores (from full population) ...")
    _, w, Sigma = compute_fld(X_all, all_papers, DIMS)
    fld_raw = X @ w
    cs_vals = np.array([
        float(p.get("cs", 7)) if isinstance(p.get("cs"), (int, float)) else 7.0
        for p in papers
    ])
    fld_raw = fld_raw * np.sqrt(cs_vals / 10.0)

    print("Computing PCA scores (from filtered subset) ...")
    pca_raw, var_explained, _, _ = compute_pca_scores(X)

    fld_norm = normalize(fld_raw)
    pca_norm = normalize(pca_raw)
    combined = args.alpha * fld_norm + (1 - args.alpha) * pca_norm

    print(f"\nα={args.alpha:.1f} → FLD×{args.alpha:.1f} + PCA×{1-args.alpha:.1f}")
    print(f"  FLD  norm: min={fld_norm.min():.2f}  mean={fld_norm.mean():.2f}  max={fld_norm.max():.2f}")
    print(f"  PCA  norm: min={pca_norm.min():.2f}  mean={pca_norm.mean():.2f}  max={pca_norm.max():.2f}")
    print(f"  combined:  min={combined.min():.2f}  mean={combined.mean():.2f}  max={combined.max():.2f}")

    Sigma_inv = pinv(Sigma[:len(DIMS), :len(DIMS)] if Sigma.shape[0] > len(DIMS) else Sigma)

    selected_idx = mmr_ensemble(X, combined, Sigma_inv, args.lam, args.n,
                                n_runs=args.runs, n_jobs=args.jobs)

    out_path = pathlib.Path(args.out)
    print(f"\nWriting {len(selected_idx)} papers to {out_path} ...")
    with open(out_path, "w") as f:
        for rank, idx in enumerate(selected_idx):
            p = papers[idx]
            rec = {
                "rank":          rank + 1,
                "paper_id":      p.get("id") or p.get("paper_id", ""),
                "title":         p.get("_title", ""),
                "venue":         p.get("_venue", ""),
                "year":          p.get("_year"),
                "ig":            p.get("ig"),
                "combined_score": round(float(combined[idx]), 4),
                "fld_score":     round(float(fld_norm[idx]), 4),
                "pca_score":     round(float(pca_norm[idx]), 4),
                "mr": p.get("mr"), "tn": p.get("tn"), "md": p.get("md"),
                "ar": p.get("ar"), "er": p.get("er"), "ei": p.get("ei"),
                "sg": p.get("sg"), "cs": p.get("cs"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Done.")
    print(f"\nTop 10:")
    print(f"{'rank':>5} {'ig':>8} {'comb':>5} {'fld':>5} {'pca':>5} "
          f"{'mr':>4} {'tn':>4} {'ar':>4} {'ei':>4} {'er':>4}  title[:50]")
    for rank, idx in enumerate(selected_idx[:10]):
        p = papers[idx]
        print(f"{rank+1:>5} {p.get('ig',''):>8} {combined[idx]:>5.3f} "
              f"{fld_norm[idx]:>5.3f} {pca_norm[idx]:>5.3f} "
              f"{p.get('mr',0):>4.1f} {p.get('tn',0):>4.1f} "
              f"{p.get('ar',0):>4.1f} {p.get('ei',0):>4.1f} {p.get('er',0):>4.1f}  "
              f"{p.get('_title','')[:48]}")

    # overlap stats
    fld_ids = {json.loads(l)['paper_id'] for l in open('03_filter/mmr_corpus.jsonl')}
    pca_ids = {json.loads(l)['paper_id'] for l in open('03_filter/pca_corpus.jsonl')}
    comb_ids = {papers[i].get('id') or papers[i].get('paper_id','') for i in selected_idx}
    print(f"\n与FLD语料库重叠: {len(comb_ids & fld_ids)}/1500 ({len(comb_ids & fld_ids)/15:.1f}%)")
    print(f"与PCA语料库重叠: {len(comb_ids & pca_ids)}/1500 ({len(comb_ids & pca_ids)/15:.1f}%)")


if __name__ == "__main__":
    main()
