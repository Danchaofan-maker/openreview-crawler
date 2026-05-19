"""
Explain the Jes final corpus with interpretable surrogate rules.

This script treats a final corpus JSONL as labels, then:
  1. builds raw score features plus Jes-style derived features
     (FLD, PCA, combined score, unsupervised density proxies),
  2. estimates how predictable membership is with tree models, and
  3. distills high-precision tree leaves into a small DNF-style rule set.

The resulting rules are explanatory surrogates, not replacement selection logic:
MMR is set-dependent, so no per-paper rule set can be exactly equivalent.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mmr_select import apply_rules, build_matrix, compute_fld  # noqa: E402


DIMS = ["mr", "tn", "md", "ar", "er", "ei", "sg"]
RAW_NUMERIC = ["mr", "tn", "md", "ar", "er", "tea", "cc", "ei", "sg", "cs"]


@dataclass
class RuleCandidate:
    name: str
    conditions: list[str]
    mask: np.ndarray
    tp: int
    fp: int
    precision: float
    recall: float


def load_papers(data_path: pathlib.Path) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    with data_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r.get("parsed"):
                p = dict(r["parsed"])
                p["_id"] = p.get("id") or p.get("paper_id")
                p["_title"] = r.get("title", "")
                p["_venue"] = r.get("venue", "")
                p["_year"] = r.get("year")
                papers.append(p)
    return papers


def load_ids(path: pathlib.Path) -> set[str]:
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.add(rec["paper_id"])
    return ids


def normalize_by_reference(values: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    ref = values[reference_mask]
    lo, hi = np.nanmin(ref), np.nanmax(ref)
    return (values - lo) / (hi - lo + 1e-9)


def compute_pca_quality(
    x_pref: np.ndarray, x_all: np.ndarray, n_pcs: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Fit the PCA score used in pca_score.py on the prefiltered subset, transform all papers."""
    mean = x_pref.mean(axis=0)
    std = x_pref.std(axis=0) + 1e-9
    pref_std = (x_pref - mean) / std
    all_std = (x_all - mean) / std

    _, s, vt = np.linalg.svd(pref_std, full_matrices=False)
    var_explained = s**2 / (s**2).sum()
    pc_scores = all_std @ vt.T

    signs: list[int] = []
    for pc_idx in range(n_pcs):
        loadings = vt[pc_idx]
        dom = int(np.argmax(np.abs(loadings)))
        dim_name = DIMS[dom]
        natural_positive = dim_name not in ("er",)
        sign = +1 if (loadings[dom] > 0) == natural_positive else -1
        signs.append(sign)

    quality = np.zeros(len(x_all))
    for i in range(n_pcs):
        quality += signs[i] * pc_scores[:, i] * math.sqrt(float(var_explained[i]))
    return quality, var_explained, vt, signs


def build_features(
    papers: list[dict[str, Any]],
    target_ids: set[str],
    rules_path: pathlib.Path,
    alpha: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    cfg = json.loads(rules_path.read_text())
    prefilter = apply_rules(papers, cfg)
    prefilter_ids = {p["_id"] for p in prefilter}
    prefilter_mask = np.array([p["_id"] in prefilter_ids for p in papers], dtype=bool)

    x_all_dims, means = build_matrix(papers, DIMS)
    x_pref_dims, _ = build_matrix(prefilter, DIMS, means)

    _, w, _ = compute_fld(x_all_dims, papers, DIMS)
    fld_raw = x_all_dims @ w
    cs_vals = np.array(
        [float(p.get("cs", 7)) if isinstance(p.get("cs"), (int, float)) else 7.0 for p in papers]
    )
    fld_raw = fld_raw * np.sqrt(cs_vals / 10.0)
    fld_norm = normalize_by_reference(fld_raw, prefilter_mask)

    pca_raw, var_explained, vt, signs = compute_pca_quality(x_pref_dims, x_all_dims)
    pca_norm = normalize_by_reference(pca_raw, prefilter_mask)
    combined = alpha * fld_norm + (1 - alpha) * pca_norm

    # Unsupervised density/diversity proxies. These approximate the information MMR uses.
    x_pref_std = (x_pref_dims - x_pref_dims.mean(axis=0)) / (x_pref_dims.std(axis=0) + 1e-9)
    x_all_std = (x_all_dims - x_pref_dims.mean(axis=0)) / (x_pref_dims.std(axis=0) + 1e-9)
    km = MiniBatchKMeans(n_clusters=32, random_state=random_state, batch_size=2048, n_init=10)
    km.fit(x_pref_std)
    cluster = km.predict(x_all_std)
    center_dist = np.linalg.norm(x_all_std - km.cluster_centers_[cluster], axis=1)
    pref_cluster = km.predict(x_pref_std)
    sizes = np.bincount(pref_cluster, minlength=32).astype(float)
    cluster_log_size = np.log1p(sizes[cluster])

    cols: list[np.ndarray] = []
    names: list[str] = []

    for f in RAW_NUMERIC:
        vals = []
        missing = []
        for p in papers:
            v = p.get(f)
            ok = isinstance(v, (int, float)) and v < 20
            vals.append(float(v) if ok else -1.0)
            missing.append(0.0 if ok else 1.0)
        cols.append(np.array(vals, dtype=float))
        names.append(f)
        if f == "tea":
            cols.append(np.array(missing, dtype=float))
            names.append("tea_missing")

    igs = np.array([p.get("ig") for p in papers], dtype=object)
    for ig in ["intact", "partial", "broken", "absent"]:
        cols.append((igs == ig).astype(float))
        names.append(f"ig_{ig}")

    cols.append(np.array([float(bool(p.get("mk_f"))) for p in papers]))
    names.append("marketing")
    cols.append(np.array([float(bool(p.get("hr_f"))) for p in papers]))
    names.append("human_review")
    cols.append(prefilter_mask.astype(float))
    names.append("jes_prefilter")

    for name, arr in [
        ("fld_norm", fld_norm),
        ("pca_norm", pca_norm),
        ("combined_score", combined),
        ("cluster_center_dist", center_dist),
        ("cluster_log_size", cluster_log_size),
    ]:
        cols.append(np.asarray(arr, dtype=float))
        names.append(name)

    x = np.column_stack(cols)
    y = np.array([p["_id"] in target_ids for p in papers], dtype=bool)
    meta = {
        "prefilter_count": int(prefilter_mask.sum()),
        "alpha": alpha,
        "pca_var_explained": [float(v) for v in var_explained[:4]],
        "pca_signs": signs,
        "pca_loadings": vt[:4].tolist(),
    }
    return x, y, names, meta


def top_k_metrics(scores: np.ndarray, y: np.ndarray, k: int) -> dict[str, float]:
    order = np.argsort(scores)[::-1]
    pred = np.zeros(len(y), dtype=bool)
    pred[order[:k]] = True
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "k": float(k),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_models(x: np.ndarray, y: np.ndarray, names: list[str], random_state: int) -> list[dict[str, Any]]:
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.30, random_state=random_state, stratify=y
    )
    models = [
        (
            "random_forest_raw",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            [i for i, n in enumerate(names) if n not in {"fld_norm", "pca_norm", "combined_score", "cluster_center_dist", "cluster_log_size"}],
        ),
        (
            "extra_trees_raw",
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            [i for i, n in enumerate(names) if n not in {"fld_norm", "pca_norm", "combined_score", "cluster_center_dist", "cluster_log_size"}],
        ),
        (
            "random_forest_derived",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=11,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            list(range(len(names))),
        ),
        (
            "extra_trees_derived",
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=11,
                min_samples_leaf=20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=random_state,
            ),
            list(range(len(names))),
        ),
    ]

    results: list[dict[str, Any]] = []
    for model_name, model, cols in models:
        model.fit(x[train_idx][:, cols], y[train_idx])
        test_scores = model.predict_proba(x[test_idx][:, cols])[:, 1]
        train_scores = model.predict_proba(x[train_idx][:, cols])[:, 1]
        test_k = int(y[test_idx].sum())
        train_k = int(y[train_idx].sum())
        test_top = top_k_metrics(test_scores, y[test_idx], test_k)
        train_top = top_k_metrics(train_scores, y[train_idx], train_k)
        pred_default = test_scores >= 0.5
        p, r, f1, _ = precision_recall_fscore_support(
            y[test_idx], pred_default, average="binary", zero_division=0
        )
        results.append(
            {
                "model": model_name,
                "features": len(cols),
                "train_top_k": train_top,
                "test_top_k": test_top,
                "test_default_precision": float(p),
                "test_default_recall": float(r),
                "test_default_f1": float(f1),
                "test_roc_auc": float(roc_auc_score(y[test_idx], test_scores)),
                "test_avg_precision": float(average_precision_score(y[test_idx], test_scores)),
            }
        )
    return results


def condition_text(name: str, op: str, value: float) -> str:
    if name.startswith("ig_"):
        label = name.removeprefix("ig_")
        return f"integrity {'!=' if op == '<=' else '='} {label}"
    if name in {"marketing", "human_review", "tea_missing", "jes_prefilter"}:
        val = "false" if op == "<=" else "true"
        return f"{name} = {val}"
    return f"{name} {op} {value:.3g}"


def simplify_conditions(conditions: list[str]) -> list[str]:
    """Collapse repeated tree-path bounds into a smaller human-readable conjunction."""
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    equality: dict[str, str] = {}
    not_values: dict[str, set[str]] = {}
    passthrough: list[str] = []

    for cond in conditions:
        m = re.match(r"^([A-Za-z0-9_]+) (<=|>) (-?\d+(?:\.\d+)?)$", cond)
        if m:
            feat, op, raw = m.group(1), m.group(2), float(m.group(3))
            if op == ">":
                lower[feat] = max(lower.get(feat, -float("inf")), raw)
            else:
                upper[feat] = min(upper.get(feat, float("inf")), raw)
            continue

        m = re.match(r"^(marketing|human_review|tea_missing|jes_prefilter) = (true|false)$", cond)
        if m:
            equality[m.group(1)] = m.group(2)
            continue

        m = re.match(r"^integrity = ([A-Za-z_]+)$", cond)
        if m:
            equality["integrity"] = m.group(1)
            continue

        m = re.match(r"^integrity != ([A-Za-z_]+)$", cond)
        if m:
            not_values.setdefault("integrity", set()).add(m.group(1))
            continue

        passthrough.append(cond)

    out: list[str] = []
    if "integrity" in equality:
        out.append(f"integrity = {equality.pop('integrity')}")
    elif "integrity" in not_values:
        excluded = sorted(not_values["integrity"])
        remaining = [v for v in ["intact", "partial", "broken", "absent"] if v not in excluded]
        if remaining and len(remaining) <= 2:
            out.append(f"integrity in {remaining}")
        else:
            out.extend(f"integrity != {v}" for v in excluded)

    for feat in sorted(equality):
        out.append(f"{feat} = {equality[feat]}")

    for feat in sorted(set(lower) | set(upper)):
        lo = lower.get(feat)
        hi = upper.get(feat)
        if lo is not None and hi is not None:
            if lo >= hi:
                out.append(f"{feat} > {lo:.3g} AND {feat} <= {hi:.3g}  [contradictory]")
            else:
                out.append(f"{lo:.3g} < {feat} <= {hi:.3g}")
        elif lo is not None:
            out.append(f"{feat} > {lo:.3g}")
        elif hi is not None:
            out.append(f"{feat} <= {hi:.3g}")

    out.extend(passthrough)
    return out


def extract_rule_candidates(
    model: RandomForestClassifier | ExtraTreesClassifier,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    min_support: int,
    min_tp: int,
    min_precision: float,
    max_candidates: int,
) -> list[RuleCandidate]:
    candidates: list[RuleCandidate] = []
    n = len(y)

    for est_idx, est in enumerate(model.estimators_):
        tree = est.tree_
        leaf_for_row = est.apply(x)

        def walk(node: int, path: list[str]) -> None:
            left, right = tree.children_left[node], tree.children_right[node]
            if left == right:
                mask = leaf_for_row == node
                support = int(mask.sum())
                if support < min_support:
                    return
                tp = int((mask & y).sum())
                fp = support - tp
                if tp < min_tp:
                    return
                precision = tp / support
                recall = tp / int(y.sum())
                if precision < min_precision:
                    return
                candidates.append(
                    RuleCandidate(
                        name=f"tree_{est_idx:03d}_leaf_{node}",
                        conditions=simplify_conditions(path),
                        mask=mask.copy(),
                        tp=tp,
                        fp=fp,
                        precision=precision,
                        recall=recall,
                    )
                )
                return

            feat = int(tree.feature[node])
            th = float(tree.threshold[node])
            name = feature_names[feat]
            walk(left, path + [condition_text(name, "<=", th)])
            walk(right, path + [condition_text(name, ">", th)])

        walk(0, [])

    def quality(c: RuleCandidate) -> float:
        f1 = 2 * c.precision * c.recall / (c.precision + c.recall) if c.precision + c.recall else 0.0
        return f1 + 0.05 * c.precision

    candidates.sort(key=quality, reverse=True)
    # De-duplicate exact condition lists.
    deduped: list[RuleCandidate] = []
    seen: set[tuple[str, ...]] = set()
    for c in candidates:
        key = tuple(c.conditions)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
        if len(deduped) >= max_candidates:
            break
    return deduped


def union_metrics(mask: np.ndarray, y: np.ndarray) -> dict[str, float]:
    tp = int((mask & y).sum())
    fp = int((mask & ~y).sum())
    fn = int((~mask & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "kept": float(int(mask.sum())),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def greedy_select_rules(
    candidates: list[RuleCandidate],
    y: np.ndarray,
    target_size: int,
    max_rules: int,
) -> tuple[list[RuleCandidate], np.ndarray, dict[str, float]]:
    best_overall: tuple[float, list[RuleCandidate], np.ndarray, dict[str, float]] | None = None
    penalties = [0.0, 0.02, 0.04, 0.06, 0.09, 0.12]

    for size_penalty in penalties:
        selected: list[RuleCandidate] = []
        union = np.zeros(len(y), dtype=bool)
        best_score = -1e9

        for _ in range(max_rules):
            best_step: tuple[float, RuleCandidate, np.ndarray, dict[str, float]] | None = None
            for cand in candidates:
                if cand in selected:
                    continue
                new_union = union | cand.mask
                m = union_metrics(new_union, y)
                score = (
                    m["f1"]
                    - size_penalty * abs(m["kept"] - target_size) / target_size
                    - 0.002 * (len(selected) + 1)
                )
                if best_step is None or score > best_step[0]:
                    best_step = (score, cand, new_union, m)
            if best_step is None or best_step[0] <= best_score + 1e-5:
                break
            best_score = best_step[0]
            selected.append(best_step[1])
            union = best_step[2]

        # Backward pruning if it improves the same objective.
        changed = True
        while changed and len(selected) > 1:
            changed = False
            current_m = union_metrics(union, y)
            current_score = (
                current_m["f1"]
                - size_penalty * abs(current_m["kept"] - target_size) / target_size
                - 0.002 * len(selected)
            )
            remove_best = None
            for idx, _ in enumerate(selected):
                trial = [c for j, c in enumerate(selected) if j != idx]
                trial_union = np.zeros(len(y), dtype=bool)
                for c in trial:
                    trial_union |= c.mask
                trial_m = union_metrics(trial_union, y)
                trial_score = (
                    trial_m["f1"]
                    - size_penalty * abs(trial_m["kept"] - target_size) / target_size
                    - 0.002 * len(trial)
                )
                if trial_score > current_score + 1e-5:
                    remove_best = (idx, trial, trial_union, trial_m, trial_score)
                    current_score = trial_score
            if remove_best is not None:
                selected = remove_best[1]
                union = remove_best[2]
                changed = True

        m = union_metrics(union, y)
        final_score = m["f1"] - 0.04 * abs(m["kept"] - target_size) / target_size - 0.002 * len(selected)
        if best_overall is None or final_score > best_overall[0]:
            best_overall = (final_score, selected, union, m)

    assert best_overall is not None
    return best_overall[1], best_overall[2], best_overall[3]


def render_report(
    out_path: pathlib.Path,
    final_path: pathlib.Path,
    rules_path: pathlib.Path,
    meta: dict[str, Any],
    model_results: list[dict[str, Any]],
    selected: list[RuleCandidate],
    selected_mask: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model: RandomForestClassifier | ExtraTreesClassifier,
    x: np.ndarray,
) -> None:
    lines: list[str] = []
    lines.append("# Jes Final Surrogate Explanation")
    lines.append("")
    lines.append(f"- Target corpus: `{final_path}`")
    lines.append(f"- Prefilter rules: `{rules_path}`")
    lines.append(f"- Target positives: `{int(y.sum())}`")
    lines.append(f"- Jes prefilter size: `{meta['prefilter_count']}`")
    lines.append("")
    lines.append("## Predictability Upper Bound")
    lines.append("")
    lines.append("| model | feature set | test top-k P | test top-k R | test top-k F1 | ROC-AUC | AP |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in model_results:
        lines.append(
            "| {model} | {features} | {p:.3f} | {rec:.3f} | {f1:.3f} | {auc:.3f} | {ap:.3f} |".format(
                model=r["model"],
                features=r["features"],
                p=r["test_top_k"]["precision"],
                rec=r["test_top_k"]["recall"],
                f1=r["test_top_k"]["f1"],
                auc=r["test_roc_auc"],
                ap=r["test_avg_precision"],
            )
        )
    lines.append("")
    lines.append("Top-k means the classifier is forced to select the same number of papers as the split contains positives.")
    lines.append("")

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        order = np.argsort(importances)[::-1][:15]
        lines.append("## Feature Importance")
        lines.append("")
        lines.append("| feature | importance |")
        lines.append("|---|---:|")
        for i in order:
            lines.append(f"| {feature_names[i]} | {importances[i]:.4f} |")
        lines.append("")

    lines.append("## Distilled Rule Set")
    lines.append("")
    m = union_metrics(selected_mask, y)
    lines.append(
        "Selected surrogate union: "
        f"`kept={int(m['kept'])}`, `hit={int(m['tp'])}`, "
        f"`precision={m['precision']:.3f}`, `recall={m['recall']:.3f}`, `F1={m['f1']:.3f}`."
    )
    lines.append("")
    lines.append("These rules explain membership tendencies. They are not equivalent to MMR because MMR is set-dependent.")
    lines.append("")

    running = np.zeros(len(y), dtype=bool)
    for idx, cand in enumerate(selected, 1):
        marginal = cand.mask & ~running
        running |= cand.mask
        marg_m = union_metrics(marginal, y)
        lines.append(f"### Rule {idx}: {cand.name}")
        lines.append("")
        lines.append(
            f"- Own: kept={int(cand.mask.sum())}, hit={cand.tp}, "
            f"precision={cand.precision:.3f}, recall={cand.recall:.3f}"
        )
        lines.append(
            f"- Marginal: kept={int(marg_m['kept'])}, hit={int(marg_m['tp'])}, "
            f"precision={marg_m['precision']:.3f}"
        )
        lines.append("- Conditions:")
        for cond in cand.conditions:
            lines.append(f"  - `{cond}`")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")


def write_rules_json(out_path: pathlib.Path, selected: list[RuleCandidate], selected_mask: np.ndarray, y: np.ndarray) -> None:
    payload = {
        "note": "Explanatory surrogate rules for Jes final. Derived features are not supported by apply_rules; use as analysis artifact.",
        "metrics": union_metrics(selected_mask, y),
        "rules": [
            {
                "name": cand.name,
                "own_metrics": {
                    "kept": int(cand.mask.sum()),
                    "hit": cand.tp,
                    "fp": cand.fp,
                    "precision": cand.precision,
                    "recall": cand.recall,
                },
                "conditions": cand.conditions,
            }
            for cand in selected
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/full/output.jsonl")
    ap.add_argument("--final", default="03_filter/combined_corpus_jes_current.jsonl")
    ap.add_argument("--rules", default="03_filter/rules/rules_jes.json")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-rules", type=int, default=20)
    ap.add_argument("--target-size", type=int, default=1500)
    ap.add_argument("--report-out", default="03_filter/experiments/jes_surrogate_report.md")
    ap.add_argument("--rules-out", default="03_filter/experiments/jes_surrogate_rules.json")
    args = ap.parse_args()

    papers = load_papers(pathlib.Path(args.data))
    target_ids = load_ids(pathlib.Path(args.final))
    x, y, names, meta = build_features(
        papers,
        target_ids,
        pathlib.Path(args.rules),
        args.alpha,
        args.random_state,
    )
    print(f"Loaded {len(papers)} papers; positives={int(y.sum())}; prefilter={meta['prefilter_count']}")

    print("Evaluating model upper bounds ...")
    model_results = evaluate_models(x, y, names, args.random_state)
    for r in model_results:
        top = r["test_top_k"]
        print(
            f"  {r['model']:<22} test top-k F1={top['f1']:.3f} "
            f"P={top['precision']:.3f} R={top['recall']:.3f} AUC={r['test_roc_auc']:.3f}"
        )

    print("Training rule source forest on all papers ...")
    source_model = ExtraTreesClassifier(
        n_estimators=900,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.random_state,
    )
    source_model.fit(x, y)

    print("Extracting candidate rules ...")
    candidates = extract_rule_candidates(
        source_model,
        x,
        y,
        names,
        min_support=25,
        min_tp=10,
        min_precision=0.35,
        max_candidates=8000,
    )
    print(f"  candidates={len(candidates)}")

    print("Greedy-selecting compact surrogate rules ...")
    selected, selected_mask, metrics = greedy_select_rules(
        candidates,
        y,
        target_size=args.target_size,
        max_rules=args.max_rules,
    )
    print(
        f"  selected={len(selected)} kept={int(metrics['kept'])} hit={int(metrics['tp'])} "
        f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
    )

    report_out = pathlib.Path(args.report_out)
    rules_out = pathlib.Path(args.rules_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    render_report(
        report_out,
        pathlib.Path(args.final),
        pathlib.Path(args.rules),
        meta,
        model_results,
        selected,
        selected_mask,
        y,
        names,
        source_model,
        x,
    )
    write_rules_json(rules_out, selected, selected_mask, y)
    print(f"Wrote {report_out}")
    print(f"Wrote {rules_out}")


if __name__ == "__main__":
    main()
