"""
python_filter.py — Paper selection using three non-Euclidean mechanisms.

JSON rules can only cut hyperrectangles in the score space.
This module implements three structurally different mechanisms:

  Mechanism 1 — Dimensional consistency:
    Detect papers whose score pattern is internally inconsistent.
    "Formalism theater" (mr>>tn, mr>>md) and "experimental claims"
    (high er + intact/partial) are signatures of noise, not theory.
    This is an anomaly-detection lens: we're not checking "is mr high?"
    but "does mr make sense given tn and md and er?"

  Mechanism 2 — Multi-signal vote counting:
    mr/tn/md are correlated (r=0.82-0.90) and represent ONE underlying dimension.
    Counting them as separate conditions triple-counts the same signal.
    Instead: one composite vote for the math-backbone trio, then separate
    votes for each genuinely independent dimension (er, ei, ig, cc, tea-presence).
    Six independent signals simultaneously pointing "theoretical" is far
    stronger evidence than a single dimension at a very high threshold.

  Mechanism 3 — Intra-group relative rank:
    Not "mr >= 6" but "top 40% of your integrity group by FLD score."
    A partial paper in the 90th percentile of partials is more valuable
    than an intact paper in the 15th percentile of intacts.
    Absolute thresholds are blind to the local distribution; percentile
    rank adapts to it.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from mmr_select import build_matrix, compute_fld  # noqa: E402

DIMS = ["mr", "tn", "md", "ar", "er", "ei", "sg"]

_NEGATIVE_FLAGS = frozenset({"FORMALISM_THEATER", "EXPERIMENTAL_CLAIM", "MARKETING_THEORY"})
_RESCUE_FLAGS   = frozenset({"DEEP_MATH_OUTLIER", "THEORY_BREAKTHROUGH", "HIGH_EPISTEMIC"})


# ── helpers ───────────────────────────────────────────────────────────────────

def _f(p: dict[str, Any], field: str) -> float | None:
    """Extract numeric score, handling both flat and nested-dict formats."""
    v = p.get(field)
    if isinstance(v, (int, float)) and v < 20:
        return float(v)
    if isinstance(v, dict):
        s = v.get("s") or v.get("score")
        if isinstance(s, (int, float)):
            return float(s)
    return None


def _pid(p: dict[str, Any], i: int = 0) -> str:
    return str(p.get("id") or p.get("paper_id") or p.get("_id") or i)


# ── Mechanism 1: Dimensional consistency flags ────────────────────────────────

def _consistency_flags(p: dict[str, Any]) -> frozenset[str]:
    """
    Detect papers where score dimensions are in unexpected conflict.

    Negative patterns — inconsistency as a red flag:

      FORMALISM_THEATER: mr >= 7 but tn <= 3 and md <= 3.
        High rigor score combined with near-zero novelty and depth signals
        "we use equations to look rigorous, not to prove things."
        In the data, this pattern captures papers that import heavy notation
        from a well-known area but contribute no new mathematics.

      EXPERIMENTAL_CLAIM: ig in {intact,partial} and er >= 7 and ei <= 3.
        The classifier found a proof chain, but the paper is heavily experiment-
        driven and shows no epistemological commitment to theoretical explanation.
        Most likely: (a) proving performance bounds on an empirical method, or
        (b) a misclassification in the ig scoring. Either way, not our target.

      MARKETING_THEORY: marketing flag True and mr >= 5.
        Papers with genuine theoretical contributions don't need to market
        themselves. This combination is a signal of "appearance of rigor."

    Positive anomalies — undervalued by FLD/ig, deserving rescue:

      DEEP_MATH_OUTLIER: md >= 9.
        Maximal mathematical depth regardless of ig. Even a "broken" proof
        using spectral graph theory or differential geometry is worth reading.
        The math tool itself is the value, not the completeness of the argument.

      THEORY_BREAKTHROUGH: ig in {partial,broken} and mr >= 7 and tn >= 8.
        Real theoretical contribution buried under an incomplete writeup.
        FLD penalises partial/broken; this pattern rescues papers where the
        math is genuinely new even though the proof isn't fully assembled.

      HIGH_EPISTEMIC: ei >= 9 and mr >= 5.
        Maximum epistemological commitment to foundational explanation,
        with real mathematics. Rare. Always worth including.
    """
    mr = _f(p, "mr") or 0
    tn = _f(p, "tn") or 0
    md = _f(p, "md") or 0
    er = _f(p, "er") or 5
    ei = _f(p, "ei") or 5
    ig = p.get("ig")
    mk = bool(p.get("mk_f"))

    flags: set[str] = set()

    if mr >= 7 and tn <= 3 and md <= 3:
        flags.add("FORMALISM_THEATER")
    if ig in ("intact", "partial") and er >= 7 and ei <= 3:
        flags.add("EXPERIMENTAL_CLAIM")
    if mk and mr >= 5:
        flags.add("MARKETING_THEORY")

    if md >= 9:
        flags.add("DEEP_MATH_OUTLIER")
    if ig in ("partial", "broken") and mr >= 7 and tn >= 8:
        flags.add("THEORY_BREAKTHROUGH")
    if ei >= 9 and mr >= 5:
        flags.add("HIGH_EPISTEMIC")

    return frozenset(flags)


# ── Mechanism 2: Multi-signal vote counting ───────────────────────────────────

def _vote_score(p: dict[str, Any]) -> int:
    """
    Count of independent binary signals pointing toward "genuine theory paper."

    Critical design decision: mr/tn/md are correlated at r=0.82-0.90 in this
    dataset. They represent one underlying dimension (math rigor). Counting
    them as three separate votes would triple-penalise the same signal.
    They get ONE composite vote.

    Independent dimensions (confirmed by PCA analysis — see selection_methodology.md):
      PC1 57.6%: mr/tn/md cluster  → one combined vote
      PC2 13.4%: ar alone          → not counted (ar std=1.27, unreliable)
      PC3 10.7%: ei alone          → one vote
      er: strongly anti-correlated → one independent vote
      ig: categorical direct evidence → weighted vote
      cc: weakly independent (low compute = theory) → one vote
      tea=null: pure-theory flag when combined with low er → one vote

    Scale (observed range for real papers): -4 to +8.
    """
    mr  = _f(p, "mr") or 0
    tn  = _f(p, "tn") or 0
    md  = _f(p, "md") or 0
    er  = _f(p, "er") or 5
    ei  = _f(p, "ei") or 5
    cc  = _f(p, "cc") or 5
    tea = _f(p, "tea")
    ig  = p.get("ig")
    mk  = bool(p.get("mk_f"))
    hr  = bool(p.get("hr_f"))

    v = 0

    # Vote 1: math backbone — one vote for the correlated (mr, tn, md) cluster
    math_avg = (mr + tn + md) / 3
    if   math_avg >= 7.0: v += 2
    elif math_avg >= 5.5: v += 1

    # Vote 2: experiment reliance — genuinely independent axis (r≈-0.83 with mr)
    if   er <= 2: v += 2
    elif er <= 4: v += 1

    # Vote 3: epistemological intent — PCA PC3, orthogonal to the main theory axis
    if ei >= 8: v += 1

    # Vote 4: integrity classification — direct categorical evidence
    if   ig == "intact":  v += 2
    elif ig == "partial": v += 1
    elif ig == "broken":  v -= 1
    elif ig == "absent":  v -= 3

    # Vote 5: compute scale — theory papers don't need GPU clusters
    if cc <= 1: v += 1

    # Vote 6: pure-theory signal — no tea value AND low er
    # (tea=null means paper is either pure theory OR pure empirical;
    # the er<=4 condition selects the pure-theory case)
    if tea is None and er <= 4: v += 1

    if mk: v -= 1  # marketing is a negative signal
    if hr: v += 3  # human review overrides everything below

    return v


# ── Mechanism 3: Intra-group relative rank ────────────────────────────────────

def _intragroup_percentiles(
    papers: list[dict[str, Any]],
    fld_raw: np.ndarray,
) -> dict[str, float]:
    """
    Within each ig group, compute each paper's percentile by FLD score.

    Why this matters: absolute thresholds apply the same cut across all groups.
    But a partial paper at the 85th percentile of partials is more interesting
    than an intact paper at the 10th percentile of intacts. Percentile rank
    makes the selection criterion relative to the local distribution, not the
    global one. This is especially important for rescuing high-value partial papers.
    """
    groups: dict[str, list[tuple[str, float]]] = {}
    for i, p in enumerate(papers):
        pid = _pid(p, i)
        ig  = p.get("ig") or "absent"
        groups.setdefault(ig, []).append((pid, float(fld_raw[i])))

    pcts: dict[str, float] = {}
    for ig_group, members in groups.items():
        n = len(members)
        sorted_pids = [pid for pid, _ in sorted(members, key=lambda x: x[1])]
        for rank, pid in enumerate(sorted_pids):
            pcts[pid] = rank / max(n - 1, 1)
    return pcts


# ── Main filter function ──────────────────────────────────────────────────────

def filter_papers(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Select the theory-paper corpus using the three mechanisms above.

    Decision logic per paper:
      1. Rescue path: human_review, DEEP_MATH_OUTLIER, THEORY_BREAKTHROUGH,
         or HIGH_EPISTEMIC with enough votes → keep unconditionally.
      2. Hard exclusion: ig=absent → drop.
      3. Negative consistency: 2+ negative flags → drop (multiple independent
         signals of noise are reliable even when individual signals are not).
      4. Primary gate: vote threshold × percentile cutoff, tuned per ig group.
         A single negative flag raises the vote threshold by 1 (the paper needs
         to compensate with stronger positive signals).

    Target: 1000-2000 papers from ~31k. Thresholds calibrated for that range;
    adjust THRESHOLDS dict below if the corpus drifts outside it.
    """
    THRESHOLDS = {
        # (min_votes, min_percentile_within_group)
        # intact: keep roughly the top 30% of intact papers
        "intact":  (5, 0.40),
        # partial: keep only high-rank partials with strong multi-signal convergence
        "partial": (6, 0.62),
        # broken: only extreme cases not caught by rescue rules
        "broken":  (7, 0.85),
    }

    # FLD weights from the full population (see ourinsights/selection_methodology.md
    # for why computing from a filtered subset corrupts the weights)
    X, _ = build_matrix(papers, DIMS)
    _, w, _ = compute_fld(X, papers, DIMS)
    fld_raw = X @ w

    pcts = _intragroup_percentiles(papers, fld_raw)

    kept: list[dict[str, Any]] = []

    for i, p in enumerate(papers):
        pid   = _pid(p, i)
        ig    = p.get("ig") or "absent"
        flags = _consistency_flags(p)
        votes = _vote_score(p)
        pct   = pcts.get(pid, 0.0)
        hr    = bool(p.get("hr_f"))

        # ── Hard exclusion: absent — no formal claims by definition ─────────
        # Nothing overrides this. The hr_f flag in this dataset has 402 absent
        # papers with hr=True (vs only 9 intact), which looks like a data artifact
        # rather than genuine human annotation. We exclude all absent papers.
        if ig == "absent":
            continue

        # ── Rescue (non-absent papers only) ───────────────────────────────
        # hr_f has large numbers of spurious True values in this dataset
        # (54 broken + 402 absent with hr=True vs only 9 intact).
        # Broken papers cannot be rescued by hr alone — they need a positive
        # content signal. Partial/intact can be rescued by hr.
        if hr and ig != "broken":
            kept.append(p); continue
        if "DEEP_MATH_OUTLIER" in flags:
            kept.append(p); continue
        if "THEORY_BREAKTHROUGH" in flags:
            kept.append(p); continue
        if "HIGH_EPISTEMIC" in flags and votes >= 4:
            kept.append(p); continue

        # ── Multi-negative exclusion ───────────────────────────────────────
        n_neg = len(flags & _NEGATIVE_FLAGS)
        if n_neg >= 2:
            continue

        # ── Primary gate ───────────────────────────────────────────────────
        if ig not in THRESHOLDS:
            continue

        min_votes, min_pct = THRESHOLDS[ig]
        # One negative flag raises the vote bar — the paper must compensate.
        effective_min_votes = min_votes + n_neg

        if votes >= effective_min_votes and pct >= min_pct:
            kept.append(p)

    return kept


# ── Validation ────────────────────────────────────────────────────────────────

def main() -> None:
    data_path = pathlib.Path("data/full/output.jsonl")
    print(f"Loading {data_path} ...")

    papers: list[dict[str, Any]] = []
    with data_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r.get("parsed"):
                p = dict(r["parsed"])
                p["_id"]    = p.get("id") or p.get("paper_id")
                p["_title"] = r.get("title", "")
                p["_venue"] = r.get("venue", "")
                p["_year"]  = r.get("year")
                papers.append(p)

    print(f"Total papers: {len(papers)}")
    kept = filter_papers(papers)
    print(f"Kept:  {len(kept)} ({len(kept)/len(papers):.1%})")
    print(f"ig distribution: {dict(Counter(p.get('ig') for p in kept))}")

    # Diagnostic: how many papers were rescued by each positive flag?
    rescue_counts: Counter[str] = Counter()
    for p in papers:
        flags = _consistency_flags(p)
        for f in flags & _RESCUE_FLAGS:
            rescue_counts[f] += 1
    print(f"\nRescue flag coverage (full corpus): {dict(rescue_counts)}")

    # Diagnostic: how many papers were caught by each negative flag?
    neg_counts: Counter[str] = Counter()
    for p in papers:
        flags = _consistency_flags(p)
        for f in flags & _NEGATIVE_FLAGS:
            neg_counts[f] += 1
    print(f"Negative flag coverage (full corpus): {dict(neg_counts)}")

    print("\nSample kept papers (top 15 by mr+tn):")
    ranked = sorted(kept, key=lambda p: (_f(p, "mr") or 0) + (_f(p, "tn") or 0), reverse=True)
    for p in ranked[:15]:
        print(
            f"  ig={p.get('ig','?'):8s}  mr={_f(p,'mr') or 0:.1f}  "
            f"tn={_f(p,'tn') or 0:.1f}  er={_f(p,'er') or 0:.1f}  "
            f"{p.get('_title','')[:58]}"
        )


if __name__ == "__main__":
    main()
