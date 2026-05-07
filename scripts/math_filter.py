#!/usr/bin/env python3
"""
数学严谨性粗筛 — 基于摘要文本的关键词信号
输出：data/filtered/math_candidates.jsonl（通过粗筛的论文）
      data/filtered/math_filter_stats.json（各来源统计）
"""
import json
import re
import os
from collections import defaultdict
from datetime import datetime

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/filtered"

# 强信号：出现即高度相关（定理/引理/证明结构）
STRONG_PATTERNS = [
    r'\btheorem\b', r'\blemma\b', r'\bproposition\b', r'\bcorollary\b',
    r'\bproof\b', r'\bwe prove\b', r'\bwe show that\b',
    r'\bconvergence (rate|guarantee|analysis|bound)\b',
    r'\bsample complexity\b', r'\bregret bound\b', r'\bexcess risk\b',
    r'\bgeneralization bound\b', r'\bpac(-|\s)learn',
    r'\bnp-hard\b', r'\bnp-complete\b',
]

# 中等信号：数学分析相关术语
MEDIUM_PATTERNS = [
    r'\boptimality\b', r'\blower bound\b', r'\bupper bound\b',
    r'\boptimal rate\b', r'\bminimax\b', r'\bconsistency\b',
    r'\basymptotic\b', r'\banalysis\b.{0,30}\brate\b',
    r'\bconverges?\b', r'\bconvex\b', r'\bconcave\b',
    r'\bgradient\b.{0,20}\bconverg',
    r'\bstochastic\b.{0,20}\bconverg',
    r'\bvariational\b', r'\bfunctional\b.{0,20}\bspace\b',
    r'\bhilbert\b', r'\bbanach\b', r'\bsobolev\b',
    r'\bmeasure theor', r'\bprobability space\b',
    r'\brandom variable\b', r'\bexpectation\b.{0,30}\bbound\b',
]

# 弱信号（单独出现不算，需配合其他信号）
WEAK_PATTERNS = [
    r'\balgorithm\b', r'\boptimization\b', r'\bstatistical\b',
    r'\bestimator\b', r'\binference\b', r'\bapproximation\b',
]

STRONG_RE = [re.compile(p, re.IGNORECASE) for p in STRONG_PATTERNS]
MEDIUM_RE = [re.compile(p, re.IGNORECASE) for p in MEDIUM_PATTERNS]
WEAK_RE   = [re.compile(p, re.IGNORECASE) for p in WEAK_PATTERNS]


def score_paper(title: str, abstract: str) -> tuple[int, list[str]]:
    text = (title + " " + abstract).lower()
    hits = []
    score = 0

    for pat in STRONG_RE:
        if pat.search(text):
            score += 3
            hits.append(pat.pattern)

    for pat in MEDIUM_RE:
        if pat.search(text):
            score += 1
            hits.append(pat.pattern)

    weak_count = sum(1 for pat in WEAK_RE if pat.search(text))
    if weak_count >= 2:
        score += 1
        hits.append(f"weak×{weak_count}")

    return score, hits


def load_all_papers():
    papers = []
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(INPUT_DIR, fname)
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                p["_source_file"] = fname
                papers.append(p)
    return papers


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("加载数据...", flush=True)
    papers = load_all_papers()
    print(f"总计：{len(papers)} 篇", flush=True)

    # 按来源统计
    stats = defaultdict(lambda: {"total": 0, "passed": defaultdict(int)})
    candidates = []

    for p in papers:
        title    = p.get("title", "") or ""
        abstract = p.get("abstract", "") or ""
        venue    = p.get("venue", p.get("_source_file", "?"))
        score, hits = score_paper(title, abstract)
        p["math_score"] = score
        p["math_hits"]  = hits[:8]  # 最多保留8个命中

        stats[venue]["total"] += 1
        if score >= 3:   # 至少一个强信号，或多个中等信号
            stats[venue]["passed"]["score>=3"] += 1
        if score >= 6:
            stats[venue]["passed"]["score>=6"] += 1
        if score >= 9:
            stats[venue]["passed"]["score>=9"] += 1

        if score >= 3:
            candidates.append(p)

    # 写出候选论文
    out_path = os.path.join(OUTPUT_DIR, "math_candidates.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in candidates:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # 打印统计
    print(f"\n{'='*60}", flush=True)
    print(f"{'来源':<20} {'总计':>6} {'≥3':>6} {'≥6':>6} {'≥9':>6} {'≥3%':>7}", flush=True)
    print("-" * 60, flush=True)
    grand_total = grand_passed = 0
    for venue in sorted(stats):
        t = stats[venue]["total"]
        p3 = stats[venue]["passed"]["score>=3"]
        p6 = stats[venue]["passed"]["score>=6"]
        p9 = stats[venue]["passed"]["score>=9"]
        pct = p3 / t * 100 if t else 0
        grand_total += t
        grand_passed += p3
        print(f"{venue:<20} {t:>6} {p3:>6} {p6:>6} {p9:>6} {pct:>6.1f}%", flush=True)
    print("-" * 60, flush=True)
    print(f"{'合计':<20} {grand_total:>6} {grand_passed:>6} {'':>6} {'':>6} {grand_passed/grand_total*100:>6.1f}%", flush=True)

    # 保存 JSON 统计
    stats_out = {
        venue: {
            "total": v["total"],
            "passed_3": v["passed"]["score>=3"],
            "passed_6": v["passed"]["score>=6"],
            "passed_9": v["passed"]["score>=9"],
        }
        for venue, v in stats.items()
    }
    with open(os.path.join(OUTPUT_DIR, "math_filter_stats.json"), "w") as f:
        json.dump({"run_at": datetime.now().isoformat(), "threshold": 3, "venues": stats_out}, f, indent=2, ensure_ascii=False)

    print(f"\n候选论文写入：{out_path}（{len(candidates)} 篇）", flush=True)


if __name__ == "__main__":
    main()
