#!/usr/bin/env python3
"""重复打分方差实验

对 N 篇论文各重复打分 R 次，测量：
- 每个维度的组内标准差（within-paper std）
- 是否出现双峰（bimodal）
- 锚定效应（整数/半整数过度使用）
- ig 分类的一致性

用法: uv run 02_score/score_repeatability.py [N=10] [R=5] [model=deepseek-v4-pro]
输出: data/repeatability/results.jsonl  +  data/repeatability/report.json
"""

import os, sys, json, time, random, threading, statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

N_PAPERS  = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MODEL     = sys.argv[3] if len(sys.argv) > 3 else "deepseek-v4-pro"

PROJECT    = Path(__file__).parent.parent
PROMPT_PATH  = PROJECT / "02_score/prompts/prompt_c3.md"
FULL_DATA    = PROJECT / "data/full/output.jsonl"
INPUT_DATA   = PROJECT / "data/full/input.json"
OUT_DIR     = PROJECT / "data/repeatability"
OUT_JSONL   = OUT_DIR / "results.jsonl"
OUT_REPORT  = OUT_DIR / "report.json"
API_URL     = "https://api.deepseek.com/v1/chat/completions"

SCORE_FIELDS = ["mr", "tn", "md", "ar", "er", "tea", "cc", "ei", "sg", "cs"]
_write_lock  = threading.Lock()


def pick_papers() -> list[dict]:
    """从全量结果里选 N 篇有代表性的论文（覆盖不同 ig 和分数段），join input.json 补 abstract"""
    # load abstracts from input.json
    with open(INPUT_DATA) as f:
        input_papers = {p['paper_id']: p for p in json.load(f)['samples']}

    with open(FULL_DATA) as f:
        records = [json.loads(l) for l in f
                   if not json.loads(l).get('error') and json.loads(l).get('ok')]

    def valid_scores(r):
        p = r.get('parsed', {})
        return all(isinstance(p.get(f), (int, float)) and 0 <= p[f] <= 10
                   for f in ['mr', 'tn', 'md'])

    # 各 ig 组各取若干篇，覆盖低/中/高分段
    selected = []
    random.seed(42)
    for ig in ['intact', 'partial', 'absent']:
        pool = [r for r in records
                if r['parsed'].get('ig') == ig and valid_scores(r)
                and r['paper_id'] in input_papers]
        pool.sort(key=lambda r: r['parsed'].get('mr', 0))
        thirds = max(len(pool) // 3, 1)
        for seg in [pool[:thirds], pool[thirds:2*thirds], pool[2*thirds:]]:
            if seg:
                selected.append(random.choice(seg))
        if len(selected) >= N_PAPERS:
            break

    selected = selected[:N_PAPERS]
    # merge abstract from input
    result = []
    for r in selected:
        inp = input_papers[r['paper_id']]
        result.append({**inp, 'parsed': r['parsed']})

    print(f"Selected {len(result)} papers: " +
          ", ".join(f"{p['paper_id']}(ig={p['parsed'].get('ig')},mr={p['parsed'].get('mr')})"
                    for p in result))
    return result


def build_user_msg(paper: dict) -> str:
    return (
        "<paper>\n"
        f"<paper_id>{paper['paper_id']}</paper_id>\n"
        f"<title>{paper.get('title','')}</title>\n"
        f"<venue>{paper.get('venue','')}</venue>\n"
        f"<year>{paper.get('year','')}</year>\n"
        f"<abstract>{paper.get('abstract','')}</abstract>\n"
        "</paper>"
    )


def call_api(system: str, user: str, api_key: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    r = requests.post(API_URL,
                      headers={"Authorization": f"Bearer {api_key}",
                               "Content-Type": "application/json"},
                      json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def parse_scores(content: str) -> dict | None:
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def score_once(paper: dict, system: str, api_key: str, repeat_idx: int) -> dict:
    t0 = time.time()
    try:
        resp = call_api(system, build_user_msg(paper), api_key)
        content = resp["choices"][0]["message"].get("content", "").strip()
        parsed  = parse_scores(content)
        elapsed = time.time() - t0
        scores  = {}
        if parsed:
            for f in SCORE_FIELDS:
                v = parsed.get(f)
                if isinstance(v, (int, float)) and 0 <= v <= 10:
                    scores[f] = float(v)
        ig = parsed.get("ig") if parsed else None
        # ok = parsed successfully; absent papers have null scores by design
        ok = parsed is not None and ig is not None
        return {
            "paper_id":   paper["paper_id"],
            "repeat_idx": repeat_idx,
            "scores":     scores,
            "ig":         ig,
            "mk_f":       parsed.get("mk_f") if parsed else None,
            "elapsed_s":  round(elapsed, 2),
            "ok":         ok,
            "content":    content,
        }
    except Exception as e:
        return {
            "paper_id":   paper["paper_id"],
            "repeat_idx": repeat_idx,
            "scores":     {},
            "ig":         None,
            "ok":         False,
            "error":      str(e),
            "elapsed_s":  round(time.time() - t0, 2),
        }


def write_result(rec: dict):
    with _write_lock:
        with open(OUT_JSONL, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_report(results: list[dict]) -> dict:
    from collections import Counter

    # group by paper_id
    by_paper: dict[str, list] = {}
    for r in results:
        if r.get("ok"):
            by_paper.setdefault(r["paper_id"], []).append(r)

    report = {"n_papers": len(by_paper), "n_repeats": N_REPEATS, "model": MODEL, "fields": {}}

    for field in SCORE_FIELDS:
        within_stds = []
        all_vals    = []
        anchor_hits = 0
        anchor_total = 0

        for pid, reps in by_paper.items():
            vals = [r["scores"][field] for r in reps if field in r["scores"]]
            if len(vals) >= 2:
                within_stds.append(statistics.stdev(vals))
                all_vals.extend(vals)

        for v in all_vals:
            anchor_total += 1
            if v == int(v) or (v * 2) == int(v * 2):  # integer or .5
                anchor_hits += 1

        if within_stds:
            report["fields"][field] = {
                "mean_within_std":   round(statistics.mean(within_stds), 3),
                "median_within_std": round(statistics.median(within_stds), 3),
                "max_within_std":    round(max(within_stds), 3),
                "anchor_rate":       round(anchor_hits / anchor_total, 3) if anchor_total else None,
                "n_papers":          len(within_stds),
            }

    # ig consistency
    ig_consistency = []
    for pid, reps in by_paper.items():
        ig_vals = [r["ig"] for r in reps if r.get("ig")]
        if len(ig_vals) >= 2:
            most_common = Counter(ig_vals).most_common(1)[0][1]
            ig_consistency.append(most_common / len(ig_vals))

    report["ig_consistency_mean"] = round(statistics.mean(ig_consistency), 3) if ig_consistency else None

    return report


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("DEEPSEEK_API_KEY not set")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.unlink(missing_ok=True)

    system = PROMPT_PATH.read_text()
    papers = pick_papers()

    # build task list: each (paper, repeat_idx) pair
    tasks = [(p, i) for p in papers for i in range(N_REPEATS)]
    total = len(tasks)
    print(f"\n{total} API calls ({len(papers)} papers × {N_REPEATS} repeats), model={MODEL}")

    results = []
    # 先同步跑第一篇的第一次，让 system prompt 进入 DeepSeek 缓存
    print("Warming cache with first paper...")
    warmup = score_once(tasks[0][0], system, api_key, tasks[0][1])
    write_result(warmup)
    results.append(warmup)
    remaining = tasks[1:]
    print(f"Cache warm. Launching {len(remaining)} concurrent calls...")

    with ThreadPoolExecutor(max_workers=len(remaining)) as pool:
        futs = {pool.submit(score_once, p, system, api_key, i): (p["paper_id"], i)
                for p, i in remaining}
        with tqdm(total=len(remaining)) as bar:
            for fut in as_completed(futs):
                rec = fut.result()
                write_result(rec)
                results.append(rec)
                bar.set_postfix(ok=sum(1 for r in results if r.get("ok")))
                bar.update(1)

    report = compute_report(results)
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"\n=== 重复打分方差报告 ===")
    print(f"papers={report['n_papers']}  repeats={N_REPEATS}  model={MODEL}")
    print(f"ig 一致率: {report['ig_consistency_mean']}")
    print()
    print(f"{'field':6s}  {'组内std均值':>10}  {'组内std中位':>10}  {'锚定率':>8}")
    for f, v in report["fields"].items():
        print(f"{f:6s}  {v['mean_within_std']:>10.3f}  {v['median_within_std']:>10.3f}  {v['anchor_rate']:>8.1%}")

    print(f"\n结果: {OUT_JSONL}")
    print(f"报告: {OUT_REPORT}")


if __name__ == "__main__":
    main()
