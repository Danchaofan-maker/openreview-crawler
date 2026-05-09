#!/usr/bin/env python3
"""验证 v0.8 prompt 在 50 篇分层抽样上的表现。

分两层校验:
1. Schema 合规: 字段齐全、格式对
2. Fuse 一致性: 用模型自评的分数代入规则代码算"应该的 fuse",对比模型实际 fuse
"""
import json
import os
import pathlib
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]

for line in (ROOT / ".env").read_text().strip().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip('"').strip("'")

API_KEY = os.environ["DEEPSEEK_API_KEY"]
PROMPT = re.sub(r"<!--.*?-->", "", (ROOT / "prompt_c2.md").read_text(encoding="utf-8"), flags=re.DOTALL)
API_URL = "https://api.deepseek.com/v1/chat/completions"

SCORED = ROOT / "data" / "llm_v4pro_thinking_N600_seed42.jsonl"
N_SAMPLE = 50
SEED = 1234
random.seed(SEED)

# ── 加载已打分数据,从中分层抽样 ────────────────────────────────
def load_scored():
    out = []
    for l in SCORED.read_text().strip().splitlines():
        r = json.loads(l)
        p = r.get("parsed") or {}
        if not isinstance(p, dict):
            continue
        def gs(key):
            v = p.get(key)
            return v.get("score") if isinstance(v, dict) else (v if isinstance(v, (int, float)) else None)
        rec = {
            "pid": r.get("paper_id"),
            "title": r.get("title", ""),
            "venue": r.get("venue", ""),
            "mr": gs("mathematical_rigor"), "tn": gs("theoretical_novelty"),
            "md": gs("mathematical_depth"), "er": gs("empirical_reliance"),
            "ei": gs("epistemological_intent"), "cc": gs("compute_complexity"),
            "ig": (p.get("logical_chain") or {}).get("integrity"),
        }
        out.append(rec)
    return out

def stratify(scored):
    """50 篇 = 15 明显熔断 + 15 明显保留 + 10 边界 + 10 LLM 类"""
    obvious_fuse = [s for s in scored
                    if (s["mr"] or 10) <= 1 and (s["er"] or 0) >= 8 and s["ig"] == "absent"]
    obvious_keep = [s for s in scored
                    if (s["mr"] or 0) >= 6 and s["ig"] in ("intact", "partial")]
    borderline = [s for s in scored
                  if 2 <= (s["mr"] or 0) <= 5 and s["ig"] in ("partial", "broken")]
    llm_related = [s for s in scored
                   if any(w in (s["title"] or "").lower()
                          for w in ["llm","language model","gpt","prompt","instruction","alignment","rlhf","preference","cot","chain-of-thought","jailbreak","safety","privacy"])]

    random.shuffle(obvious_fuse); random.shuffle(obvious_keep)
    random.shuffle(borderline);   random.shuffle(llm_related)

    picks = []
    picks += [("obvious_fuse", s) for s in obvious_fuse[:15]]
    picks += [("obvious_keep", s) for s in obvious_keep[:15]]
    picks += [("borderline",   s) for s in borderline[:10]]
    picks += [("llm_related",  s) for s in llm_related[:10]]

    seen = set()
    dedup = []
    for label, s in picks:
        if s["pid"] in seen: continue
        seen.add(s["pid"])
        dedup.append((label, s))
    return dedup

def find_raw(pid):
    for f in (ROOT / "data" / "raw").glob("*.jsonl"):
        for l in f.read_text().splitlines():
            if not l.strip(): continue
            try: r = json.loads(l)
            except: continue
            if r.get("paper_id") == pid: return r
    return None

# ── API 调用 ──────────────────────────────────────────────────
def call_api(paper, label):
    user = (
        "<paper>\n"
        f"<paper_id>{paper.get('paper_id', '')}</paper_id>\n"
        f"<title>{paper.get('title', '')}</title>\n"
        f"<abstract>{paper.get('abstract', '')}</abstract>\n"
        "</paper>"
    )
    t0 = time.time()
    try:
        r = requests.post(API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": "deepseek-v4-pro",
                  "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": user}],
                  "stream": False, "thinking": {"type": "disabled"}},
            timeout=180)
        r.raise_for_status()
    except Exception as e:
        return {"label": label, "pid": paper["paper_id"], "title": paper.get("title",""),
                "elapsed": time.time()-t0, "err": str(e), "parsed": None}
    j = r.json()
    msg = j["choices"][0]["message"]["content"].strip()
    if msg.startswith("```"):
        msg = msg.strip("`").lstrip("json").strip()
    usage = j.get("usage", {})
    parsed = None; err = None
    try: parsed = json.loads(msg)
    except Exception as e: err = str(e)
    return {"label": label, "pid": paper["paper_id"], "title": paper.get("title",""),
            "elapsed": time.time()-t0, "in_tok": usage.get("prompt_tokens"),
            "out_tok": usage.get("completion_tokens"),
            "cache_hit": usage.get("prompt_cache_hit_tokens"),
            "raw": msg, "parsed": parsed, "err": err}

# ── 核心: 用模型自评分代入规则代码,算"期望 fuse" ─────────────────
def eval_fuse_from_scores(p):
    """返回 (expected_fuse, expected_rr, rescue_triggered)"""
    fuse = p.get("fuse")
    is_compact = (fuse is True)

    def num(k):
        v = p.get(k)
        if isinstance(v, dict): v = v.get("s")
        return v if isinstance(v, (int, float)) else None

    mr  = num("mr"); tn = num("tn"); md = num("md")
    er  = num("er"); ei = num("ei"); cc = num("cc")
    ig  = p.get("ig") if "ig" in p else (p.get("lc") or {}).get("ig")
    hrv = p.get("hr")
    hr_f = (hrv if isinstance(hrv, bool)
            else hrv.get("f") if isinstance(hrv, dict) else None)

    if any(v is None for v in [mr, tn, er, ei, cc, ig]):
        return (None, None, None)

    rules = []
    if mr <= 5  and tn <= 2.5: rules.append(1)
    if er >= 8  and ei <= 2:   rules.append(2)
    if cc >= 8:                rules.append(3)
    if mr <= 1:                rules.append(4)
    if ig == "absent":         rules.append(5)
    if er >= 7  and tn <= 3:   rules.append(6)

    rescue = (tn >= 8) or (ei >= 8) or (hr_f is True)
    expected_fuse = bool(rules) and not rescue
    return (expected_fuse, sorted(rules), rescue)

# ── 校验 ──────────────────────────────────────────────────────
def validate_schema(rec):
    p = rec["parsed"]
    if p is None: return [f"JSON parse failed: {rec.get('err')}"]
    issues = []
    if p.get("pv") != "v0.8": issues.append(f"pv != v0.8 (got {p.get('pv')!r})")
    if "fuse" not in p: issues.append("missing fuse")
    fuse = p.get("fuse")
    score_keys = ["mr","tn","md","ar","er","tea","cc","ei","sg","cs"]
    if fuse is True:
        if "rr" not in p: issues.append("compact missing rr")
        elif not isinstance(p["rr"], list): issues.append(f"rr not list: {p.get('rr')}")
        if "dm" not in p: issues.append("compact missing dm")
        for k in ["lc","osn"]:
            if k in p: issues.append(f"compact has forbidden field {k}")
        for sk in score_keys:
            v = p.get(sk)
            if v is not None and not isinstance(v, (int, float)):
                issues.append(f"compact {sk} not flat number")
    elif fuse is False:
        for sk in score_keys:
            v = p.get(sk)
            if not isinstance(v, dict) or "s" not in v:
                if not (sk == "tea" and (v is None or (isinstance(v, dict) and v.get("s") is None))):
                    issues.append(f"full {sk} not nested {{s,r}}")
        if "lc" not in p: issues.append("full missing lc")
    else: issues.append(f"fuse not bool: {fuse!r}")
    return issues

# ── 主流程 ────────────────────────────────────────────────────
def main():
    scored = load_scored()
    picks = stratify(scored)
    print(f"分层抽样 {len(picks)} 篇 (seed={SEED}):")
    by_label = {}
    for label, s in picks:
        by_label.setdefault(label, 0)
        by_label[label] += 1
    for k, v in by_label.items():
        print(f"  {k}: {v}")

    raw_papers = []
    for label, s in picks:
        raw = find_raw(s["pid"])
        if not raw:
            print(f"  [SKIP] {label} {s['pid']}")
            continue
        raw_papers.append((label, raw, s))
    print(f"实际可调用 {len(raw_papers)} 篇\n")

    print("=== 调用 API (并发 16) ===")
    results = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(call_api, raw, label): (label, raw, old) for label, raw, old in raw_papers}
        for f in as_completed(futs):
            try: rec = f.result()
            except Exception as e:
                label, raw, old = futs[f]
                rec = {"label": label, "pid": raw["paper_id"], "title": raw.get("title",""),
                       "err": str(e), "parsed": None}
            results.append(rec)

    print(f"\n=== 汇总 ({len(results)} 篇) ===")
    schema_ok = schema_bad = 0
    fuse_match = fuse_mismatch = fuse_unknown = 0
    rr_complete = rr_partial = rr_overshoot = 0
    rescue_triggers = 0
    total_in = total_out = total_hit = 0
    n_with_tok = 0
    mismatches = []

    for r in results:
        issues = validate_schema(r) if r.get("parsed") else ["no parse"]
        if issues: schema_bad += 1
        else: schema_ok += 1

        p = r.get("parsed") or {}
        actual_fuse = p.get("fuse")
        actual_rr = p.get("rr") or []
        expected_fuse, expected_rr, rescue = eval_fuse_from_scores(p)

        if expected_fuse is None:
            fuse_unknown += 1
        elif expected_fuse == actual_fuse:
            fuse_match += 1
        else:
            fuse_mismatch += 1
            mismatches.append({
                "label": r["label"], "title": r["title"][:55],
                "expected": expected_fuse, "actual": actual_fuse,
                "expected_rr": expected_rr, "actual_rr": actual_rr,
                "rescue": rescue,
            })

        if actual_fuse is True and expected_rr is not None:
            arr_set, err_set = set(actual_rr), set(expected_rr)
            if arr_set == err_set:        rr_complete += 1
            elif arr_set <= err_set:      rr_partial += 1
            else:                         rr_overshoot += 1

        if rescue: rescue_triggers += 1

        if r.get("in_tok"):
            total_in += r["in_tok"]; total_out += r["out_tok"]
            total_hit += r.get("cache_hit") or 0
            n_with_tok += 1

    print(f"\n## Schema 合规: {schema_ok}/{schema_ok+schema_bad}")
    print(f"## Fuse 一致性: {fuse_match}/{fuse_match+fuse_mismatch} 匹配 ({fuse_unknown} 篇分数不全无法判定)")
    print(f"   rr 状态: {rr_complete} 完美 / {rr_partial} 子集 / {rr_overshoot} 多报 (基于已熔断)")
    print(f"   rescue 触发: {rescue_triggers}")
    if mismatches:
        print(f"\n## Fuse 不一致 ({len(mismatches)} 篇):")
        for m in mismatches[:20]:
            print(f"   [{m['label']}] expected={m['expected']} actual={m['actual']} "
                  f"exp_rr={m['expected_rr']} act_rr={m['actual_rr']} rescue={m['rescue']}")
            print(f"      {m['title']}")

    if n_with_tok:
        print(f"\n## 成本: 平均 in={total_in/n_with_tok:.0f} out={total_out/n_with_tok:.0f} cache={total_hit/n_with_tok:.0f}")
        miss = total_in - total_hit
        cost = (total_hit*0.025 + miss*3 + total_out*6) / 1e6
        print(f"   本次 {n_with_tok} 篇 ¥{cost:.4f}, 推算 3 万篇 ¥{cost/n_with_tok*30000:.1f}")

    out_path = ROOT / "data" / "validate_v0_8.jsonl"
    with open(out_path, "w") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n详细结果 → {out_path}")

if __name__ == "__main__":
    main()
