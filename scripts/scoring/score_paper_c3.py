#!/usr/bin/env python3
"""C3 模型打分脚本 — v0.8 扁平 schema，先分数后 fuse 再条件 rationale

流程: 打分 → fuse 机械判断 → 路径分叉（compact / full）→ 输出

用法: uv run scripts/score_paper_c3.py [N] [model] [workers]
默认 N=50, model=deepseek-v4-pro, workers=16

环境变量: DEEPSEEK_API_KEY 必填
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

PROMPT_PATH = "prompt_c3.md"
SAMPLE_PATH = "data/sample_N50_c3test.json"
OUTPUT_PATH = "data/llm_c3_v08_N50.jsonl"
API_URL = "https://api.deepseek.com/v1/chat/completions"

_write_lock = threading.Lock()

SCORE_FIELDS = ("mr", "tn", "md", "ar", "er", "tea", "cc", "ei", "sg", "cs")
DIM_NAMES = {1: "mr", 2: "tn", 3: "md", 4: "ar", 5: "er"}


def build_user_message(paper: dict) -> str:
    return (
        "<paper>\n"
        f"<paper_id>{paper.get('paper_id', '')}</paper_id>\n"
        f"<title>{paper.get('title', '')}</title>\n"
        f"<venue>{paper.get('venue', '')}</venue>\n"
        f"<year>{paper.get('year', '')}</year>\n"
        f"<abstract>{paper.get('abstract', '')}</abstract>\n"
        "</paper>"
    )


def call_deepseek(system: str, user: str, model: str, api_key: str) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    r = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=900,
    )
    r.raise_for_status()
    return r.json()


def parse_content(content: str) -> dict:
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    return json.loads(s)


def validate_c3(parsed: dict) -> tuple[bool, list[str]]:
    """验证 C3 schema 的关键约束。返回 (ok, errors)。"""
    errors = []

    if parsed.get("pv") != "v0.8":
        errors.append(f"pv={parsed.get('pv')} != v0.8")

    ig = parsed.get("ig")
    fuse = parsed.get("fuse")
    rr = parsed.get("rr")

    if ig not in ("intact", "partial", "broken", "absent"):
        errors.append(f"ig='{ig}' 无效")

    # 核心约束：ig==absent → fuse 必须为 true
    if ig == "absent" and fuse is not True:
        errors.append(f"ig=absent 但 fuse={fuse}（应为 true）")

    if not isinstance(fuse, bool):
        errors.append(f"fuse={fuse!r} 不是 bool")

    # 数值评分范围检查 [0, 10]
    score_fields = ["mr", "tn", "md", "ar", "er", "tea", "cc", "ei", "sg", "cs"]
    for sf in score_fields:
        v = parsed.get(sf)
        if v is not None and isinstance(v, (int, float)) and not (0 <= v <= 10):
            errors.append(f"{sf}={v} 超出 [0,10] 范围")

    # rr 格式：自动修复，不拒绝
    if not isinstance(rr, list):
        errors.append(f"rr={rr!r} 不是 list")
    else:
        # 过滤合法元素（1–5整数），取最多2个
        valid = sorted({x for x in rr if isinstance(x, int) and 1 <= x <= 5})
        if len(valid) >= 2:
            parsed["rr"] = valid[:2]
        elif len(rr) == 0:
            parsed["rr"] = []
        else:
            # 无法修复（元素全部非法），从原始分数重新推导
            scores = {i: parsed.get(k) for i, k in enumerate(["mr","tn","md","ar","er"], 1)}
            valid_scores = {i: v for i, v in scores.items() if isinstance(v, (int, float))}
            if len(valid_scores) >= 2:
                parsed["rr"] = sorted(sorted(valid_scores, key=valid_scores.__getitem__)[:2])
            else:
                parsed["rr"] = []

    # fuse=true 时不应有 rationale 字段
    if fuse is True:
        rationale_fields = [k for k in parsed if k.endswith("_r") or k in ("lc_p", "lc_t", "lc_d", "lc_c", "osn")]
        if rationale_fields:
            errors.append(f"fuse=true 但存在 rationale 字段: {rationale_fields[:3]}")

    # fuse=false 时必须有 rationale 块
    if fuse is False:
        required_r = ["mr_r", "tn_r", "md_r", "ar_r", "er_r", "cc_r", "ei_r", "sg_r", "cs_r",
                      "lc_p", "lc_t", "lc_d", "lc_c", "mk_r", "hr_r", "osn"]
        missing = [f for f in required_r if f not in parsed]
        if missing:
            errors.append(f"fuse=false 缺少字段: {missing[:5]}")

    return len(errors) == 0, errors


def verify_fuse_logic(parsed: dict) -> tuple[bool, str]:
    """机械验证 fuse 是否符合 ig==absent 规则。"""
    ig = parsed.get("ig")
    fuse = parsed.get("fuse")
    expected_fuse = (ig == "absent")
    if fuse != expected_fuse:
        return False, f"ig={ig} → expected fuse={expected_fuse}, got fuse={fuse}"
    return True, "ok"


def process_one(paper: dict, system_prompt: str, model: str, api_key: str, order_idx: int) -> dict:
    record = {
        "_order_idx": order_idx,
        "paper_id": paper["paper_id"],
        "title": paper.get("title", ""),
        "venue": paper.get("venue"),
        "year": paper.get("year"),
    }
    t0 = time.time()
    try:
        resp = call_deepseek(system_prompt, build_user_message(paper), model, api_key)
        msg = resp["choices"][0]["message"]
        usage = resp.get("usage", {})
        elapsed = time.time() - t0

        content = (msg.get("content", "") or "").strip()

        try:
            parsed = parse_content(content)
            schema_ok, schema_errors = validate_c3(parsed)
            fuse_ok, fuse_msg = verify_fuse_logic(parsed)
            ok = schema_ok and fuse_ok
            err = "; ".join(schema_errors) if not schema_ok else (None if fuse_ok else fuse_msg)
        except Exception as e:
            parsed = None
            schema_ok = fuse_ok = ok = False
            schema_errors = []
            err = f"JSON parse failed: {e}"

        record.update({
            "elapsed_s": round(elapsed, 1),
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            "content": content,
            "parsed": parsed,
            "ok": ok,
            "schema_ok": schema_ok,
            "fuse_ok": fuse_ok,
            "error": err,
        })

    except requests.HTTPError as e:
        body = e.response.text[:1000] if e.response is not None else ""
        record.update({
            "ok": False,
            "schema_ok": False,
            "fuse_ok": False,
            "elapsed_s": round(time.time() - t0, 1),
            "error": f"HTTPError: {e}",
            "error_body": body,
        })
    except Exception as e:
        record.update({
            "ok": False,
            "schema_ok": False,
            "fuse_ok": False,
            "elapsed_s": round(time.time() - t0, 1),
            "error": repr(e),
        })
    return record


def append_jsonl(path: str, record: dict):
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec["paper_id"])
            except Exception:
                continue
    return done


def print_summary(output_path: str):
    recs = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except Exception:
                continue

    ok_recs = [r for r in recs if r.get("ok")]
    print(f"\n{'='*60}")
    print(f"C3 v0.8 测试结果摘要 ({len(ok_recs)}/{len(recs)} 成功)")
    print(f"{'='*60}")

    if not ok_recs:
        print("无成功记录")
        return

    fuse_true = sum(1 for r in ok_recs if r.get("parsed", {}).get("fuse") is True)
    fuse_false = sum(1 for r in ok_recs if r.get("parsed", {}).get("fuse") is False)
    schema_ok = sum(1 for r in ok_recs if r.get("schema_ok"))
    fuse_logic_ok = sum(1 for r in ok_recs if r.get("fuse_ok"))

    print(f"fuse=true  (紧凑模式): {fuse_true} ({fuse_true/len(ok_recs)*100:.1f}%)")
    print(f"fuse=false (完整模式): {fuse_false} ({fuse_false/len(ok_recs)*100:.1f}%)")
    print(f"schema 校验通过: {schema_ok}/{len(ok_recs)}")
    print(f"fuse 逻辑正确:   {fuse_logic_ok}/{len(ok_recs)}")

    # Token 统计
    tout_compact = [r["tokens_out"] for r in ok_recs if r.get("parsed", {}).get("fuse") is True and r.get("tokens_out")]
    tout_full = [r["tokens_out"] for r in ok_recs if r.get("parsed", {}).get("fuse") is False and r.get("tokens_out")]
    if tout_compact:
        print(f"\n输出 tokens — 紧凑模式均值: {sum(tout_compact)//len(tout_compact)}")
    if tout_full:
        print(f"输出 tokens — 完整模式均值: {sum(tout_full)//len(tout_full)}")

    # 与 C1 的分数对比基准
    print(f"\n分数分布（C3 ok 样本）:")
    for field in ("mr", "tn", "md", "er"):
        vals = [r["parsed"][field] for r in ok_recs if r.get("parsed") and r["parsed"].get(field) is not None]
        if vals:
            print(f"  {field}: mean={sum(vals)/len(vals):.2f}, n={len(vals)}")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4-pro"
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY not set in environment")

    system_prompt = Path(PROMPT_PATH).read_text(encoding="utf-8")
    sample_payload = json.loads(Path(SAMPLE_PATH).read_text(encoding="utf-8"))
    samples = sample_payload["samples"][:n]

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_ids(OUTPUT_PATH)
    todo = [p for p in samples if p["paper_id"] not in completed]

    print(f"C3 v0.8 | 模型: {model} | 并发: {workers} | prompt: {len(system_prompt)} 字符")
    print(f"目标: {len(samples)} 篇  已完成: {len(completed & {p['paper_id'] for p in samples})}  待处理: {len(todo)}")

    if not todo:
        print("无新任务，直接输出摘要。")
        print_summary(OUTPUT_PATH)
        return

    t_start = time.time()
    ok_count = fail_count = 0
    pid_to_order = {p["paper_id"]: i for i, p in enumerate(samples)}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_one, p, system_prompt, model, api_key, pid_to_order[p["paper_id"]]): p
            for p in todo
        }
        with tqdm(total=len(todo), desc="C3打分", unit="篇") as pbar:
            for fut in as_completed(futures):
                rec = fut.result()
                append_jsonl(OUTPUT_PATH, rec)
                if rec.get("ok"):
                    ok_count += 1
                    p = rec.get("parsed") or {}
                    pbar.set_postfix(ok=ok_count, fail=fail_count,
                                     fuse=p.get("fuse"), ig=p.get("ig", "")[:3])
                else:
                    fail_count += 1
                    pbar.set_postfix(ok=ok_count, fail=fail_count,
                                     err=str(rec.get("error", ""))[:30])
                pbar.update(1)

    elapsed = time.time() - t_start
    print(f"\n完成 ok={ok_count} fail={fail_count}  墙钟 {elapsed:.0f}s")

    # 按原始顺序重排
    all_records = []
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    all_records.sort(key=lambda r: r.get("_order_idx", 10**9))
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print_summary(OUTPUT_PATH)
    print(f"\n→ {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
