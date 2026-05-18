#!/usr/bin/env python3
"""DeepSeek V4-Flash 思考 打分测试 — 并发版

用法: uv run scripts/score_paper_test_v4flash_think.py [N] [model] [workers]
默认 N=20, model=deepseek-v4-flash, workers=16

特性:
- 并发请求(ThreadPoolExecutor),墙钟时长降到 ~ ceil(N / workers) × 单篇耗时
- 实时落盘:每篇完成立即追加到输出 JSONL,中断不丢已完成数据
- 续跑:启动时扫描已有输出,跳过已完成的 paper_id
- 开启思考模式 reasoning_effort=high(与 pro 思考对照)

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

PROMPT_PATH = "prompt_c1t.md"
SAMPLE_PATH = "data/sample_20.json"
OUTPUT_PATH = "data/llm_test_v4flash_think_run.jsonl"
API_URL = "https://api.deepseek.com/v1/chat/completions"

_write_lock = threading.Lock()


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
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
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


def parse_content(content: str):
    """容忍模型偶发的 markdown 围栏"""
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    return json.loads(s)


def process_one(
    paper: dict, system_prompt: str, model: str, api_key: str, order_idx: int
) -> dict:
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

        reasoning = msg.get("reasoning_content", "") or ""
        content = (msg.get("content", "") or "").strip()

        try:
            parsed = parse_content(content)
            ok = True
            err = None
        except Exception as e:
            parsed = None
            ok = False
            err = f"JSON parse failed: {e}"

        record.update(
            {
                "elapsed_s": round(elapsed, 1),
                "tokens_in": usage.get("prompt_tokens"),
                "tokens_out": usage.get("completion_tokens"),
                "tokens_cache_hit": usage.get("prompt_cache_hit_tokens"),
                "tokens_cache_miss": usage.get("prompt_cache_miss_tokens"),
                "reasoning_chars": len(reasoning),
                "reasoning": reasoning,
                "raw_content": content,
                "parsed": parsed,
                "ok": ok,
                "error": err,
            }
        )
    except requests.HTTPError as e:
        body = e.response.text[:1000] if e.response is not None else ""
        record.update(
            {
                "ok": False,
                "elapsed_s": round(time.time() - t0, 1),
                "error": f"HTTPError: {e}",
                "error_body": body,
            }
        )
    except Exception as e:
        record.update(
            {
                "ok": False,
                "elapsed_s": round(time.time() - t0, 1),
                "error": repr(e),
            }
        )
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


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4-flash"
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

    print(f"模型: {model}(思考)  并发: {workers}  prompt: {len(system_prompt)} 字符")
    print(
        f"目标: {len(samples)} 篇  已完成: {len(completed & {p['paper_id'] for p in samples})}  待处理: {len(todo)}"
    )

    if not todo:
        print("无新任务,直接退出。")
        return

    t_start = time.time()
    ok_count = 0
    fail_count = 0

    pid_to_order = {p["paper_id"]: i for i, p in enumerate(samples)}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                process_one,
                p,
                system_prompt,
                model,
                api_key,
                pid_to_order[p["paper_id"]],
            ): p
            for p in todo
        }
        with tqdm(total=len(todo), desc="打分中", unit="篇") as pbar:
            for fut in as_completed(futures):
                rec = fut.result()
                append_jsonl(OUTPUT_PATH, rec)
                if rec.get("ok"):
                    ok_count += 1
                    parsed = rec.get("parsed") or {}
                    mr = (parsed.get("mathematical_rigor") or {}).get("score")
                    integ = (parsed.get("logical_chain") or {}).get("integrity")
                    pbar.set_postfix(
                        ok=ok_count,
                        fail=fail_count,
                        last=f"rigor={mr} integ={integ} reasoning={rec.get('reasoning_chars', 0)}c",
                    )
                else:
                    fail_count += 1
                    pbar.set_postfix(
                        ok=ok_count,
                        fail=fail_count,
                        last=f"ERR: {str(rec.get('error', ''))[:40]}",
                    )
                pbar.update(1)

    elapsed = time.time() - t_start
    print(f"\n完成 ok={ok_count} fail={fail_count}  墙钟 {elapsed:.0f}s")

    print("按原始顺序重排输出 ...", end=" ", flush=True)
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
    print(f"完成({len(all_records)} 条)")

    total_hit = total_miss = total_in = total_out = 0
    n_with_usage = 0
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if not rec.get("ok"):
                    continue
                hit = rec.get("tokens_cache_hit") or 0
                miss = rec.get("tokens_cache_miss") or 0
                ti = rec.get("tokens_in") or 0
                to = rec.get("tokens_out") or 0
                total_hit += hit
                total_miss += miss
                total_in += ti
                total_out += to
                if hit or miss:
                    n_with_usage += 1
            except Exception:
                continue
    if n_with_usage:
        hit_pct = total_hit / max(total_in, 1) * 100
        print(f"输入 tokens: {total_in:,}  其中 cache_hit={total_hit:,} ({hit_pct:.1f}%)  cache_miss={total_miss:,}")
    print(f"输出 tokens: {total_out:,}")
    avg_reasoning = sum(r.get("reasoning_chars", 0) for r in all_records if r.get("ok")) / max(ok_count, 1)
    print(f"avg reasoning: {avg_reasoning:.0f} chars")
    print(f"→ {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
