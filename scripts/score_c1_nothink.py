#!/usr/bin/env python3
"""C1-nothink 打分脚本 — v0.7 schema，thinking mode disabled

基于 score_N600_seed42_v4pro_thinking.py，唯一改动：
  API payload 中 thinking disabled（移除 thinking:enabled 和 reasoning_effort）

用法:
  uv run scripts/score_c1_nothink.py [N] [workers] [model]

默认:
  N=50, workers=16, model=deepseek-v4-pro

环境变量:
  DEEPSEEK_API_KEY  必填
  MAX_COST_USD      软上限,默认 $50
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

import requests
from tqdm import tqdm


# ---------- 配置 ----------
PROMPT_PATH  = "prompt_c1nt.md"
SAMPLE_PATH  = "data/sample_N50_c3test.json"
OUTPUT_PATH  = "data/llm_c1_nothink_N50_test.jsonl"
FAILURE_PATH = "data/llm_c1_nothink_N50_test_failures.jsonl"
LOG_PATH     = "data/llm_c1_nothink_N50_test.log"
API_URL      = "https://api.deepseek.com/v1/chat/completions"

PRICE_INPUT_HIT  = 0.003625
PRICE_INPUT_MISS = 0.435
PRICE_OUTPUT     = 0.87

MAX_RETRIES          = 3
RETRY_BASE_BACKOFF   = 5
HEARTBEAT_EVERY      = 60
ERROR_RATE_THRESHOLD = 0.10
ERROR_PAUSE_SEC      = 180

_lock = threading.Lock()
_state = {
    "ok": 0, "fail": 0,
    "in_tokens": 0, "out_tokens": 0,
    "cache_hit": 0, "cache_miss": 0,
    "cost_usd": 0.0,
    "stopped": False,
}


# ---------- 工具 ----------
def log(msg: str):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    tqdm.write(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def estimate_cost(in_hit: int, in_miss: int, out: int) -> float:
    return (in_hit * PRICE_INPUT_HIT + in_miss * PRICE_INPUT_MISS + out * PRICE_OUTPUT) / 1_000_000


def load_done_ids() -> set[str]:
    if not os.path.exists(OUTPUT_PATH):
        return set()
    done = set()
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("ok"):
                    done.add(rec["paper_id"])
            except json.JSONDecodeError:
                continue
    return done


# ---------- API ----------
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


def call_with_retry(system: str, user: str, model: str, api_key: str) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "thinking": {"type": "disabled"},  # 唯一改动
                },
                timeout=900,
            )
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", RETRY_BASE_BACKOFF * attempt))
                time.sleep(wait)
                last_err = f"429 rate limit, retry {attempt}/{MAX_RETRIES}"
                continue
            if 500 <= r.status_code < 600:
                last_err = f"{r.status_code} server error: {r.text[:200]}"
                time.sleep(RETRY_BASE_BACKOFF * attempt)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(RETRY_BASE_BACKOFF * attempt)
        except Exception as e:
            last_err = f"unexpected {type(e).__name__}: {e}"
            time.sleep(RETRY_BASE_BACKOFF * attempt)
    raise RuntimeError(f"all retries failed: {last_err}")


def parse_content(content: str):
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    return json.loads(s)


def process_one(paper: dict, order_idx: int, system: str, model: str, api_key: str) -> dict:
    record = {
        "_order_idx": order_idx,
        "paper_id": paper["paper_id"],
        "title": paper.get("title", ""),
        "venue": paper.get("venue"),
        "year": paper.get("year"),
    }
    t0 = time.time()
    try:
        resp = call_with_retry(system, build_user_message(paper), model, api_key)
        msg = resp["choices"][0]["message"]
        usage = resp.get("usage", {})

        content = (msg.get("content", "") or "").strip()

        try:
            parsed = parse_content(content)
            ok, err = True, None
        except Exception as e:
            parsed, ok, err = None, False, f"JSON parse failed: {e}"

        record.update({
            "elapsed_s": round(time.time() - t0, 1),
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            "tokens_cache_hit": usage.get("prompt_cache_hit_tokens"),
            "tokens_cache_miss": usage.get("prompt_cache_miss_tokens"),
            "raw_content": content,
            "parsed": parsed,
            "ok": ok,
            "error": err,
        })
    except Exception as e:
        record.update({
            "ok": False,
            "elapsed_s": round(time.time() - t0, 1),
            "error": repr(e)[:500],
        })
    return record


def append_jsonl(path: str, record: dict):
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_state(rec: dict):
    with _lock:
        if rec.get("ok"):
            _state["ok"] += 1
            _state["in_tokens"] += rec.get("tokens_in") or 0
            _state["out_tokens"] += rec.get("tokens_out") or 0
            hit  = rec.get("tokens_cache_hit") or 0
            miss = rec.get("tokens_cache_miss") or 0
            _state["cache_hit"]  += hit
            _state["cache_miss"] += miss
            _state["cost_usd"]   += estimate_cost(hit, miss, rec.get("tokens_out") or 0)
        else:
            _state["fail"] += 1


def heartbeat_loop(total: int, t_start: float, max_cost: float):
    while not _state["stopped"]:
        time.sleep(HEARTBEAT_EVERY)
        if _state["stopped"]:
            break
        with _lock:
            done    = _state["ok"] + _state["fail"]
            elapsed = time.time() - t_start
            rate    = done / max(elapsed, 1)
            eta_sec = (total - done) / max(rate, 0.001)
            err_rate  = _state["fail"] / max(done, 1)
            cache_pct = _state["cache_hit"] / max(_state["cache_hit"] + _state["cache_miss"], 1) * 100
            log(
                f"[心跳] done={done}/{total}({done/total*100:.1f}%)  "
                f"ok={_state['ok']}  fail={_state['fail']}({err_rate*100:.1f}%)  "
                f"rate={rate:.2f}/s  ETA={timedelta(seconds=int(eta_sec))}  "
                f"cache={cache_pct:.1f}%  cost=${_state['cost_usd']:.2f}"
            )
            if done >= 30 and err_rate > ERROR_RATE_THRESHOLD:
                log(f"错误率 {err_rate*100:.1f}% 超过阈值,暂停 {ERROR_PAUSE_SEC}s")
                time.sleep(ERROR_PAUSE_SEC)
            if _state["cost_usd"] >= max_cost:
                log(f"成本达到上限 ${max_cost},触发 stop")
                _state["stopped"] = True


# ---------- 主流程 ----------
def main():
    n        = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers  = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    model    = sys.argv[3] if len(sys.argv) > 3 else "deepseek-v4-pro"
    max_cost = float(os.environ.get("MAX_COST_USD", "50"))

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY not set")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    system_prompt  = Path(PROMPT_PATH).read_text(encoding="utf-8")
    sample_payload = json.loads(Path(SAMPLE_PATH).read_text(encoding="utf-8"))
    all_samples    = sample_payload["samples"][:n]

    log("=== 启动 C1-nothink 打分 ===")
    log(f"样本: {SAMPLE_PATH}  N={len(all_samples)}  模型={model}  并发={workers}")
    log(f"prompt={len(system_prompt)} 字符")

    done_ids     = load_done_ids()
    pid_to_order = {p["paper_id"]: i for i, p in enumerate(all_samples)}
    todo         = [p for p in all_samples if p["paper_id"] not in done_ids]
    log(f"已完成 {len(done_ids)} 篇,待处理 {len(todo)} 篇")

    if not todo:
        log("无新任务,退出。")
        return

    def handle_sigint(signum, frame):
        log("收到 SIGINT,标记停止...")
        _state["stopped"] = True
    signal.signal(signal.SIGINT, handle_sigint)

    t_start = time.time()
    hb = threading.Thread(
        target=heartbeat_loop, args=(len(todo), t_start, max_cost), daemon=True
    )
    hb.start()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures: dict[Future, dict] = {}
        for p in todo:
            if _state["stopped"]:
                break
            futures[ex.submit(
                process_one, p, pid_to_order[p["paper_id"]], system_prompt, model, api_key
            )] = p

        with tqdm(total=len(futures), desc="C1-nothink", unit="篇", smoothing=0.1) as pbar:
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {
                        "paper_id": futures[fut]["paper_id"],
                        "_order_idx": pid_to_order[futures[fut]["paper_id"]],
                        "ok": False,
                        "error": repr(e)[:500],
                    }
                if rec.get("ok"):
                    append_jsonl(OUTPUT_PATH, rec)
                else:
                    append_jsonl(FAILURE_PATH, rec)
                update_state(rec)
                pbar.set_postfix(ok=_state["ok"], fail=_state["fail"],
                                 cost=f"${_state['cost_usd']:.3f}")
                pbar.update(1)

    _state["stopped"] = True
    elapsed = time.time() - t_start
    log(f"=== 完成 ===  用时 {timedelta(seconds=int(elapsed))}  ok={_state['ok']}  fail={_state['fail']}")
    log(f"in={_state['in_tokens']:,}  out={_state['out_tokens']:,}  cost=${_state['cost_usd']:.4f}")

    if os.path.exists(OUTPUT_PATH):
        log("按原始顺序重排输出 ...")
        records = []
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        records.sort(key=lambda r: r.get("_order_idx", 10**9))
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        ok_recs = [r for r in records if r.get("ok")]
        tout = [r["tokens_out"] for r in ok_recs if r.get("tokens_out")]
        print(f"\n{'='*50}")
        print(f"C1-nothink 结果: {len(ok_recs)}/{len(records)} ok")
        if tout:
            print(f"output tokens 均值: {sum(tout)//len(tout)}")
        log(f"输出 {len(records)} 条 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
