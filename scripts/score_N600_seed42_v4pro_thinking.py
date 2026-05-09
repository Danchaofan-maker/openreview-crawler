#!/usr/bin/env python3
"""data/sample_N600_seed42.json 专用打分脚本 — DeepSeek V4 Pro + thinking

用法:
  uv run scripts/score_N600_seed42_v4pro_thinking.py [workers]

默认:
  workers=128, model=deepseek-v4-pro

特性:
  - 固定读取 data/sample_N600_seed42.json(600 篇,seed=42 均匀随机样本)
  - 实时落盘到 data/llm_v4pro_thinking_N600_seed42.jsonl(append + 锁,中断不丢)
  - 失败记录单独写入 data/llm_v4pro_thinking_N600_seed42_failures.jsonl
  - 续跑:启动时跳过已完成的 paper_id
  - 429 / 5xx / 超时 指数退避重试 3 次
  - 每分钟心跳:进度、错误率、token 用量、估算成本、ETA
  - 错误率 > 10% 自动暂停 3 分钟
  - SIGINT 优雅停止
  - 完成后按抽样原始顺序重排输出文件

环境变量:
  DEEPSEEK_API_KEY  必填
  MAX_COST_USD      软上限,默认 $50,超过自动停
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
PROMPT_PATH   = "prompt_c1t.md"
SAMPLE_PATH   = "data/sample_N600_seed42.json"
OUTPUT_PATH   = "data/llm_v4pro_thinking_N600_seed42.jsonl"
FAILURE_PATH  = "data/llm_v4pro_thinking_N600_seed42_failures.jsonl"
LOG_PATH      = "data/llm_v4pro_thinking_N600_seed42.log"
API_URL       = "https://api.deepseek.com/v1/chat/completions"

# 价格(v4-pro, USD per 1M tokens)
PRICE_INPUT_HIT  = 0.003625
PRICE_INPUT_MISS = 0.435
PRICE_OUTPUT     = 0.87

MAX_RETRIES            = 3
RETRY_BASE_BACKOFF     = 5    # 秒
HEARTBEAT_EVERY        = 60   # 秒
ERROR_RATE_THRESHOLD   = 0.10
ERROR_PAUSE_SEC        = 180

# 全局状态(锁保护)
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
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": "high",
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

        reasoning = msg.get("reasoning_content", "") or ""
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
            "reasoning_chars": len(reasoning),
            "reasoning": reasoning,
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
    workers  = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    model    = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4-pro"
    max_cost = float(os.environ.get("MAX_COST_USD", "50"))

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY not set")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    system_prompt  = Path(PROMPT_PATH).read_text(encoding="utf-8")
    sample_payload = json.loads(Path(SAMPLE_PATH).read_text(encoding="utf-8"))
    all_samples    = sample_payload["samples"]   # 全部 600 篇

    log("=== 启动 N600_seed42 v4pro+thinking 打分 ===")
    log(f"样本来源: {SAMPLE_PATH}  共 {len(all_samples)} 篇")
    log(f"模型={model}  并发={workers}  max_cost=${max_cost}")
    log(f"prompt={len(system_prompt)} 字符")

    done_ids = load_done_ids()
    pid_to_order = {p["paper_id"]: i for i, p in enumerate(all_samples)}
    todo = [p for p in all_samples if p["paper_id"] not in done_ids]
    log(f"已完成 {len(done_ids)} 篇,本次待处理 {len(todo)} 篇")

    if not todo:
        log("无新任务,退出。")
        return

    def handle_sigint(signum, frame):
        log("收到 SIGINT,标记停止(已 dispatch 的请求会跑完)...")
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

        with tqdm(total=len(futures), desc="打分中", unit="篇", smoothing=0.1) as pbar:
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
                pbar.set_postfix(
                    ok=_state["ok"], fail=_state["fail"],
                    cost=f"${_state['cost_usd']:.2f}",
                )
                pbar.update(1)

    _state["stopped"] = True
    elapsed = time.time() - t_start
    log(f"=== 完成 ===")
    log(f"总用时 {timedelta(seconds=int(elapsed))}  ok={_state['ok']}  fail={_state['fail']}")
    log(f"in={_state['in_tokens']:,}  out={_state['out_tokens']:,}  cost=${_state['cost_usd']:.4f}")
    cache_pct = _state["cache_hit"] / max(_state["cache_hit"] + _state["cache_miss"], 1) * 100
    log(f"cache_hit={cache_pct:.1f}%")

    # 按抽样原始顺序重排成功输出
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
        log(f"输出 {len(records)} 条 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
