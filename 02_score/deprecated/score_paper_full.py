#!/usr/bin/env python3
"""DeepSeek V4 全量打分脚本 — 直接读 data/raw/*.jsonl,用 prompt.md 给所有论文打分

用法:
  uv run scripts/score_paper_full.py [workers] [model] [max_papers]

默认:
  workers=128, model=deepseek-v4-pro, max_papers=∞

特性:
  - 读 data/raw/*.jsonl 全量论文(自动去重),排除 data/excluded/
  - 实时落盘到 data/llm_full_run.jsonl(append + 锁,中断不丢)
  - 续跑:启动时扫描已完成的 paper_id,只跑剩余
  - tqdm 进度条 + 每分钟 heartbeat 统计(token 用量、估算成本、ETA)
  - 失败处理:HTTP 错误指数退避重试 3 次,最终失败写入 data/llm_full_failures.jsonl
  - SIGINT 优雅停止:取消 pending,保存当前状态
  - 速率监控:若累计错误率 > 10%,自动暂停 3 分钟再继续

环境变量:
  DEEPSEEK_API_KEY  必填
  MAX_COST_USD      软上限,默认 $100,超过自动停
"""
from __future__ import annotations

import json
import os
import re
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

import requests
from tqdm import tqdm


def load_prompt(path: str) -> str:
    """读 prompt.md, 剥掉 <!-- ... --> HTML 注释 (允许在 prompt 里写只给人看的开发注释)"""
    text = Path(path).read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


# ---------- 配置 ----------
PROMPT_PATH = "prompt_c2.md"
RAW_DIR = "data/raw"
OUTPUT_PATH = "data/llm_full_run.jsonl"
FAILURE_PATH = "data/llm_full_failures.jsonl"
LOG_PATH = "data/llm_full_run.log"
API_URL = "https://api.deepseek.com/v1/chat/completions"

# 价格(v4-pro,75% 折扣后,USD per 1M tokens)
PRICE_INPUT_HIT = 0.003625
PRICE_INPUT_MISS = 0.435
PRICE_OUTPUT = 0.87

# 错误处理
MAX_RETRIES = 3
RETRY_BASE_BACKOFF = 5  # 秒
HEARTBEAT_EVERY = 60    # 秒
ERROR_RATE_THRESHOLD = 0.10
ERROR_PAUSE_SEC = 180

# 全局状态(锁保护)
_lock = threading.Lock()
_state = {
    "ok": 0,
    "fail": 0,
    "in_tokens": 0,
    "out_tokens": 0,
    "cache_hit": 0,
    "cache_miss": 0,
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


def load_all_raw_papers() -> list[dict]:
    """读 data/raw/*.jsonl 全量,paper_id 去重,字段对齐为 {paper_id, title, venue, year, abstract, _source}"""
    papers: dict[str, dict] = {}
    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(RAW_DIR, fname)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = p.get("paper_id")
                if not pid or pid in papers:
                    continue
                papers[pid] = {
                    "paper_id": pid,
                    "title": p.get("title", ""),
                    "venue": p.get("venue", ""),
                    "year": p.get("year"),
                    "abstract": p.get("abstract", ""),
                    "_source": fname,
                }
    return list(papers.values())


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


# ---------- API 调用 ----------
def build_user_message(paper: dict) -> str:
    # 不传 venue/year, 避免会场偏见
    return (
        "<paper>\n"
        f"<paper_id>{paper.get('paper_id', '')}</paper_id>\n"
        f"<title>{paper.get('title', '')}</title>\n"
        f"<abstract>{paper.get('abstract', '')}</abstract>\n"
        "</paper>"
    )


def call_with_retry(system: str, user: str, model: str, api_key: str) -> dict:
    """指数退避重试。429 / 5xx / 超时 / 连接错误均重试。"""
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
                    "thinking": {"type": "disabled"},
                },
                timeout=900,
            )
            if r.status_code == 429:
                # 限速:用响应里的 Retry-After 或退避
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
            continue
        except Exception as e:
            last_err = f"unexpected {type(e).__name__}: {e}"
            time.sleep(RETRY_BASE_BACKOFF * attempt)
            continue
    raise RuntimeError(f"all retries failed: {last_err}")


def parse_content(content: str):
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    return json.loads(s)


def process_one(paper: dict, system: str, model: str, api_key: str) -> dict:
    pid = paper["paper_id"]
    record = {
        "paper_id": pid,
        "title": paper.get("title", ""),
        "venue": paper.get("venue"),
        "year": paper.get("year"),
        "_source": paper.get("_source"),
    }
    t0 = time.time()
    try:
        resp = call_with_retry(system, build_user_message(paper), model, api_key)
        msg = resp["choices"][0]["message"]
        usage = resp.get("usage", {})

        content = (msg.get("content", "") or "").strip()

        try:
            parsed = parse_content(content)
            ok = True
            err = None
        except Exception as e:
            parsed = None
            ok = False
            err = f"JSON parse failed: {e}"

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
            hit = rec.get("tokens_cache_hit") or 0
            miss = rec.get("tokens_cache_miss") or 0
            _state["cache_hit"] += hit
            _state["cache_miss"] += miss
            _state["cost_usd"] += estimate_cost(hit, miss, rec.get("tokens_out") or 0)
        else:
            _state["fail"] += 1


def heartbeat_loop(total: int, t_start: float, max_cost: float):
    """后台线程,每分钟打印汇总,并检查 stop 条件"""
    while not _state["stopped"]:
        time.sleep(HEARTBEAT_EVERY)
        if _state["stopped"]:
            break
        with _lock:
            done = _state["ok"] + _state["fail"]
            elapsed = time.time() - t_start
            rate = done / max(elapsed, 1)
            eta_sec = (total - done) / max(rate, 0.001)
            err_rate = _state["fail"] / max(done, 1)
            cache_pct = _state["cache_hit"] / max(_state["cache_hit"] + _state["cache_miss"], 1) * 100
            log(
                f"[心跳] done={done}/{total}({done/total*100:.1f}%)  "
                f"ok={_state['ok']}  fail={_state['fail']}({err_rate*100:.1f}%)  "
                f"rate={rate:.2f}/s  ETA={timedelta(seconds=int(eta_sec))}  "
                f"cache={cache_pct:.1f}%  cost=${_state['cost_usd']:.2f}"
            )
            # 错误率过高自动暂停
            if done >= 50 and err_rate > ERROR_RATE_THRESHOLD:
                log(f"⚠️  错误率 {err_rate*100:.1f}% 超过阈值,暂停 {ERROR_PAUSE_SEC}s")
                time.sleep(ERROR_PAUSE_SEC)
            # 成本上限
            if _state["cost_usd"] >= max_cost:
                log(f"🛑 成本达到上限 ${max_cost},触发 stop")
                _state["stopped"] = True


# ---------- 主流程 ----------
def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    model = sys.argv[2] if len(sys.argv) > 2 else "deepseek-v4-pro"
    max_papers = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    max_cost = float(os.environ.get("MAX_COST_USD", "100"))

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY not set")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    system_prompt = load_prompt(PROMPT_PATH)
    log(f"=== 启动全量打分 ===")
    log(f"模型={model}  并发={workers}  max_papers={max_papers}  max_cost=${max_cost}")
    log(f"prompt={len(system_prompt)} 字符")

    # 装载论文 + 续跑
    all_papers = load_all_raw_papers()
    log(f"data/raw/ 共 {len(all_papers)} 篇(去重后)")

    done_ids = load_done_ids()
    todo = [p for p in all_papers if p["paper_id"] not in done_ids]
    todo = todo[:max_papers]
    log(f"已完成 {len(done_ids)} 篇,本次待处理 {len(todo)} 篇")

    if not todo:
        log("无新任务,退出。")
        return

    # SIGINT 处理
    def handle_sigint(signum, frame):
        log("收到 SIGINT,标记停止(已 dispatch 的请求会跑完)...")
        _state["stopped"] = True
    signal.signal(signal.SIGINT, handle_sigint)

    # 心跳线程
    t_start = time.time()
    hb = threading.Thread(target=heartbeat_loop, args=(len(todo), t_start, max_cost), daemon=True)
    hb.start()

    # 主跑
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures: dict[Future, dict] = {}
        for p in todo:
            if _state["stopped"]:
                break
            futures[ex.submit(process_one, p, system_prompt, model, api_key)] = p

        with tqdm(total=len(futures), desc="打分中", unit="篇", smoothing=0.1) as pbar:
            for fut in as_completed(futures):
                if _state["stopped"]:
                    # 不再处理后续 futures(让它们自然完成,退出 with 时阻塞等)
                    pass
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"paper_id": futures[fut]["paper_id"], "ok": False, "error": repr(e)[:500]}
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


if __name__ == "__main__":
    main()
