#!/usr/bin/env python3
"""C3 v0.8 全量打分脚本 — 生产版

特性:
  - 全量 / 指定 N 篇，自动续跑（跳过已完成 paper_id）
  - ThreadPoolExecutor 并发，可配置 workers
  - 指数退避重试（应对 429 / 5xx / 网络抖动）
  - Rich 实时仪表盘（适配 tmux，每秒刷新）
  - SIGINT / SIGTERM 优雅退出，已完成数据不丢
  - C3 schema 全量校验 + fuse 机械验证
  - 运行结束按原始顺序重排输出

用法:
  python run_c3_full.py                        # 全量，默认 workers=32
  python run_c3_full.py --n 200 --workers 16   # 前 200 篇，16 并发
  python run_c3_full.py --model deepseek-v4-pro

必要文件（与本脚本同目录）:
  prompt_c3.md              评分 prompt
  data/input.json           输入样本，格式同 sample_N600_seed42.json

环境变量:
  DEEPSEEK_API_KEY          必填（或在同目录 .env 文件中）

输出:
  data/output.jsonl         实时追加，中断可续跑
  data/run.log              结构化日志
"""

import argparse
import json
import logging
import os
import signal
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

# ─── 路径 ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "prompt_c3.md"
INPUT_PATH  = ROOT / "data" / "input.json"
OUTPUT_PATH = ROOT / "data" / "output.jsonl"
LOG_PATH    = ROOT / "data" / "run.log"
API_URL     = "https://api.deepseek.com/v1/chat/completions"

# ─── C3 schema ───────────────────────────────────────────────────────────────
SCORE_FIELDS = ("mr", "tn", "md", "ar", "er", "tea", "cc", "ei", "sg", "cs")
DIM_NAMES    = {1: "mr", 2: "tn", 3: "md", 4: "ar", 5: "er"}

# ─── 全局状态 ─────────────────────────────────────────────────────────────────
_write_lock    = threading.Lock()
_stats_lock    = threading.Lock()
_shutdown_flag = threading.Event()

_stats = {
    "ok": 0, "fail": 0,
    "fuse_true": 0, "fuse_false": 0,
    "tokens_out_compact": [], "tokens_out_full": [],
    "recent": [],          # 最近完成的记录（最多 10 条）
    "errors": [],          # 最近失败（最多 5 条）
}


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8")],
    )


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


def call_api(system: str, user: str, model: str, api_key: str,
             retry_max: int = 4, base_delay: float = 2.0) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(retry_max):
        if _shutdown_flag.is_set():
            raise RuntimeError("shutdown")
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if r.status_code == 429:
                delay = base_delay * (3 ** attempt)
                logging.warning("429 rate-limited, retry %d after %.0fs", attempt + 1, delay)
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.json()
        except requests.Timeout:
            delay = base_delay * (2 ** attempt)
            logging.warning("Timeout attempt %d, retry after %.0fs", attempt + 1, delay)
            if attempt < retry_max - 1:
                time.sleep(delay)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code >= 500:
                delay = base_delay * (2 ** attempt)
                logging.warning("HTTP %d attempt %d, retry after %.0fs",
                                e.response.status_code, attempt + 1, delay)
                if attempt < retry_max - 1:
                    time.sleep(delay)
                    continue
            raise
    raise RuntimeError(f"API failed after {retry_max} attempts")


def parse_content(content: str) -> dict:
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    return json.loads(s)


def validate_c3(parsed: dict) -> tuple[bool, list[str]]:
    errors = []

    if parsed.get("pv") != "v0.8":
        errors.append(f"pv={parsed.get('pv')!r} != v0.8")

    ig   = parsed.get("ig")
    fuse = parsed.get("fuse")
    rr   = parsed.get("rr")

    if ig not in ("intact", "partial", "broken", "absent"):
        errors.append(f"ig={ig!r} 无效")

    if ig == "absent" and fuse is not True:
        errors.append(f"ig=absent 但 fuse={fuse}")

    if not isinstance(fuse, bool):
        errors.append(f"fuse={fuse!r} 不是 bool")

    if not isinstance(rr, list):
        errors.append(f"rr={rr!r} 不是 list")
    else:
        valid = sorted({x for x in rr if isinstance(x, int) and 1 <= x <= 5})
        if len(valid) >= 2:
            parsed["rr"] = valid[:2]
        elif len(rr) == 0:
            parsed["rr"] = []
        else:
            scores = {i: parsed.get(k) for i, k in enumerate(["mr","tn","md","ar","er"], 1)}
            valid_scores = {i: v for i, v in scores.items() if isinstance(v, (int, float))}
            if len(valid_scores) >= 2:
                parsed["rr"] = sorted(sorted(valid_scores, key=valid_scores.__getitem__)[:2])
            else:
                parsed["rr"] = []

    if fuse is True:
        extra = [k for k in parsed if k.endswith("_r") or k in ("lc_p","lc_t","lc_d","lc_c","osn")]
        if extra:
            errors.append(f"fuse=true 但存在 rationale 字段: {extra[:3]}")

    if fuse is False:
        required = ["mr_r","tn_r","md_r","ar_r","er_r","cc_r","ei_r","sg_r","cs_r",
                    "lc_p","lc_t","lc_d","lc_c","mk_r","hr_r","osn"]
        missing = [f for f in required if f not in parsed]
        if missing:
            errors.append(f"fuse=false 缺少字段: {missing[:5]}")

    return len(errors) == 0, errors


def verify_fuse(parsed: dict) -> tuple[bool, str]:
    ig, fuse = parsed.get("ig"), parsed.get("fuse")
    expected = (ig == "absent")
    if fuse != expected:
        return False, f"ig={ig} → expected fuse={expected}, got {fuse}"
    return True, "ok"


# ─── 单篇处理 ────────────────────────────────────────────────────────────────

def process_one(paper: dict, system: str, model: str, api_key: str, order_idx: int) -> dict:
    pid = paper["paper_id"]
    record: dict = {"_order_idx": order_idx, "paper_id": pid,
                    "title": paper.get("title",""), "venue": paper.get("venue"),
                    "year": paper.get("year")}
    t0 = time.time()
    try:
        resp    = call_api(system, build_user_message(paper), model, api_key)
        msg     = resp["choices"][0]["message"]
        usage   = resp.get("usage", {})
        elapsed = time.time() - t0
        content = (msg.get("content","") or "").strip()

        try:
            parsed = parse_content(content)
            schema_ok, schema_errs = validate_c3(parsed)
            fuse_ok, fuse_msg      = verify_fuse(parsed)
            ok  = schema_ok and fuse_ok
            err = ("; ".join(schema_errs) if not schema_ok
                   else (None if fuse_ok else fuse_msg))
        except Exception as e:
            parsed = None; schema_ok = fuse_ok = ok = False
            schema_errs = []; err = f"JSON parse failed: {e}"

        record.update({
            "elapsed_s": round(elapsed, 1),
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            "content": content, "parsed": parsed,
            "ok": ok, "schema_ok": schema_ok, "fuse_ok": fuse_ok, "error": err,
        })

    except Exception as e:
        record.update({
            "ok": False, "schema_ok": False, "fuse_ok": False,
            "elapsed_s": round(time.time() - t0, 1), "error": repr(e),
        })

    return record


# ─── I/O ─────────────────────────────────────────────────────────────────────

def append_jsonl(record: dict):
    with _write_lock:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed() -> set[str]:
    if not OUTPUT_PATH.exists():
        return set()
    done = set()
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("ok"):
                    done.add(r["paper_id"])
            except Exception:
                pass
    return done


def sort_output():
    records = []
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            try: records.append(json.loads(line))
            except Exception: pass
    records.sort(key=lambda r: r.get("_order_idx", 10**9))
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ─── Rich 仪表盘 ──────────────────────────────────────────────────────────────

def make_dashboard(progress: Progress, total: int, t_start: float) -> Table:
    with _stats_lock:
        ok        = _stats["ok"]
        fail      = _stats["fail"]
        ft        = _stats["fuse_true"]
        ff        = _stats["fuse_false"]
        tok_c     = _stats["tokens_out_compact"]
        tok_f     = _stats["tokens_out_full"]
        recent    = list(_stats["recent"])
        errors    = list(_stats["errors"])

    done     = ok + fail
    fuse_pct = f"{ft/(ok or 1)*100:.1f}%" if ok else "-"
    avg_c    = f"{int(statistics.mean(tok_c))}" if tok_c else "-"
    avg_f    = f"{int(statistics.mean(tok_f))}" if tok_f else "-"

    # ── 统计行 ──
    stat = Table.grid(padding=(0, 2))
    stat.add_column(style="bold cyan")
    stat.add_column(style="green")
    stat.add_column(style="bold cyan")
    stat.add_column(style="yellow")
    stat.add_column(style="bold cyan")
    stat.add_column()
    stat.add_column(style="bold cyan")
    stat.add_column()
    stat.add_column(style="bold cyan")
    stat.add_column()
    stat.add_row(
        "OK", str(ok),
        "FAIL", str(fail),
        "FUSE%", fuse_pct,
        "TOK紧凑", avg_c,
        "TOK完整", avg_f,
    )

    # ── 最近完成 ──
    rec_table = Table(title="最近完成", box=None, header_style="bold magenta",
                      show_edge=False, padding=(0,1))
    for col, style in [("paper_id",""), ("ig","cyan"), ("fuse",""),
                       ("mr",""), ("tn",""), ("cs",""), ("s","dim")]:
        rec_table.add_column(col, style=style, no_wrap=True)
    for r in recent[-8:]:
        p = r.get("parsed") or {}
        rec_table.add_row(
            r["paper_id"][:12],
            p.get("ig","")[:3],
            "T" if p.get("fuse") else "F",
            str(p.get("mr","-")),
            str(p.get("tn","-")),
            str(p.get("cs","-")),
            f"{r.get('elapsed_s',0):.1f}s",
        )

    # ── 最近失败 ──
    err_table = Table(title="近期失败", box=None, header_style="bold red",
                      show_edge=False, padding=(0,1))
    err_table.add_column("paper_id"); err_table.add_column("error", no_wrap=False)
    for e in errors[-4:]:
        err_table.add_row(e["paper_id"][:12], str(e.get("error",""))[:60])

    outer = Table.grid(padding=1)
    outer.add_row(progress)
    outer.add_row(stat)
    outer.add_row(Columns([rec_table, err_table]))
    return Panel(outer, title=f"[bold]C3 v0.8 全量打分[/bold]  完成 {done}/{total}",
                 border_style="blue")


# ─── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int,   default=None,           help="处理前 N 篇（默认全量）")
    parser.add_argument("--workers", type=int,   default=128,            help="并发线程数")
    parser.add_argument("--model",   type=str,   default="deepseek-v4-pro")
    parser.add_argument("--input",   type=str,   default=str(INPUT_PATH))
    args = parser.parse_args()

    load_env()
    setup_logging()

    api_key = os.environ.get("DEEPSEEK_API_KEY","")
    if not api_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY not set")

    system  = PROMPT_PATH.read_text(encoding="utf-8")
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    samples = payload["samples"]
    if args.n:
        samples = samples[:args.n]
    total = len(samples)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed    = load_completed()
    todo         = [p for p in samples if p["paper_id"] not in completed]
    pid_to_order = {p["paper_id"]: i for i, p in enumerate(samples)}

    console = Console()
    console.print(f"[bold]C3 v0.8[/bold]  模型: {args.model}  并发: {args.workers}  "
                  f"prompt: {len(system)} chars")
    console.print(f"总计: {total}  已完成: {len(completed)}  待处理: {len(todo)}")

    if not todo:
        console.print("[green]无新任务，直接输出摘要。[/green]")
        print_summary(console, total)
        return

    # ── 信号处理 ──
    def _on_signal(sig, frame):
        console.print("\n[yellow]收到退出信号，等待当前任务完成后退出...[/yellow]")
        _shutdown_flag.set()
    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ── Warmup：先跑第一篇，验证全链路 ──
    console.print("\n[bold yellow]▶ Warmup[/bold yellow] 发送第一篇验证管道...")
    warmup_paper = todo[0]
    warmup_rec   = process_one(warmup_paper, system, args.model, api_key,
                               pid_to_order[warmup_paper["paper_id"]])
    append_jsonl(warmup_rec)

    if not warmup_rec.get("ok"):
        console.print(f"[bold red]✗ Warmup 失败，终止。[/bold red]")
        console.print(f"  paper_id : {warmup_rec['paper_id']}")
        console.print(f"  error    : {warmup_rec.get('error','')}")
        sys.exit(1)

    wp = warmup_rec.get("parsed", {})
    console.print(
        f"[bold green]✓ Warmup 通过[/bold green]  "
        f"ig=[cyan]{wp.get('ig','')}[/cyan]  "
        f"fuse=[cyan]{wp.get('fuse')}[/cyan]  "
        f"mr={wp.get('mr','-')}  tn={wp.get('tn','-')}  "
        f"tok_out={warmup_rec.get('tokens_out','-')}  "
        f"elapsed={warmup_rec.get('elapsed_s','-')}s"
    )
    console.print(f"[dim]管道验证通过，启动 {args.workers} 并发...[/dim]\n")

    # warmup 篇已完成，从剩余列表继续
    todo = todo[1:]
    with _stats_lock:
        _stats["ok"] += 1
        if wp.get("fuse") is True:
            _stats["fuse_true"] += 1
            if warmup_rec.get("tokens_out"):
                _stats["tokens_out_compact"].append(warmup_rec["tokens_out"])
        else:
            _stats["fuse_false"] += 1
            if warmup_rec.get("tokens_out"):
                _stats["tokens_out_full"].append(warmup_rec["tokens_out"])
        _stats["recent"].append(warmup_rec)

    t_start = time.time()
    progress = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        "[progress.percentage]{task.percentage:>5.1f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        refresh_per_second=2,
    )
    task_id = progress.add_task("打分中", total=len(todo))  # todo 已去掉 warmup 篇

    with Live(make_dashboard(progress, total, t_start),
              console=console, refresh_per_second=2) as live:

        def update_live():
            live.update(make_dashboard(progress, total, t_start))

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(process_one, p, system, args.model, api_key,
                          pid_to_order[p["paper_id"]]): p
                for p in todo
            }
            for fut in as_completed(futures):
                if _shutdown_flag.is_set():
                    for f in futures:
                        f.cancel()
                    break

                rec = fut.result()
                append_jsonl(rec)
                progress.advance(task_id)

                p = rec.get("parsed") or {}
                with _stats_lock:
                    if rec.get("ok"):
                        _stats["ok"] += 1
                        if p.get("fuse") is True:
                            _stats["fuse_true"] += 1
                            if rec.get("tokens_out"):
                                _stats["tokens_out_compact"].append(rec["tokens_out"])
                        else:
                            _stats["fuse_false"] += 1
                            if rec.get("tokens_out"):
                                _stats["tokens_out_full"].append(rec["tokens_out"])
                        _stats["recent"].append(rec)
                        if len(_stats["recent"]) > 20:
                            _stats["recent"].pop(0)
                    else:
                        _stats["fail"] += 1
                        logging.error("FAIL %s: %s", rec["paper_id"], rec.get("error",""))
                        _stats["errors"].append(rec)
                        if len(_stats["errors"]) > 10:
                            _stats["errors"].pop(0)

                update_live()

    elapsed = time.time() - t_start
    console.print(f"\n[bold]完成[/bold]  ok={_stats['ok']}  fail={_stats['fail']}  "
                  f"耗时 {elapsed/60:.1f}min")

    console.print("排序输出...", end=" ")
    sort_output()
    console.print("done")

    print_summary(console, total)
    console.print(f"\n→ {OUTPUT_PATH}")
    logging.info("完成 ok=%d fail=%d elapsed=%.0fs", _stats["ok"], _stats["fail"], elapsed)


def print_summary(console: Console, total: int):
    if not OUTPUT_PATH.exists():
        return
    recs    = []
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            try: recs.append(json.loads(line))
            except Exception: pass

    ok_recs = [r for r in recs if r.get("ok")]
    if not ok_recs:
        console.print("[red]无成功记录[/red]")
        return

    ft = sum(1 for r in ok_recs if r.get("parsed",{}).get("fuse") is True)
    ff = sum(1 for r in ok_recs if r.get("parsed",{}).get("fuse") is False)
    n  = len(ok_recs)

    t = Table(title=f"C3 结果摘要  {n}/{total} 成功", box=None, header_style="bold")
    t.add_column("指标"); t.add_column("值", justify="right")
    t.add_row("fuse=true  (紧凑)", f"{ft} ({ft/n*100:.1f}%)")
    t.add_row("fuse=false (完整)", f"{ff} ({ff/n*100:.1f}%)")

    tok_c = [r["tokens_out"] for r in ok_recs if r.get("parsed",{}).get("fuse") is True  and r.get("tokens_out")]
    tok_f = [r["tokens_out"] for r in ok_recs if r.get("parsed",{}).get("fuse") is False and r.get("tokens_out")]
    if tok_c: t.add_row("output tokens 紧凑均值", str(int(statistics.mean(tok_c))))
    if tok_f: t.add_row("output tokens 完整均值", str(int(statistics.mean(tok_f))))

    console.print(t)

    score_t = Table(title="分数分布", box=None, header_style="bold")
    score_t.add_column("维度"); score_t.add_column("均值", justify="right"); score_t.add_column("n", justify="right")
    for field in ("mr","tn","md","ar","er","cs"):
        vals = [r["parsed"][field] for r in ok_recs if r.get("parsed") and r["parsed"].get(field) is not None]
        if vals:
            score_t.add_row(field, f"{statistics.mean(vals):.2f}", str(len(vals)))
    console.print(score_t)


if __name__ == "__main__":
    main()
