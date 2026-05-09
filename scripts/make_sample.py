#!/usr/bin/env python3
"""从全量论文元数据中抽取样本（可溯源）

用法:
  uv run scripts/make_sample.py [N] [--seed SEED] [--source DIR] [--out PATH]

参数:
  N           抽取篇数，默认 20
  --seed      随机种子，默认 42
  --source    原始元数据目录，默认 data/raw
  --out       输出路径，默认 data/sample_N{N}_seed{seed}.json

输出 JSON 格式与 sample_20.json 兼容，顶层包含可溯源字段:
  source_dir, source_files, seed, n, total_pool, created_at, samples
"""
import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def extract_year(venue: str) -> int | None:
    m = re.search(r"(\d{4})", venue or "")
    return int(m.group(1)) if m else None


def load_pool(source_dir: Path) -> tuple[list[dict], list[str]]:
    files = sorted(source_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"ERROR: {source_dir} 下没有找到 .jsonl 文件")

    pool: dict[str, dict] = {}
    for fpath in files:
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("paper_id")
                if pid and pid not in pool:
                    pool[pid] = rec

    source_files = [f.name for f in files]
    return list(pool.values()), source_files


def to_sample(rec: dict) -> dict:
    return {
        "paper_id": rec.get("paper_id", ""),
        "venue": rec.get("venue", ""),
        "year": rec.get("year") or extract_year(rec.get("venue", "")),
        "title": rec.get("title", ""),
        "authors": rec.get("authors", ""),
        "abstract": rec.get("abstract", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    n, seed = args.n, args.seed
    source_dir = args.source
    out_path = args.out or Path(f"data/sample_N{n}_seed{seed}.json")

    pool, source_files = load_pool(source_dir)
    total_pool = len(pool)

    if n > total_pool:
        sys.exit(f"ERROR: 请求 {n} 篇，但池子只有 {total_pool} 篇")

    rng = random.Random(seed)
    chosen = rng.sample(pool, n)
    samples = [to_sample(r) for r in chosen]

    output = {
        "source_dir": str(source_dir),
        "source_files": source_files,
        "seed": seed,
        "n": n,
        "total_pool": total_pool,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": samples,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"池子总量 : {total_pool:,} 篇  (来自 {len(source_files)} 个文件)")
    print(f"抽取数量 : {n}  种子: {seed}")
    print(f"输出路径 : {out_path}")


if __name__ == "__main__":
    main()
