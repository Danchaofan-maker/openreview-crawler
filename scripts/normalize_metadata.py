#!/usr/bin/env python3
"""
全量数据集 schema 归一化(原地改写)
- 补齐 year:OpenReview 会议从 venue 末尾 `_(\\d{4})` 提取;TMLR 用 pdate[:4]
- 移除 pdate 字段
- 缺失 doi 时补 null

作用域:data/raw/*.jsonl 与 data/excluded/*.jsonl
"""
import json
import os
import re
import sys

DIRS = ["data/raw", "data/excluded"]
VENUE_YEAR_RE = re.compile(r"_(\d{4})$")


def normalize(row: dict) -> dict:
    if row.get("year") in (None, ""):
        pdate = row.get("pdate")
        if pdate and isinstance(pdate, str) and len(pdate) >= 4 and pdate[:4].isdigit():
            row["year"] = int(pdate[:4])
        else:
            m = VENUE_YEAR_RE.search(row.get("venue") or "")
            if m:
                row["year"] = int(m.group(1))

    row.pop("pdate", None)

    if "doi" not in row:
        row["doi"] = None

    return row


def process_file(path: str) -> tuple[int, int, int]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    year_filled = sum(1 for r in rows if r.get("year") in (None, ""))
    pdate_removed = sum(1 for r in rows if "pdate" in r)
    doi_added = sum(1 for r in rows if "doi" not in r)

    rows = [normalize(r) for r in rows]

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

    still_missing_year = sum(1 for r in rows if r.get("year") in (None, ""))
    return len(rows), year_filled - still_missing_year, pdate_removed


def main():
    total_rows = total_year = total_pdate = 0
    for d in DIRS:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".jsonl"):
                continue
            path = os.path.join(d, fname)
            n, y, p = process_file(path)
            total_rows += n
            total_year += y
            total_pdate += p
            print(f"  {path:<55} rows={n:>6}  +year={y:>6}  -pdate={p:>5}", flush=True)

    print(f"\n合计 rows={total_rows}  year填补={total_year}  pdate删除={total_pdate}")


if __name__ == "__main__":
    main()
