#!/usr/bin/env python3
"""
期刊论文元数据爬虫 — 通过 OpenAlex API 抓取 JMLR / SIMODS / SIOPT
只抓 2024 年起发表的论文，边抓边写，遇到错误自动重试。
"""
import requests
import json
import time
import sys
from datetime import datetime

OUTPUT_DIR = "data/raw"
EMAIL = "danchaofan554@gmail.com"  # OpenAlex polite pool
BASE_URL = "https://api.openalex.org/works"
FROM_DATE = "2024-01-01"
PER_PAGE = 200

JOURNALS = {
    "jmlr":   {"source_id": "S118988714",  "name": "Journal of Machine Learning Research"},
    "simods": {"source_id": "S4210229561", "name": "SIAM Journal on Mathematics of Data Science"},
    "siopt":  {"source_id": "S928796702",  "name": "SIAM Journal on Optimization"},
}


def fetch_page(source_id, cursor="*", retries=5):
    params = {
        "filter": f"primary_location.source.id:{source_id},from_publication_date:{FROM_DATE}",
        "select": "id,title,abstract_inverted_index,authorships,publication_year,ids,doi",
        "per-page": PER_PAGE,
        "cursor": cursor,
        "mailto": EMAIL,
    }
    for attempt in range(retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"  [429] 限速，等待 {wait}s", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            print(f"  [超时] 第 {attempt+1} 次，重试...", flush=True)
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            print(f"  [错误] {e}，第 {attempt+1} 次，重试...", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def extract_abstract(inv_index):
    if not inv_index:
        return ""
    positions = {}
    for word, locs in inv_index.items():
        for pos in locs:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions))


def crawl_journal(key, source_id, display_name):
    out_path = f"{OUTPUT_DIR}/{key}_metadata.jsonl"
    print(f"\n{'='*60}", flush=True)
    print(f"[{key.upper()}] {display_name}", flush=True)
    print(f"  过滤：{FROM_DATE} 起", flush=True)

    cursor = "*"
    total_written = 0
    page_num = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        while True:
            page_num += 1
            data = fetch_page(source_id, cursor)
            if data is None:
                print(f"  [中止] 连续失败，已写入 {total_written} 篇", flush=True)
                break

            meta = data.get("meta", {})
            results = data.get("results", [])
            total_count = meta.get("count", "?")

            if page_num == 1:
                print(f"  OpenAlex 报告总量: {total_count} 篇", flush=True)

            if not results:
                break

            for work in results:
                authors = [
                    a["author"]["display_name"]
                    for a in work.get("authorships", [])
                    if a.get("author", {}).get("display_name")
                ]
                ids = work.get("ids", {})
                arxiv_raw = ids.get("arxiv", "")
                arxiv_id = arxiv_raw.replace("https://arxiv.org/abs/", "").strip() if arxiv_raw else None

                paper = {
                    "paper_id": work.get("id", "").replace("https://openalex.org/", ""),
                    "arxiv_id": arxiv_id,
                    "venue": key.upper(),
                    "decision_track": None,
                    "title": work.get("title", ""),
                    "abstract": extract_abstract(work.get("abstract_inverted_index")),
                    "authors": "; ".join(authors),
                    "year": work.get("publication_year"),
                    "doi": work.get("doi", ""),
                    "crawled_at": datetime.now().isoformat(),
                }
                fout.write(json.dumps(paper, ensure_ascii=False) + "\n")
                total_written += 1

            fout.flush()
            print(f"  第 {page_num} 页，本页 {len(results)} 篇，累计 {total_written} 篇", flush=True)

            next_cursor = meta.get("next_cursor")
            if not next_cursor:
                break
            cursor = next_cursor
            time.sleep(0.15)

    print(f"  完成：{total_written} 篇 → {out_path}", flush=True)
    return total_written


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(JOURNALS.keys())
    print(f"目标期刊: {targets}", flush=True)

    for key in targets:
        if key not in JOURNALS:
            print(f"未知期刊: {key}", flush=True)
            continue
        j = JOURNALS[key]
        crawl_journal(key, j["source_id"], j["name"])
        time.sleep(1)

    print(f"\n{'='*60}")
    print("[全部完成]")


if __name__ == "__main__":
    main()
