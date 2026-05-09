#!/usr/bin/env python3
"""
COLT 爬虫 — 从 PMLR 抓取 COLT 2024/2025 论文元数据（含摘要）
COLT 2024: PMLR v247
COLT 2025: PMLR v291
"""
import requests
import json
import re
import time
from datetime import datetime

OUTPUT_FILE = "data/raw/colt_metadata.jsonl"
BASE = "https://proceedings.mlr.press"

VOLUMES = {
    247: 2024,
    291: 2025,
}


def fetch(url, retries=4):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(f"  [重试 {i+1}] {url[-60:]} — {e}", flush=True)
            time.sleep(3 * (i + 1))
    return None


def parse_index(html, vol):
    """从 PMLR 卷首页提取论文链接列表"""
    links = re.findall(rf'href="(https://proceedings\.mlr\.press/v{vol}/[a-z0-9]+\.html)"', html)
    return list(dict.fromkeys(links))  # 去重保序


def parse_paper(html, url, vol, year):
    """从论文页面提取元数据"""
    title_m = re.search(r'<h1>(.*?)</h1>', html, re.DOTALL)
    title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip() if title_m else ""

    # 作者：citation_author meta 标签最可靠
    authors = re.findall(r'<meta name="citation_author" content="([^"]+)"', html)
    authors_str = "; ".join(authors)

    # 摘要
    abs_m = re.search(r'<div[^>]+class=["\']abstract["\'][^>]*>(.*?)</div>', html, re.DOTALL | re.IGNORECASE)
    abstract = re.sub(r'<[^>]+>', '', abs_m.group(1)).strip() if abs_m else ""

    # paper_id 从 URL 提取
    paper_id = "pmlr-v{}-{}".format(vol, url.split("/")[-1].replace(".html", ""))

    # arxiv
    arxiv_m = re.search(r'arxiv\.org/abs/([0-9]+\.[0-9]+)', html)
    arxiv_id = arxiv_m.group(1) if arxiv_m else None

    return {
        "paper_id": paper_id,
        "arxiv_id": arxiv_id,
        "venue": f"COLT_{year}",
        "decision_track": None,
        "title": title,
        "abstract": abstract,
        "authors": authors_str,
        "year": year,
        "doi": None,
        "crawled_at": datetime.now().isoformat(),
    }


def crawl_volume(vol, year):
    index_url = f"{BASE}/v{vol}/"
    print(f"\n{'='*60}", flush=True)
    print(f"[COLT {year}] PMLR v{vol} — {index_url}", flush=True)

    html = fetch(index_url)
    if not html:
        print("  无法获取索引页", flush=True)
        return []

    links = parse_index(html, vol)
    print(f"  找到 {len(links)} 篇论文链接", flush=True)

    papers = []
    for i, url in enumerate(links):
        page = fetch(url)
        if not page:
            print(f"  [跳过] {url}", flush=True)
            continue

        paper = parse_paper(page, url, vol, year)
        if paper["title"]:
            papers.append(paper)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(links)} 篇...", flush=True)
        time.sleep(0.12)

    print(f"  v{vol} 完成：{len(papers)} 篇", flush=True)
    return papers


def main():
    all_papers = []
    for vol, year in VOLUMES.items():
        papers = crawl_volume(vol, year)
        all_papers.extend(papers)
        time.sleep(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n总计写入 {len(all_papers)} 篇 → {OUTPUT_FILE}", flush=True)

    # 摘要覆盖率
    has_abs = sum(1 for p in all_papers if p["abstract"].strip())
    print(f"摘要覆盖率：{has_abs}/{len(all_papers)} ({has_abs/len(all_papers)*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
