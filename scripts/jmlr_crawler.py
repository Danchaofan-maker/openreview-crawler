#!/usr/bin/env python3
"""
JMLR 爬虫 — 直接从 jmlr.org 按卷号抓取论文元数据
v25=2024, v26=2025, v27=2026(进行中)
"""
import requests
import json
import re
import time
from datetime import datetime

OUTPUT_FILE = "data/raw/jmlr_metadata.jsonl"
VOLUMES = {25: 2024, 26: 2025}
BASE = "https://jmlr.org"


def fetch(url, retries=4):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(f"  [重试 {i+1}] {e}", flush=True)
            time.sleep(3 * (i + 1))
    return None


def parse_volume(html, vol, year):
    papers = []
    # Each paper is wrapped in <dl>...<dt>TITLE</dt><dd>AUTHORS; ...</dd></dl>
    blocks = re.findall(r'<dl>(.*?)</dl>', html, re.DOTALL)
    for block in blocks:
        title_m = re.search(r'<dt>(.*?)</dt>', block, re.DOTALL)
        # <dd> tag is often unclosed; grab everything after it
        dd_m = re.search(r'<dd>(.*?)(?:</dd>|$)', block, re.DOTALL)
        abs_m = re.search(r"href=['\"]?(/papers/v\d+/[^'\"]+\.html)", block)
        if not title_m or not dd_m:
            continue

        title = re.sub(r'<[^>]+>', '', title_m.group(1)).strip()
        dd_text = dd_m.group(1)

        # Authors are in <b><i>...</i></b>
        authors_m = re.search(r'<b><i>(.*?)</i></b>', dd_text, re.DOTALL)
        authors = re.sub(r'<[^>]+>', '', authors_m.group(1)).strip() if authors_m else ""
        authors = re.sub(r'\s+', ' ', authors).replace(',', ';')

        # arxiv link if present
        arxiv_m = re.search(r'arxiv\.org/abs/([0-9]+\.[0-9]+)', block)
        arxiv_id = arxiv_m.group(1) if arxiv_m else None

        abs_url = BASE + abs_m.group(1) if abs_m else None
        paper_id = abs_m.group(1).split('/')[-1].replace('.html', '') if abs_m else title[:20]

        papers.append({
            "paper_id": f"jmlr-v{vol}-{paper_id}",
            "arxiv_id": arxiv_id,
            "venue": "JMLR",
            "decision_track": None,
            "title": title,
            "abstract": "",  # 需要进abs页面才有，暂不抓
            "authors": authors,
            "year": year,
            "doi": None,
            "abs_url": abs_url,
            "crawled_at": datetime.now().isoformat(),
        })
    return papers


def fetch_abstract(abs_url):
    html = fetch(abs_url)
    if not html:
        return ""
    m = re.search(r'<p>(.*?)</p>', html, re.DOTALL)
    if m:
        return re.sub(r'<[^>]+>', '', m.group(1)).strip()
    return ""


def main():
    all_papers = []
    for vol, year in VOLUMES.items():
        url = f"{BASE}/papers/v{vol}/"
        print(f"\n{'='*60}", flush=True)
        print(f"[JMLR v{vol} / {year}] {url}", flush=True)
        html = fetch(url)
        if not html:
            print(f"  无法获取页面", flush=True)
            continue

        papers = parse_volume(html, vol, year)
        print(f"  找到 {len(papers)} 篇，正在抓取摘要...", flush=True)

        for i, p in enumerate(papers):
            if p["abs_url"]:
                p["abstract"] = fetch_abstract(p["abs_url"])
                p.pop("abs_url", None)
            else:
                p.pop("abs_url", None)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(papers)} 篇...", flush=True)
            time.sleep(0.1)

        all_papers.extend(papers)
        print(f"  v{vol} 完成：{len(papers)} 篇", flush=True)
        time.sleep(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n总计写入 {len(all_papers)} 篇 → {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
