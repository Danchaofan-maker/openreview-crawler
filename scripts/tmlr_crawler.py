#!/usr/bin/env python3
"""
TMLR 爬虫 — 抓取 2024-01-01 之后正式发表的论文
通过 pdate（正式发表时间）过滤，不依赖 API 的 minpdate 参数。
"""
import openreview
import json
import re
import time
from datetime import datetime

OUTPUT_FILE = "data/raw/tmlr_metadata.jsonl"
PDATE_MIN = 1704038400000  # 2024-01-01 00:00:00 UTC (ms)


def extract_arxiv_id(content):
    patterns = [r'arxiv\.org/abs/([0-9]+\.[0-9]+)', r'arxiv\.org/pdf/([0-9]+\.[0-9]+)']
    for key in ['pdf', 'arxiv', 'arxiv_id', 'arxiv_link']:
        val = content.get(key, '')
        if isinstance(val, dict):
            val = val.get('value', val)
        if not val or not isinstance(val, str):
            continue
        for pattern in patterns:
            m = re.search(pattern, val)
            if m:
                return m.group(1)
    return None


def extract_paper(note):
    content = note.content

    title = content.get('title', '')
    title = title.get('value', title) if isinstance(title, dict) else title

    abstract = content.get('abstract', '')
    abstract = abstract.get('value', abstract) if isinstance(abstract, dict) else abstract

    authors = content.get('authors', [])
    authors = authors.get('value', authors) if isinstance(authors, dict) else authors
    authors_str = '; '.join(authors) if isinstance(authors, list) else str(authors)

    arxiv_id = extract_arxiv_id(content)

    pdate_ts = note.pdate
    pdate_str = datetime.utcfromtimestamp(pdate_ts / 1000).strftime('%Y-%m-%d') if pdate_ts else None

    return {
        "paper_id": note.id,
        "arxiv_id": arxiv_id,
        "venue": "TMLR",
        "decision_track": None,
        "title": title,
        "abstract": abstract,
        "authors": authors_str,
        "year": int(pdate_str[:4]) if pdate_str else None,
        "doi": None,
        "pdate": pdate_str,
        "crawled_at": datetime.now().isoformat(),
    }


def main():
    print("TMLR 爬虫（2024-01-01 起）", flush=True)
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

    print("正在获取所有已接受论文...", flush=True)
    all_notes = client.get_all_notes(invitation='TMLR/-/Accepted')
    print(f"  总计获取: {len(all_notes)} 篇", flush=True)

    accepted_2024 = [n for n in all_notes if n.pdate and n.pdate >= PDATE_MIN]
    print(f"  pdate >= 2024-01-01: {len(accepted_2024)} 篇", flush=True)

    papers = []
    for i, note in enumerate(accepted_2024):
        try:
            papers.append(extract_paper(note))
        except Exception as e:
            print(f"  [错误] {note.id}: {e}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"  处理中: {i+1}/{len(accepted_2024)}", flush=True)
        time.sleep(0.02)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\n写入 {len(papers)} 篇 → {OUTPUT_FILE}", flush=True)

    # 年份分布
    from collections import Counter
    years = Counter(p['year'] for p in papers)
    for y, cnt in sorted(years.items()):
        print(f"  {y}: {cnt} 篇", flush=True)


if __name__ == "__main__":
    main()
