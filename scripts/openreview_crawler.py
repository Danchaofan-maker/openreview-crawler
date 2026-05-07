#!/usr/bin/env python3
"""
OpenReview 论文元数据爬虫
抓取顶级 AI 会议的已接受论文元数据
"""
import openreview
import json
import re
import time
import signal
from datetime import datetime
from tqdm import tqdm

VENUES = {
    "ICLR_2024":    "ICLR.cc/2024/Conference",
    "ICML_2024":    "ICML.cc/2024/Conference",
    "NeurIPS_2023": "NeurIPS.cc/2023/Conference",
    "NeurIPS_2024": "NeurIPS.cc/2024/Conference",
    "ICLR_2025":    "ICLR.cc/2025/Conference",
    "ICML_2025":    "ICML.cc/2025/Conference",
    "NeurIPS_2025": "NeurIPS.cc/2025/Conference",
    "ICLR_2026":    "ICLR.cc/2026/Conference",
}

REQUEST_TIMEOUT = 300
BATCH_TIMEOUT = 60
REQUEST_DELAY = 0.05

VENUE_TIMEOUTS = {
    "ICLR_2024":    600,
    "ICML_2024":    600,
    "NeurIPS_2023": 600,
    "NeurIPS_2024": 600,
    "ICLR_2025":    600,
    "ICML_2025":    600,
    "NeurIPS_2025": 600,
    "ICLR_2026":    600,
}


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


def extract_arxiv_id(content):
    patterns = [
        r'arxiv\.org/abs/([0-9]+\.[0-9]+)',
        r'arxiv\.org/pdf/([0-9]+\.[0-9]+)',
    ]
    for key in ['pdf', 'arxiv', 'arxiv_id', 'arxiv_link']:
        val = content.get(key, '')
        if isinstance(val, dict):
            val = val.get('value', val)
        if not val or not isinstance(val, str):
            continue
        for pattern in patterns:
            match = re.search(pattern, val)
            if match:
                return match.group(1)
    return None


def parse_track(venue_str):
    if not venue_str:
        return None
    vl = str(venue_str).lower()
    if 'oral' in vl:
        return 'Oral'
    elif 'spotlight' in vl:
        return 'Spotlight'
    elif 'poster' in vl:
        return 'Poster'
    return None


def extract_paper(note, venue_name):
    content = note.content

    title = content.get('title', '')
    title = title.get('value', title) if isinstance(title, dict) else title

    abstract = content.get('abstract', '')
    abstract = abstract.get('value', abstract) if isinstance(abstract, dict) else abstract

    authors = content.get('authors', [])
    authors = authors.get('value', authors) if isinstance(authors, dict) else authors
    authors_str = '; '.join(authors) if isinstance(authors, list) else str(authors)

    venue_field = content.get('venue', '')
    venue_field = venue_field.get('value', venue_field) if isinstance(venue_field, dict) else venue_field
    track = parse_track(venue_field)

    arxiv_id = extract_arxiv_id(content)

    return {
        "paper_id": note.id,
        "arxiv_id": arxiv_id,
        "venue": venue_name,
        "decision_track": track,
        "title": title,
        "abstract": abstract,
        "authors": authors_str,
        "crawled_at": datetime.now().isoformat(),
    }


def crawl_venue(client, venue_name, venue_id):
    print(f"\n{'='*60}")
    print(f"[{venue_name}] 正在获取已接受论文...")

    venue_timeout = VENUE_TIMEOUTS.get(venue_name, 600)
    start_time = time.time()

    papers = []
    accepted = []
    try:
        all_notes = client.get_all_notes(invitation=f"{venue_id}/-/Submission")
        accepted = [n for n in all_notes if (n.content.get('venueid') or {}).get('value') == venue_id]
        print(f"  已接受论文: {len(accepted)} 篇")

        with tqdm(total=len(accepted), desc=f"抓取 {venue_name}", unit="篇") as pbar:
            for note in accepted:
                if time.time() - start_time > venue_timeout:
                    print(f"  [超时] 已运行 {int(time.time()-start_time)}s，停止抓取")
                    break

                try:
                    paper = extract_paper(note, venue_name)
                    papers.append(paper)
                except Exception as e:
                    print(f"  [错误] 提取论文失败: {e}")

                pbar.update(1)
                time.sleep(REQUEST_DELAY)

    except TimeoutException:
        print(f"  [超时] 批次请求超时")
    except Exception as e:
        print(f"  [错误] {e}")

    elapsed = int(time.time() - start_time)
    print(f"  完成 {len(papers)}/{len(accepted)} 篇，耗时 {elapsed}s")

    return papers


def save_to_jsonl(papers, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')


def print_stats(papers, venue_name):
    tracks = {"Oral": 0, "Spotlight": 0, "Poster": 0, "Unknown": 0}
    arxiv_found = 0

    for p in papers:
        t = p['decision_track'] or 'Unknown'
        tracks[t] = tracks.get(t, 0) + 1
        if p['arxiv_id']:
            arxiv_found += 1

    total = len(papers)
    print(f"\n[统计] {venue_name}")
    print(f"  总计: {total} 篇")
    for t in ['Oral', 'Spotlight', 'Poster']:
        cnt = tracks.get(t, 0)
        print(f"  {t}: {cnt} ({cnt/total*100:.1f}%)" if total else f"  {t}: 0")
    unk = tracks.get('Unknown', 0)
    if unk:
        print(f"  未知track: {unk}")
    print(f"  arxiv_id 提取率: {arxiv_found}/{total} ({arxiv_found/total*100:.1f}%)" if total else "")


def main():
    print("OpenReview 论文元数据爬虫")
    print("=" * 60)

    signal.signal(signal.SIGALRM, timeout_handler)

    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    print(f"已连接 OpenReview API\n")

    for venue_name, venue_id in VENUES.items():
        papers = crawl_venue(client, venue_name, venue_id)

        if papers:
            output_file = f"data/raw/{venue_name.lower()}_metadata.jsonl"
            save_to_jsonl(papers, output_file)
            print(f"  已保存到: {output_file}")
            print_stats(papers, venue_name)
        else:
            print(f"  未获取到论文")

        time.sleep(2)

    print(f"\n{'='*60}")
    print("[完成]")


if __name__ == "__main__":
    main()
