#!/usr/bin/env python3
"""探测 venueid 在 content 中的分布"""
import openreview
from collections import Counter

client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

venue_id = "ICLR.cc/2024/Conference"
notes = client.get_all_notes(invitation=f"{venue_id}/-/Submission")

venueid_counts = Counter()
samples = {}

for note in notes:
    content = note.content
    venueid = content.get('venueid', 'MISSING')
    if isinstance(venueid, dict):
        venueid = venueid.get('value', venueid)

    venueid_counts[venueid] += 1

    if venueid not in samples:
        title = content.get('title', '')
        if isinstance(title, dict):
            title = title.get('value', title)
        pdf = content.get('pdf', '')
        if isinstance(pdf, dict):
            pdf = pdf.get('value', pdf)
        samples[venueid] = {
            'paper_id': note.id,
            'title': title[:80],
            'venueid': venueid,
            'pdf': pdf[:100],
        }

print("venueid 在 content 中的分布:")
for vid, count in venueid_counts.most_common():
    marker = " <<< ACCEPTED" if 'Accept' in str(vid) else ""
    print(f"  {count:5d}  {vid}{marker}")

print("\n各 venueid 第一个示例:")
for vid in sorted(samples.keys()):
    s = samples[vid]
    print(f"\n  [{vid}]")
    print(f"    paper_id: {s['paper_id']}")
    print(f"    title: {s['title']}")
    print(f"    pdf: {s['pdf']}")

print(f"\n总计: {len(notes)}")
