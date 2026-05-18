#!/usr/bin/env python3
"""探测 ICLR 2024 的 venueid 结构"""
import openreview
from collections import Counter

client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

venue_id = "ICLR.cc/2024/Conference"

print("获取所有 Submissions，检查其 venueid 字段...\n")

submissions = client.get_all_notes(
    invitation=f"{venue_id}/-/Submission",
    details='venueid'
)

venueid_counts = Counter()
samples = {}

for note in submissions:
    venueid = getattr(note, 'venueid', None)
    if venueid is None:
        details = getattr(note, 'details', None)
        if details and isinstance(details, dict):
            venueid = details.get('venueid', 'NO_VENUEID')
        else:
            venueid = 'NO_VENUEID'
    venueid_counts[venueid] += 1

    if venueid not in samples and venueid_counts[venueid] == 1:
        content = note.content
        title = content.get('title', '')
        if isinstance(title, dict):
            title = title.get('value', title)
        decision_val = content.get('decision', '')
        if isinstance(decision_val, dict):
            decision_val = decision_val.get('value', decision_val)

        samples[venueid] = {
            'paper_id': note.id,
            'forum': note.forum,
            'title': title,
            'decision': decision_val,
            'venueid': venueid,
        }

    if len(venueid_counts) > 20 and venueid_counts.most_common()[-1][1] >= 3:
        break

print("venueid 分布:")
for vid, count in venueid_counts.most_common():
    print(f"  {vid}: {count}")

print("\n各 venueid 示例:")
for vid, s in sorted(samples.items()):
    print(f"\n  [{vid}]")
    print(f"    paper_id: {s['paper_id']}")
    print(f"    title: {s['title']}")
    print(f"    decision: {s['decision']}")

print(f"\n总计扫描: {sum(venueid_counts.values())}")
print(f"venueid 类型: {len(venueid_counts)} 种")
