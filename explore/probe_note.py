#!/usr/bin/env python3
"""深入探测 Submission Note 结构"""
import openreview

client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

notes = client.get_all_notes(invitation="ICLR.cc/2024/Conference/-/Submission")
print(f"获取到 {len(notes)} 条笔记\n")

n = notes[0]
print(f"Note 对象类型: {type(n)}")
print(f"Note 属性: {[a for a in dir(n) if not a.startswith('_')]}")
print(f"\n  id: {n.id}")
print(f"  forum: {n.forum}")
print(f"  number: {getattr(n, 'number', 'N/A')}")
print(f"  cdate: {getattr(n, 'cdate', 'N/A')}")
print(f"  mdate: {getattr(n, 'mdate', 'N/A')}")
print(f"  tcdate: {getattr(n, 'tcdate', 'N/A')}")
print(f"  tmdate: {getattr(n, 'tmdate', 'N/A')}")
print(f"  content keys: {list(n.content.keys())}")
print(f"\n  content['title']: {n.content.get('title')}")
print(f"  content['decision']: {n.content.get('decision')}")

notes_with_decision = [x for x in notes if x.content.get('decision')]
print(f"\n有 decision 字段的笔记: {len(notes_with_decision)}")
if notes_with_decision:
    nn = notes_with_decision[0]
    print(f"  title: {nn.content.get('title')}")
    print(f"  decision: {nn.content.get('decision')}")
    print(f"  forum: {nn.forum}")
    print(f"  id: {nn.id}")

print("\n检查 notes[0] 的 details attr:")
details = getattr(n, 'details', None)
print(f"  details attr: {details}")
print(f"  details type: {type(details)}")
