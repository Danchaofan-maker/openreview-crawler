#!/usr/bin/env python3
"""探测 accepted papers 的 decision track 信息"""
import openreview

client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

venue_id = "ICLR.cc/2024/Conference"

accepted_notes = client.get_all_notes(invitation=f"{venue_id}/-/Submission")
accepted = [n for n in accepted_notes if n.content.get('venueid', {}).get('value') == venue_id]
print(f"已接受论文: {len(accepted)}")

sample = accepted[0]
print(f"\n示例 paper_id={sample.id}, forum={sample.forum}")

print("\n=== 尝试从 Submission content 中获取 track ===")
content = sample.content
for k in content:
    v = content[k]
    if isinstance(v, dict):
        v = v.get('value', v)
    if v and isinstance(v, str) and ('oral' in v.lower() or 'spotlight' in v.lower() or 'poster' in v.lower()):
        print(f"  {k}: {v}")

print("\n=== 检查 Note 的 invitations 和 parent_invitations ===")
print(f"  invitations: {getattr(sample, 'invitations', [])}")
print(f"  parent_invitations: {getattr(sample, 'parent_invitations', [])}")

print("\n=== 检查 notes[0] 的 details 参数 ===")
notes_with_details = list(client.get_all_notes(
    invitation=f"{venue_id}/-/Submission",
    details='replyCount'
))
print(f"  有 details 的 notes: {len(notes_with_details)}")
nd = notes_with_details[0]
print(f"  details: {getattr(nd, 'details', None)}")

print("\n=== 尝试获取该 paper 的所有回复(通过 forum 查询) ===")
forum_notes = client.get_all_notes(forum=sample.forum)
print(f"  forum={sample.forum} 的 notes: {len(forum_notes)}")
for fn in forum_notes:
    print(f"  - id={fn.id}, invitation={fn.invitations}")
    print(f"    content keys: {list(fn.content.keys())}")
    for k, v in fn.content.items():
        if isinstance(v, dict):
            v = v.get('value', v)
        print(f"    {k}: {str(v)[:100]}")
    print()

print("\n=== 尝试 ICLR.cc/2024/Conference/-/Meta_Review ===")
meta_notes = client.get_all_notes(invitation=f"{venue_id}/-/Meta_Review")
print(f"  Meta_Review: {len(meta_notes)}")

print("\n=== 尝试 ICLR.cc/2024/Conference/-/Paper_Meta_Review ===")
try:
    pm_notes = client.get_all_notes(invitation=f"{venue_id}/-/Paper_Meta_Review")
    print(f"  Paper_Meta_Review: {len(pm_notes)}")
except Exception as e:
    print(f"  失败: {e}")
