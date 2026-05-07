#!/usr/bin/env python3
"""探测 ICLR 2024 的实际 Invitation 结构"""
import openreview

client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

venue_id = "ICLR.cc/2024/Conference"

print(f"=== 探测 {venue_id} ===\n")

print("1. 获取 Group content:")
try:
    group = client.get_group(venue_id)
    print(f"   ID: {group.id}")
    if hasattr(group, 'content') and group.content:
        print(f"   Keys: {list(group.content.keys())}")
        for k, v in group.content.items():
            print(f"   {k}: {v}")
except Exception as e:
    print(f"   失败: {e}")

print("\n2. 获取 Group 的公开 Groups (invitations):")
try:
    invitations = client.get_all_invitations(member=venue_id)
    print(f"   找到 {len(invitations)} 个 Invitations")
    for inv in sorted(invitations, key=lambda x: x.id if hasattr(x, 'id') else str(x))[:30]:
        if hasattr(inv, 'id'):
            print(f"   - {inv.id}")
        elif isinstance(inv, dict):
            print(f"   - (dict) {inv}")
except Exception as e:
    print(f"   失败: {e}")

print("\n3. 尝试获取 Decision Notes (多种 invitation):")
decision_patterns = [
    f"{venue_id}/-/Decision",
    f"{venue_id}/-/Acceptance",
    f"{venue_id}/.*Decision.*",
    venue_id,
]
for pat in decision_patterns:
    try:
        notes = client.get_all_notes(invitation=pat)
        print(f"   '{pat}': {len(notes)} 条")
        if notes:
            n = notes[0]
            print(f"     示例 note.id={n.id}, forum={n.forum}")
            print(f"     content keys: {list(n.content.keys()) if hasattr(n, 'content') else 'N/A'}")
    except Exception as e:
        print(f"   '{pat}': 失败 - {e}")

print("\n4. 尝试列出 ICLR 2024 会议的所有 invitation ID 前缀:")
try:
    notes = client.get_all_notes(invitation=f"{venue_id}/.*", limit=50)
    inv_ids = set()
    for n in notes:
        for inv in getattr(n, 'invitations', []):
            inv_ids.add(inv)
    for inv_id in sorted(inv_ids):
        print(f"   {inv_id}")
except Exception as e:
    print(f"   失败: {e}")
