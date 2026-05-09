#!/usr/bin/env python3
"""C1-nothink vs C3 逐篇对比，输出评分内容与差异统计

用法: uv run scripts/compare_c1nt_c3.py
"""
import json
import statistics
from pathlib import Path

C1NT_PATH = "data/llm_c1_nothink_N50_test.jsonl"
C3_PATH   = "data/llm_c3_v08_N50.jsonl"

DIMS = ["mr", "tn", "md", "ar", "er", "tea", "cc", "ei", "sg", "cs"]
DIM_FULL = {
    "mr": "mathematical_rigor", "tn": "theoretical_novelty",
    "md": "mathematical_depth", "ar": "assumption_realism",
    "er": "empirical_reliance", "tea": "theory_experiment_alignment",
    "cc": "compute_complexity", "ei": "epistemological_intent",
    "sg": "scope_generality",   "cs": "confidence_score",
}


def load_c1_nothink(path):
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if not rec.get("ok"):
                    continue
                p = rec.get("parsed") or {}
                pid = rec["paper_id"]
                flat = {"paper_id": pid, "title": rec.get("title", "")[:50]}
                for short, full in DIM_FULL.items():
                    val = p.get(full)
                    flat[short] = val.get("score") if isinstance(val, dict) else val
                lc = p.get("logical_chain") or {}
                flat["ig"] = lc.get("integrity")
                mk = p.get("marketing_detected") or {}
                flat["mk_f"] = mk.get("flag")
                result[pid] = flat
            except Exception:
                continue
    return result


def load_c3(path):
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if not rec.get("ok"):
                    continue
                p = rec.get("parsed") or {}
                pid = rec["paper_id"]
                result[pid] = dict(p, title=rec.get("title", "")[:50])
            except Exception:
                continue
    return result


def fmt(v):
    if v is None:
        return "  - "
    return f"{v:4.1f}"


def main():
    c1 = load_c1_nothink(C1NT_PATH)
    c3 = load_c3(C3_PATH)
    overlap = sorted(set(c1) & set(c3))

    print(f"C1-nothink: {len(c1)} 篇   C3: {len(c3)} 篇   重叠: {len(overlap)} 篇\n")

    # ── 逐篇表 ──────────────────────────────────────────────────────────────
    hdr = f"{'paper_id':<14} {'ig_C1':<8} {'ig_C3':<8}"
    for d in DIMS:
        hdr += f"  {d:>4}_C1 {d:>4}_C3    Δ"
    hdr += "   domain (C3)"
    print(hdr)
    print("─" * (14 + 16 + len(DIMS) * 20 + 40))

    diffs_by_dim = {d: [] for d in DIMS}
    ig_agree = 0
    mk_agree = 0
    fuse_absent_agree = 0
    total_mk = 0

    for pid in overlap:
        r1 = c1[pid]
        r3 = c3[pid]
        ig1 = r1.get("ig", "?")
        ig3 = r3.get("ig", "?")
        if ig1 == ig3:
            ig_agree += 1
        if r1.get("mk_f") == r3.get("mk_f"):
            mk_agree += 1
        total_mk += 1
        if ig1 == "absent" and r3.get("fuse") is True:
            fuse_absent_agree += 1

        row = f"{pid:<14} {ig1:<8} {ig3:<8}"
        for d in DIMS:
            v1 = r1.get(d)
            v3 = r3.get(d)
            if v1 is not None and v3 is not None:
                delta = v3 - v1
                diffs_by_dim[d].append(delta)
                row += f"  {v1:5.1f} {v3:5.1f} {delta:+5.1f}"
            else:
                row += f"  {fmt(v1):>5} {fmt(v3):>5}   N/A"
        dm = str(r3.get("dm", ""))[:38]
        print(row + f"   {dm}")

    # ── 统计汇总 ────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("维度差异统计  (Δ = C3 − C1-nothink，正值表示 C3 偏高)")
    print("=" * 72)
    header = f"{'维度':<5}  {'C1均值':>7} {'C3均值':>7} {'均差Δ':>7} {'std(Δ)':>7} {'|Δ|均':>7}  {'Δ>0':>4} {'Δ=0':>4} {'Δ<0':>4}  n"
    print(header)
    print("─" * 72)

    for d in DIMS:
        dd = diffs_by_dim[d]
        if not dd:
            continue
        c1v = [c1[pid][d] for pid in overlap if c1[pid].get(d) is not None and c3[pid].get(d) is not None]
        c3v = [c3[pid][d] for pid in overlap if c1[pid].get(d) is not None and c3[pid].get(d) is not None]
        c1m  = sum(c1v) / len(c1v)
        c3m  = sum(c3v) / len(c3v)
        mean = sum(dd) / len(dd)
        std  = statistics.stdev(dd) if len(dd) > 1 else 0.0
        absd = sum(abs(x) for x in dd) / len(dd)
        pos  = sum(1 for x in dd if x > 0)
        zero = sum(1 for x in dd if x == 0)
        neg  = sum(1 for x in dd if x < 0)
        flag = " ◄ 偏移显著" if abs(mean) >= 0.8 or std >= 1.5 else ""
        print(f"{d:<5}  {c1m:>7.2f} {c3m:>7.2f} {mean:>+7.2f} {std:>7.2f} {absd:>7.2f}  {pos:>4} {zero:>4} {neg:>4}  {len(dd)}{flag}")

    # ── 辅助字段一致性 ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("辅助字段一致性")
    print("─" * 72)
    print(f"integrity 一致:       {ig_agree}/{len(overlap)} = {ig_agree/len(overlap)*100:.1f}%")
    print(f"marketing_flag 一致:  {mk_agree}/{total_mk} = {mk_agree/total_mk*100:.1f}%")

    absent_in_c1 = sum(1 for pid in overlap if c1[pid].get("ig") == "absent")
    fuse_true_in_c3 = sum(1 for pid in overlap if c3[pid].get("fuse") is True)
    print(f"C1 ig=absent:         {absent_in_c1}/{len(overlap)}")
    print(f"C3 fuse=true:         {fuse_true_in_c3}/{len(overlap)}")
    print(f"两者完全一致(absent↔fuse): {fuse_absent_agree}/{absent_in_c1}")

    # ── ig 分布 ─────────────────────────────────────────────────────────────
    from collections import Counter
    c1_ig = Counter(c1[pid].get("ig") for pid in overlap)
    c3_ig = Counter(c3[pid].get("ig") for pid in overlap)
    print()
    print(f"{'ig 分布':<12} {'C1-nothink':>12} {'C3':>8}")
    print("─" * 34)
    for k in ("intact", "partial", "broken", "absent"):
        print(f"{k:<12} {c1_ig.get(k,0):>12} {c3_ig.get(k,0):>8}")

    # ── 分歧最大的 10 篇（按所有维度 |Δ| 之和排序）──────────────────────
    print()
    print("=" * 72)
    print("评分分歧最大的 10 篇（各维度 |Δ| 之和）")
    print("─" * 72)
    paper_totaldiff = []
    for pid in overlap:
        total = 0
        for d in DIMS:
            v1 = c1[pid].get(d)
            v3 = c3[pid].get(d)
            if v1 is not None and v3 is not None:
                total += abs(v3 - v1)
        paper_totaldiff.append((total, pid))
    paper_totaldiff.sort(reverse=True)

    for total, pid in paper_totaldiff[:10]:
        r1 = c1[pid]
        r3 = c3[pid]
        per_dim = []
        for d in DIMS:
            v1 = r1.get(d)
            v3 = r3.get(d)
            if v1 is not None and v3 is not None and v3 - v1 != 0:
                per_dim.append(f"{d}:{v1:.0f}→{v3:.0f}({v3-v1:+.0f})")
        title = r1.get("title","")[:40]
        print(f"{pid}  Σ|Δ|={total:.1f}  [{title}]")
        print(f"              {', '.join(per_dim[:6])}")


if __name__ == "__main__":
    main()
