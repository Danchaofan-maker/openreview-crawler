#!/usr/bin/env python3
"""
Deep dive: bimodal analysis, clustering, reasoning pattern sampling
"""
import json
import numpy as np
import pandas as pd
from collections import Counter

# ── Load Data ─────────────────────────────────────────────────────────────────
records = []
with open("data/llm_v4pro_thinking_N600_seed42.jsonl") as f:
    for line in f:
        d = json.loads(line.strip())
        records.append(d)

score_fields = ['mathematical_rigor','theoretical_novelty','mathematical_depth',
    'assumption_realism','empirical_reliance','theory_experiment_alignment',
    'compute_complexity','epistemological_intent','scope_generality','confidence_score']
abbrev = {'mathematical_rigor':'mr','theoretical_novelty':'tn','mathematical_depth':'md',
    'assumption_realism':'ar','empirical_reliance':'er','theory_experiment_alignment':'tea',
    'compute_complexity':'cc','epistemological_intent':'ei','scope_generality':'sg','confidence_score':'cs'}

rows = []
for r in records:
    if not r.get('ok') or not r.get('parsed'):
        continue
    p = r['parsed']
    row = {'paper_id':r.get('paper_id',''),'title':r.get('title',''),
           'venue':r.get('venue',''),'year':r.get('year',''),
           'reasoning':r.get('reasoning',''),'reasoning_chars':r.get('reasoning_chars',0)}
    for field in score_fields:
        val = p.get(field)
        row[abbrev[field]] = val.get('score') if isinstance(val, dict) else val
    lc = p.get('logical_chain',{}) or {}
    row['lc_integrity'] = lc.get('integrity') if isinstance(lc,dict) else None
    mkt = p.get('marketing_detected',{}) or {}
    row['marketing'] = mkt.get('flag') if isinstance(mkt,dict) else None
    hr = p.get('human_review_required',{}) or {}
    row['human_review'] = hr.get('flag') if isinstance(hr,dict) else None
    rows.append(row)

df = pd.DataFrame(rows)
df['venue_short'] = df['venue'].str.split('_').str[0]

# ── FINDING A: The Bimodal Gap in mr and tn ───────────────────────────────────
print("=" * 70)
print("FINDING A: Bimodal Gap in mr and tn")
print("=" * 70)

# mr: scores 3,4,5 are sparse - there's a gap between 0-2 and 6-8
mr_dist = df['mr'].round().astype('Int64').value_counts().sort_index()
tn_dist = df['tn'].round().astype('Int64').value_counts().sort_index()

print("mr distribution (showing the gap at 3-5):")
for k,v in mr_dist.items():
    bar = '#' * (v // 3)
    print(f"  mr={k:2d}: {v:4d} {bar}")

print("\ntn distribution (extreme bimodal: 0 vs 6-9):")
for k,v in tn_dist.items():
    bar = '#' * (v // 3)
    print(f"  tn={k:2d}: {v:4d} {bar}")

# Papers in the "gap" region (mr 3-5)
gap_papers = df[(df['mr'] >= 3) & (df['mr'] <= 5)]
print(f"\nPapers in the 'gap' (mr=3..5): {len(gap_papers)} ({100*len(gap_papers)/len(df):.1f}%)")
print("What is their er distribution?")
gap_er = gap_papers['er'].round().astype('Int64').value_counts().sort_index()
for k,v in gap_er.items():
    print(f"  er={k}: {v}")

# ── FINDING B: The Theory Cluster ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINDING B: Theory Cluster Properties")
print("=" * 70)

# Theory cluster: mr>=6, er<=4
theory_cluster = df[(df['mr'] >= 6) & (df['er'] <= 4)]
print(f"Theory cluster (mr>=6, er<=4): N={len(theory_cluster)} ({100*len(theory_cluster)/len(df):.1f}%)")
print(f"  avg mr={theory_cluster['mr'].mean():.2f}, tn={theory_cluster['tn'].mean():.2f}")
print(f"  avg md={theory_cluster['md'].mean():.2f}, ei={theory_cluster['ei'].mean():.2f}")
print(f"  avg sg={theory_cluster['sg'].mean():.2f}, ar={theory_cluster['ar'].mean():.2f}")
print(f"  lc_integrity: {theory_cluster['lc_integrity'].value_counts().to_dict()}")
print(f"  marketing: {theory_cluster['marketing'].sum()}/{len(theory_cluster)}")
print(f"  Venue breakdown:")
for v, cnt in theory_cluster['venue_short'].value_counts().items():
    total_v = (df['venue_short'] == v).sum()
    print(f"    {v}: {cnt}/{total_v} = {100*cnt/total_v:.1f}% of that venue's papers")

# ── FINDING C: The Empirical Cluster ──────────────────────────────────────────
print("\n" + "=" * 70)
print("FINDING C: Empirical Cluster Properties")
print("=" * 70)
empirical_cluster = df[(df['mr'] <= 2) & (df['er'] >= 7)]
print(f"Empirical cluster (mr<=2, er>=7): N={len(empirical_cluster)} ({100*len(empirical_cluster)/len(df):.1f}%)")
print(f"  avg mr={empirical_cluster['mr'].mean():.2f}, tn={empirical_cluster['tn'].mean():.2f}")
print(f"  avg md={empirical_cluster['md'].mean():.2f}, ei={empirical_cluster['ei'].mean():.2f}")
print(f"  avg cc={empirical_cluster['cc'].mean():.2f}, sg={empirical_cluster['sg'].mean():.2f}")
print(f"  lc_integrity: {empirical_cluster['lc_integrity'].value_counts().to_dict()}")
print(f"  marketing: {empirical_cluster['marketing'].sum()}/{len(empirical_cluster)}")
print(f"  Venue breakdown:")
for v, cnt in empirical_cluster['venue_short'].value_counts().items():
    total_v = (df['venue_short'] == v).sum()
    print(f"    {v}: {cnt}/{total_v} = {100*cnt/total_v:.1f}%")

# ── FINDING D: Middle Ground - Theory with Experiments (mr>=6, er 4-7) ─────────
print("\n" + "=" * 70)
print("FINDING D: 'Bridging' Papers (mr>=6, er=4..7)")
print("=" * 70)
bridge = df[(df['mr'] >= 6) & (df['er'] >= 4) & (df['er'] <= 7)]
print(f"Bridging papers: N={len(bridge)} ({100*len(bridge)/len(df):.1f}%)")
print(f"  avg mr={bridge['mr'].mean():.2f}, tn={bridge['tn'].mean():.2f}, md={bridge['md'].mean():.2f}")
print(f"  avg er={bridge['er'].mean():.2f}, ei={bridge['ei'].mean():.2f}, tea={bridge['tea'].mean():.2f}")
print(f"  TEA null: {bridge['tea'].isna().sum()}/{len(bridge)}")
print(f"  lc_integrity: {bridge['lc_integrity'].value_counts().to_dict()}")
print("  Sample titles:")
for _, row in bridge.sample(min(8, len(bridge)), random_state=42).iterrows():
    print(f"    [{row['venue']}] mr={row['mr']},er={row['er']},ei={row['ei']} | {row['title'][:65]}")

# ── FINDING E: Assumption Realism vs Theory Quality ───────────────────────────
print("\n" + "=" * 70)
print("FINDING E: Assumption Realism (ar) - the Independent Dimension")
print("=" * 70)
# ar has almost zero correlation with mr/tn - it's truly orthogonal
# Let's look at low-ar theory papers (strong math, unrealistic assumptions)
low_ar_theory = df[(df['ar'] <= 2) & (df['mr'] >= 6)]
high_ar_theory = df[(df['ar'] >= 7) & (df['mr'] >= 6)]
low_ar_empirical = df[(df['ar'] <= 2) & (df['er'] >= 7)]

print(f"Low ar (<=2) + high mr (>=6): {len(low_ar_theory)} papers (math with unrealistic assumptions)")
for _, row in low_ar_theory.head(6).iterrows():
    print(f"  ar={row['ar']}, mr={row['mr']}, sg={row['sg']} | [{row['venue']}] {row['title'][:65]}")

print(f"\nHigh ar (>=7) + high mr (>=6): {len(high_ar_theory)} papers (math with realistic assumptions)")
for _, row in high_ar_theory.head(6).iterrows():
    print(f"  ar={row['ar']}, mr={row['mr']}, sg={row['sg']} | [{row['venue']}] {row['title'][:65]}")

print(f"\nLow ar (<=2) + high er (>=7): {len(low_ar_empirical)} papers")
print("  (pure empirical papers with very specific/restrictive assumptions)")
for _, row in low_ar_empirical.head(5).iterrows():
    print(f"  ar={row['ar']}, er={row['er']} | [{row['venue']}] {row['title'][:65]}")

# ── FINDING F: Compute Complexity Patterns ────────────────────────────────────
print("\n" + "=" * 70)
print("FINDING F: Compute Complexity (cc) as Empirical Proxy")
print("=" * 70)
high_cc = df[df['cc'] >= 5]
low_cc_er = df[(df['cc'] <= 1) & (df['er'] >= 6)]
print(f"High compute (cc>=5): {len(high_cc)} papers")
print(f"  avg er={high_cc['er'].mean():.2f}, mr={high_cc['mr'].mean():.2f}")
print(f"  Sample: {high_cc.head(3)['title'].tolist()}")

print(f"\nLow compute but high empirical reliance (cc<=1, er>=6): {len(low_cc_er)} papers")
# These are papers that are empirical but don't require heavy compute
for _, row in low_cc_er.head(5).iterrows():
    print(f"  cc={row['cc']}, er={row['er']}, mr={row['mr']} | {row['title'][:65]}")

# ── FINDING G: Reasoning Length vs Paper Complexity ──────────────────────────
print("\n" + "=" * 70)
print("FINDING G: Model Reasoning Length Patterns")
print("=" * 70)
print(f"Reasoning chars: mean={df['reasoning_chars'].mean():.0f}, "
      f"median={df['reasoning_chars'].median():.0f}, "
      f"max={df['reasoning_chars'].max():.0f}")

# Papers where model thought longest
long_reasoning = df.nlargest(10, 'reasoning_chars')
print("\nTop 10 papers with longest model reasoning:")
for _, row in long_reasoning.iterrows():
    print(f"  {row['reasoning_chars']:6d} chars | mr={row['mr']}, er={row['er']}, "
          f"cs={row['cs']} | [{row['venue']}] {row['title'][:55]}")

# Papers where model thought shortest
short_reasoning = df.nsmallest(10, 'reasoning_chars')
print("\nBottom 10 papers with shortest model reasoning:")
for _, row in short_reasoning.iterrows():
    print(f"  {row['reasoning_chars']:6d} chars | mr={row['mr']}, er={row['er']}, "
          f"cs={row['cs']} | [{row['venue']}] {row['title'][:55]}")

print("\nReasoning length by paper type:")
df['theory_score'] = (df['mr'].fillna(0)+df['tn'].fillna(0)+df['md'].fillna(0)+(10-df['er'].fillna(10))+df['ei'].fillna(0))/5.0
df['type'] = 'empirical'
df.loc[(df['mr']>=6) & (df['er']<=4), 'type'] = 'theory'
df.loc[(df['mr']>=6) & (df['er']>4), 'type'] = 'bridging'
df.loc[(df['mr']<=2) & (df['er']>=7), 'type'] = 'leaderboard'
for t in ['theory','bridging','empirical','leaderboard']:
    sub = df[df['type']==t]
    print(f"  {t}: N={len(sub)}, reasoning_chars mean={sub['reasoning_chars'].mean():.0f}")

# ── FINDING H: TMLR vs Conference Venues ──────────────────────────────────────
print("\n" + "=" * 70)
print("FINDING H: TMLR Papers - Journal vs Conference Quality")
print("=" * 70)
tmlr = df[df['venue_short']=='TMLR']
jmlr = df[df['venue_short']=='JMLR']
print(f"TMLR (N={len(tmlr)}): mr={tmlr['mr'].mean():.2f}, tn={tmlr['tn'].mean():.2f}, "
      f"er={tmlr['er'].mean():.2f}, theory_score={tmlr['theory_score'].mean():.2f}")
print(f"JMLR (N={len(jmlr)}): mr={jmlr['mr'].mean():.2f}, tn={jmlr['tn'].mean():.2f}, "
      f"er={jmlr['er'].mean():.2f}, theory_score={jmlr['theory_score'].mean():.2f}")
print(f"\nTMLR papers by type:")
print(tmlr['type'].value_counts().to_dict())
print(f"\nJMLR high-theory papers:")
for _, row in jmlr.nlargest(5, 'theory_score').iterrows():
    print(f"  ts={row['theory_score']:.1f}, mr={row['mr']}, tn={row['tn']}, er={row['er']} | {row['title'][:65]}")

# ── FINDING I: Within-Venue Outliers ──────────────────────────────────────────
print("\n" + "=" * 70)
print("FINDING I: Within-Venue Outliers (theory papers in empirical venues)")
print("=" * 70)
# High-theory papers appearing in typically empirical venues
iclr_theory = df[(df['venue_short']=='ICLR') & (df['theory_score']>=6)]
neurips_theory = df[(df['venue_short']=='NeurIPS') & (df['theory_score']>=6)]
print(f"High-theory papers (ts>=6) in ICLR: {len(iclr_theory)}/{(df['venue_short']=='ICLR').sum()}")
print(f"High-theory papers (ts>=6) in NeurIPS: {len(neurips_theory)}/{(df['venue_short']=='NeurIPS').sum()}")

print("\nTop ICLR theory papers:")
for _, row in iclr_theory.nlargest(5, 'theory_score').iterrows():
    print(f"  ts={row['theory_score']:.1f} | {row['venue']} | {row['title'][:65]}")

# ── FINDING J: Confidence Score Analysis ──────────────────────────────────────
print("\n" + "=" * 70)
print("FINDING J: Confidence Score (cs) Patterns")
print("=" * 70)
low_conf = df[df['cs'] <= 5]
print(f"Low confidence (cs<=5): {len(low_conf)} papers")
for _, row in low_conf.iterrows():
    print(f"  cs={row['cs']}, mr={row['mr']}, er={row['er']}, lc={row['lc_integrity']} | [{row['venue']}] {row['title'][:60]}")

# High confidence papers - what makes them easy to score?
high_conf = df[df['cs'] >= 9]
print(f"\nHighest confidence (cs=9): {len(high_conf)} papers")
print(f"  mr avg: {high_conf['mr'].mean():.2f}")
print(f"  er avg: {high_conf['er'].mean():.2f}")
print(f"  type distribution: {high_conf['type'].value_counts().to_dict()}")
print(f"  lc_integrity: {high_conf['lc_integrity'].value_counts().to_dict()}")

# ── FINDING K: Scope Generality Patterns ──────────────────────────────────────
print("\n" + "=" * 70)
print("FINDING K: Scope Generality (sg) - Where Universal Claims Live")
print("=" * 70)
high_sg = df[(df['sg'] >= 7) & (df['mr'] >= 6)]
print(f"High generality + high rigor (sg>=7, mr>=6): {len(high_sg)} papers")
for _, row in high_sg.iterrows():
    print(f"  sg={row['sg']}, mr={row['mr']}, tn={row['tn']}, er={row['er']} | {row['title'][:65]}")

# ── FINDING L: Rationale text patterns from reasoning ────────────────────────
print("\n" + "=" * 70)
print("FINDING L: Model Decision Patterns from Reasoning Text")
print("=" * 70)

# Key phrases that predict theory cluster
theory_keywords = ['定理', '证明', '收敛', '下界', '上界', '可识别', '可学', '样本复杂度',
                   '信息论', '最优', 'theorem', 'proof', 'convergence', 'lower bound', 'tight']
empirical_keywords = ['实验', '基准', 'SOTA', '性能', '加速', 'benchmark', 'leaderboard',
                      '数据集', '模型', 'fine-tun', '大语言', 'LLM']

print("Theory cluster papers - keyword counts in reasoning:")
tc = df[df['type']=='theory']
for kw in theory_keywords[:6]:
    count = tc['reasoning'].str.contains(kw, case=False).sum()
    print(f"  '{kw}': {count}/{len(tc)} ({100*count/len(tc):.0f}%)")

print("\nLeaderboard cluster papers - keyword counts in reasoning:")
lc_df = df[df['type']=='leaderboard']
for kw in empirical_keywords[:6]:
    count = lc_df['reasoning'].str.contains(kw, case=False).sum()
    print(f"  '{kw}': {count}/{len(lc_df)} ({100*count/len(lc_df):.0f}%)")

# ── FINDING M: Key Signature Combinations ─────────────────────────────────────
print("\n" + "=" * 70)
print("FINDING M: 5 Signature Combinations Summary")
print("=" * 70)

signatures = {
    "Type-1 純理論 (mr>=7,er=0,ei>=7,lc=intact/partial)":
        df[(df['mr']>=7) & (df['er']==0) & (df['ei']>=7)],
    "Type-2 數學方法 (mr>=6,md>=5,er>=4,ei<=4)":
        df[(df['mr']>=6) & (df['md']>=5) & (df['er']>=4) & (df['ei']<=4)],
    "Type-3 理論指導的實驗 (mr>=6,er=4..7,tea>=4)":
        df[(df['mr']>=6) & (df['er']>=4) & (df['er']<=7)],
    "Type-4 軟理論+大量實驗 (mr=3..5,er>=5)":
        df[(df['mr']>=3) & (df['mr']<=5) & (df['er']>=5)],
    "Type-5 純實驗/工程 (mr<=2,er>=7)":
        df[(df['mr']<=2) & (df['er']>=7)],
}

for name, subset in signatures.items():
    print(f"\n{name}")
    print(f"  N={len(subset)} ({100*len(subset)/len(df):.1f}%)")
    if len(subset) > 0:
        print(f"  mr={subset['mr'].mean():.1f}, tn={subset['tn'].mean():.1f}, "
              f"md={subset['md'].mean():.1f}, er={subset['er'].mean():.1f}, "
              f"ei={subset['ei'].mean():.1f}, sg={subset['sg'].mean():.1f}")
        print(f"  lc: {subset['lc_integrity'].value_counts().to_dict()}")
        print(f"  marketing: {subset['marketing'].sum()}/{len(subset)}")
        venues = subset['venue_short'].value_counts()
        top_venues = ', '.join([f"{v}:{c}" for v,c in venues.head(3).items()])
        print(f"  top venues: {top_venues}")
        print(f"  Examples:")
        for _, row in subset.nlargest(min(3, len(subset)), 'mr').iterrows():
            print(f"    [{row['venue']}] {row['title'][:68]}")

print("\nDone.")
