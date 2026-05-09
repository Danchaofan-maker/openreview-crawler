#!/usr/bin/env python3
"""
Exploratory Data Analysis of ML Paper Scoring Results
DeepSeek v4-pro (thinking mode) scores on 600 ML papers
"""
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading and Parsing Data")
print("=" * 70)

records = []
with open("data/llm_v4pro_thinking_N600_seed42.jsonl") as f:
    for line in f:
        d = json.loads(line.strip())
        records.append(d)

print(f"Total records loaded: {len(records)}")
print(f"Records with ok=True: {sum(1 for r in records if r.get('ok'))}")
print(f"Records with errors: {sum(1 for r in records if r.get('error'))}")

# ── 2. Extract Scores into DataFrame ─────────────────────────────────────────
score_fields = [
    'mathematical_rigor', 'theoretical_novelty', 'mathematical_depth',
    'assumption_realism', 'empirical_reliance', 'theory_experiment_alignment',
    'compute_complexity', 'epistemological_intent', 'scope_generality',
    'confidence_score'
]
abbrev = {
    'mathematical_rigor': 'mr',
    'theoretical_novelty': 'tn',
    'mathematical_depth': 'md',
    'assumption_realism': 'ar',
    'empirical_reliance': 'er',
    'theory_experiment_alignment': 'tea',
    'compute_complexity': 'cc',
    'epistemological_intent': 'ei',
    'scope_generality': 'sg',
    'confidence_score': 'cs'
}

rows = []
for r in records:
    if not r.get('ok') or not r.get('parsed'):
        continue
    p = r['parsed']
    row = {
        'paper_id': r.get('paper_id', ''),
        'title': r.get('title', ''),
        'venue': r.get('venue', ''),
        'year': r.get('year', ''),
        'reasoning': r.get('reasoning', ''),
        'reasoning_chars': r.get('reasoning_chars', 0),
    }
    for field in score_fields:
        val = p.get(field)
        if isinstance(val, dict):
            row[abbrev[field]] = val.get('score')
        else:
            row[abbrev[field]] = val
    # logical chain
    lc = p.get('logical_chain', {}) or {}
    row['lc_integrity'] = lc.get('integrity', None) if isinstance(lc, dict) else None
    # marketing
    mkt = p.get('marketing_detected', {}) or {}
    row['marketing'] = mkt.get('flag', None) if isinstance(mkt, dict) else None
    # human review
    hr = p.get('human_review_required', {}) or {}
    row['human_review'] = hr.get('flag', None) if isinstance(hr, dict) else None

    rows.append(row)

df = pd.DataFrame(rows)
print(f"\nDataFrame shape: {df.shape}")
print(f"Papers with all scores: {df[['mr','tn','md','ar','er','cc','ei','sg','cs']].notna().all(axis=1).sum()}")

# ── 3. Venue Distribution ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Venue & Year Distribution")
print("=" * 70)
venue_counts = df['venue'].value_counts()
print(venue_counts.to_string())

# ── 4. Score Distributions ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Score Distributions (mean ± std, median, mode)")
print("=" * 70)

score_cols = ['mr', 'tn', 'md', 'ar', 'er', 'tea', 'cc', 'ei', 'sg', 'cs']
full_names = {
    'mr': 'mathematical_rigor',
    'tn': 'theoretical_novelty',
    'md': 'mathematical_depth',
    'ar': 'assumption_realism',
    'er': 'empirical_reliance',
    'tea': 'theory_experiment_alignment',
    'cc': 'compute_complexity',
    'ei': 'epistemological_intent',
    'sg': 'scope_generality',
    'cs': 'confidence_score'
}

for col in score_cols:
    s = df[col].dropna()
    if len(s) == 0:
        continue
    mode_val = s.mode().iloc[0] if len(s.mode()) > 0 else 'N/A'
    print(f"\n{full_names[col]} ({col}):")
    print(f"  N={len(s)}, mean={s.mean():.2f}, std={s.std():.2f}, median={s.median():.1f}, mode={mode_val}")
    print(f"  min={s.min()}, max={s.max()}, skew={s.skew():.2f}")
    # Distribution by integer value
    vc = s.astype(float).round(0).astype(int).value_counts().sort_index()
    dist_str = " | ".join([f"{k}:{v}" for k, v in vc.items()])
    print(f"  dist: {dist_str}")

# ── 5. Logical Chain Integrity ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Logical Chain Integrity & Boolean Fields")
print("=" * 70)
lci = df['lc_integrity'].value_counts()
print("logical_chain.integrity:")
for k, v in lci.items():
    print(f"  {k}: {v} ({100*v/len(df):.1f}%)")

print(f"\nmarketing_detected=True: {df['marketing'].sum()} ({100*df['marketing'].mean():.1f}%)")
print(f"human_review_required=True: {df['human_review'].sum()} ({100*df['human_review'].mean():.1f}%)")

# ── 6. Correlation Matrix ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Correlation Matrix (Pearson) among Score Dimensions")
print("=" * 70)
numeric_cols = ['mr', 'tn', 'md', 'ar', 'er', 'cc', 'ei', 'sg', 'cs']
corr = df[numeric_cols].corr()
print(corr.round(2).to_string())

# ── 7. Top correlations ───────────────────────────────────────────────────────
print("\n  Top positive correlations:")
corr_pairs = []
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        c1, c2 = numeric_cols[i], numeric_cols[j]
        corr_pairs.append((c1, c2, corr.loc[c1, c2]))
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for c1, c2, r in corr_pairs[:10]:
    direction = "+" if r > 0 else "-"
    print(f"  {c1} vs {c2}: r={r:.3f}")

# ── 8. Composite Score & High-Theory Papers ───────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Composite Theory Score Analysis")
print("=" * 70)

# Composite: high mr + high tn + high md + low er + high ei
df['theory_score'] = (
    df['mr'].fillna(0) +
    df['tn'].fillna(0) +
    df['md'].fillna(0) +
    (10 - df['er'].fillna(10)) +
    df['ei'].fillna(0)
) / 5.0

theory_thresh = df['theory_score'].quantile(0.8)
print(f"Theory score: mean={df['theory_score'].mean():.2f}, std={df['theory_score'].std():.2f}")
print(f"Top 20% theory score threshold: {theory_thresh:.2f}")
print(f"Papers in top 20% theory: {(df['theory_score'] >= theory_thresh).sum()}")

top_theory = df[df['theory_score'] >= theory_thresh].nlargest(15, 'theory_score')
print("\nTop 15 most theory-rigorous papers:")
for _, row in top_theory.iterrows():
    print(f"  [{row['venue']}] score={row['theory_score']:.1f} | mr={row['mr']},tn={row['tn']},md={row['md']},er={row['er']},ei={row['ei']}")
    print(f"    {row['title'][:80]}")

# ── 9. Paper Type Clustering ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: Paper Type Clustering (Rule-Based)")
print("=" * 70)

def classify_paper(row):
    mr, tn, md, er, ei = row['mr'], row['tn'], row['md'], row['er'], row['ei']
    if pd.isna(mr) or pd.isna(er) or pd.isna(ei):
        return 'incomplete'

    # Pure theory
    if er <= 1 and mr >= 7 and ei >= 6:
        return 'pure_theory'
    # Theoretical + empirical aligned
    if mr >= 6 and er <= 4 and ei >= 4:
        return 'theory_driven'
    # Deep math tools but applied
    if md >= 6 and er >= 4:
        return 'math_methods'
    # Empirical/engineering focused
    if er >= 6 and mr <= 4:
        return 'empirical_eng'
    # Mixed/moderate
    return 'mixed'

df['paper_type'] = df.apply(classify_paper, axis=1)
pt_counts = df['paper_type'].value_counts()
print("Paper type distribution:")
for k, v in pt_counts.items():
    print(f"  {k}: {v} ({100*v/len(df):.1f}%)")

# Stats per type
print("\nScore averages by paper type:")
type_stats = df.groupby('paper_type')[['mr','tn','md','er','ei','sg','cs']].mean().round(2)
print(type_stats.to_string())

# ── 10. Bimodal / Extreme Papers ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: Extreme Cases Analysis")
print("=" * 70)

# Highest mr papers
print("\nTop 10 highest mathematical_rigor (mr):")
for _, row in df.nlargest(10, 'mr').iterrows():
    print(f"  mr={row['mr']}, tn={row['tn']}, md={row['md']}, er={row['er']} | [{row['venue']}] {row['title'][:70]}")

print("\nBottom 10 lowest mathematical_rigor (mr):")
for _, row in df.nsmallest(10, 'mr').iterrows():
    print(f"  mr={row['mr']}, tn={row['tn']}, er={row['er']} | [{row['venue']}] {row['title'][:70]}")

print("\nMost novel theoretically (tn >= 8):")
high_tn = df[df['tn'] >= 8].sort_values('tn', ascending=False)
print(f"  Count: {len(high_tn)}")
for _, row in high_tn.head(10).iterrows():
    print(f"  tn={row['tn']}, mr={row['mr']}, md={row['md']} | [{row['venue']}] {row['title'][:70]}")

print("\nHighest empirical reliance (er >= 8):")
high_er = df[df['er'] >= 8].sort_values('er', ascending=False)
print(f"  Count: {len(high_er)} ({100*len(high_er)/len(df):.1f}%)")
for _, row in high_er.head(8).iterrows():
    print(f"  er={row['er']}, mr={row['mr']}, md={row['md']} | [{row['venue']}] {row['title'][:70]}")

# ── 11. TEA Analysis ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 9: Theory-Experiment Alignment (TEA) Analysis")
print("=" * 70)
null_tea = df['tea'].isna().sum()
print(f"TEA=null (pure theory or pure empirical): {null_tea} ({100*null_tea/len(df):.1f}%)")
has_tea = df['tea'].dropna()
print(f"Papers with TEA score: {len(has_tea)}")
print(f"TEA distribution: mean={has_tea.mean():.2f}, std={has_tea.std():.2f}")
vc = has_tea.astype(float).round(0).astype(int).value_counts().sort_index()
print("TEA dist:", " | ".join([f"{k}:{v}" for k, v in vc.items()]))

# ── 12. Venue-Level Analysis ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 10: Venue-Level Score Comparison")
print("=" * 70)
# Map to short venue names
df['venue_short'] = df['venue'].str.split('_').str[0]
venue_stats = df.groupby('venue_short')[['mr','tn','md','er','ei','cs']].mean().round(2)
venue_counts_s = df['venue_short'].value_counts()
print("Venue counts:")
print(venue_counts_s.to_string())
print("\nVenue score averages:")
print(venue_stats.to_string())

# ── 13. Reasoning Depth vs Score ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 11: Reasoning Depth (chars) vs Score")
print("=" * 70)
df['reasoning_q'] = pd.qcut(df['reasoning_chars'], q=4, labels=['Q1','Q2','Q3','Q4'])
reasoning_stats = df.groupby('reasoning_q', observed=True)[['mr','cs','theory_score']].mean().round(2)
print("Reasoning depth quartile vs scores:")
print(reasoning_stats.to_string())

# ── 14. Marketing & Human Review ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 12: Marketing Detection & Human Review Analysis")
print("=" * 70)
mkt_true = df[df['marketing'] == True]
print(f"Marketing detected: {len(mkt_true)} papers ({100*len(mkt_true)/len(df):.1f}%)")
if len(mkt_true) > 0:
    print(f"  Avg mr for marketing papers: {mkt_true['mr'].mean():.2f}")
    print(f"  Avg er for marketing papers: {mkt_true['er'].mean():.2f}")
    print(f"  Avg ei for marketing papers: {mkt_true['ei'].mean():.2f}")
    print("  Sample titles:")
    for _, row in mkt_true.head(5).iterrows():
        print(f"    [{row['venue']}] {row['title'][:75]}")

hr_true = df[df['human_review'] == True]
print(f"\nHuman review required: {len(hr_true)} papers ({100*len(hr_true)/len(df):.1f}%)")

# ── 15. Dimension Cross-tabs for Paper Archetypes ─────────────────────────────
print("\n" + "=" * 70)
print("STEP 13: Paper Archetype Discovery (Multi-Dimensional)")
print("=" * 70)

# Define archetype bins
df['mr_bin'] = pd.cut(df['mr'], bins=[0,3,6,10], labels=['low(0-3)','mid(4-6)','high(7-10)'], right=True, include_lowest=True)
df['er_bin'] = pd.cut(df['er'], bins=[0,3,6,10], labels=['low(0-3)','mid(4-6)','high(7-10)'], right=True, include_lowest=True)
df['ei_bin'] = pd.cut(df['ei'], bins=[0,3,6,10], labels=['low(0-3)','mid(4-6)','high(7-10)'], right=True, include_lowest=True)

# Cross-tab mr vs er
print("\nCross-tab: mr_bin vs er_bin (paper counts)")
ct = pd.crosstab(df['mr_bin'], df['er_bin'])
print(ct.to_string())

# Define 4 archetypes based on mr and er
print("\n4 Archetypes (high/low mr vs high/low er):")
archetype_data = {
    'pure_math (mr>=7, er<=3)': df[(df['mr'] >= 7) & (df['er'] <= 3)],
    'math_applied (mr>=7, er>=4)': df[(df['mr'] >= 7) & (df['er'] >= 4)],
    'empirical_light (mr<=3, er<=4)': df[(df['mr'] <= 3) & (df['er'] <= 4)],
    'leaderboard (mr<=4, er>=6)': df[(df['mr'] <= 4) & (df['er'] >= 6)],
}
for name, subset in archetype_data.items():
    if len(subset) == 0:
        continue
    print(f"\n  {name}: N={len(subset)}")
    print(f"    tn={subset['tn'].mean():.1f}, md={subset['md'].mean():.1f}, ei={subset['ei'].mean():.1f}, sg={subset['sg'].mean():.1f}")
    print(f"    lc_integrity: {subset['lc_integrity'].value_counts().to_dict()}")
    print(f"    Sample titles:")
    for _, row in subset.head(3).iterrows():
        print(f"      [{row['venue']}] {row['title'][:70]}")

# ── 16. Reasoning Sampling ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 14: Reasoning Sampling - Patterns at Score Extremes")
print("=" * 70)

# Sample highest-theory papers and look at reasoning
top5 = df.nlargest(5, 'theory_score')
print("\nTop 5 theory papers - reasoning excerpt (first 600 chars):")
for _, row in top5.iterrows():
    print(f"\n  [{row['venue']}] {row['title'][:70]}")
    print(f"  theory={row['theory_score']:.1f}, mr={row['mr']}, tn={row['tn']}, md={row['md']}, er={row['er']}, ei={row['ei']}")
    reasoning_excerpt = str(row['reasoning'])[:600].replace('\n', ' ')
    print(f"  Reasoning: {reasoning_excerpt}...")

# Sample empirical papers
bot5_er = df[df['er'] >= 8].nlargest(3, 'er')
print("\nTop 3 most empirical papers - reasoning excerpt:")
for _, row in bot5_er.iterrows():
    print(f"\n  [{row['venue']}] {row['title'][:70]}")
    print(f"  er={row['er']}, mr={row['mr']}, md={row['md']}")
    reasoning_excerpt = str(row['reasoning'])[:500].replace('\n', ' ')
    print(f"  Reasoning: {reasoning_excerpt}...")

# ── 17. Double-High (mr + tn both high) ──────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 15: 'Diamond' Papers (mr>=8 AND tn>=8 AND md>=6 AND er<=2)")
print("=" * 70)
diamonds = df[(df['mr'] >= 8) & (df['tn'] >= 8) & (df['md'] >= 6) & (df['er'] <= 2)]
print(f"Diamond papers: {len(diamonds)}")
for _, row in diamonds.iterrows():
    print(f"  mr={row['mr']},tn={row['tn']},md={row['md']},er={row['er']},ei={row['ei']},sg={row['sg']} | [{row['venue']}] {row['title'][:70]}")

# ── 18. COLT-specific analysis ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 16: COLT vs NeurIPS/ICLR Comparison")
print("=" * 70)
colt = df[df['venue_short'] == 'COLT']
neurips = df[df['venue_short'] == 'NeurIPS']
iclr = df[df['venue_short'] == 'ICLR']

for name, subset in [('COLT', colt), ('NeurIPS', neurips), ('ICLR', iclr)]:
    if len(subset) == 0:
        continue
    print(f"\n{name} (N={len(subset)}):")
    for col in ['mr', 'tn', 'md', 'er', 'ei', 'sg']:
        print(f"  {col}: mean={subset[col].mean():.2f}, std={subset[col].std():.2f}")
    print(f"  theory_score: {subset['theory_score'].mean():.2f}")
    print(f"  lc intact: {(subset['lc_integrity']=='intact').sum()} ({100*(subset['lc_integrity']=='intact').mean():.0f}%)")

# ── 19. Final Summary Statistics ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 17: Final Summary - Key Findings")
print("=" * 70)

print(f"\nTotal papers analyzed: {len(df)}")
print(f"Pure theory papers (er<=1, mr>=7, ei>=6): {len(df[(df['er']<=1) & (df['mr']>=7) & (df['ei']>=6)])}")
print(f"Leaderboard papers (er>=7, mr<=4): {len(df[(df['er']>=7) & (df['mr']<=4)])}")
print(f"High-novelty papers (tn>=8): {len(df[df['tn']>=8])}")
print(f"Deep math papers (md>=7): {len(df[df['md']>=7])}")
print(f"Logical chain intact: {(df['lc_integrity']=='intact').sum()} ({100*(df['lc_integrity']=='intact').mean():.1f}%)")
print(f"Marketing detected: {df['marketing'].sum()} ({100*df['marketing'].mean():.1f}%)")
print(f"Human review needed: {df['human_review'].sum()} ({100*df['human_review'].mean():.1f}%)")

# Score for SCREENING: composite filter
screened = df[
    (df['mr'] >= 6) &
    (df['tn'] >= 6) &
    (df['er'] <= 4) &
    (df['lc_integrity'].isin(['intact', 'partial']))
]
print(f"\nPapers passing screen (mr>=6, tn>=6, er<=4, lc intact/partial): {len(screened)} ({100*len(screened)/len(df):.1f}%)")
print("Top screened papers by theory_score:")
for _, row in screened.nlargest(20, 'theory_score').iterrows():
    print(f"  ts={row['theory_score']:.1f} mr={row['mr']},tn={row['tn']},md={row['md']},er={row['er']},ei={row['ei']} [{row['venue']}] {row['title'][:65]}")

print("\nDone.")
