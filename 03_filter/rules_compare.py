import json, pathlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="规则横向对比", layout="wide")

# ── 数据加载 ────────────────────────────────────────────────
SCORE_FIELDS = {
    "mr": "mathematical_rigor", "tn": "theoretical_novelty",
    "md": "mathematical_depth", "ar": "assumption_realism",
    "er": "empirical_reliance", "tea": "theory_experiment_alignment",
    "cc": "compute_complexity", "ei": "epistemological_intent",
    "sg": "scope_generality", "cs": "confidence_score",
}
BOOL_FIELDS = {"marketing": "marketing_detected", "human_review": "human_review_required"}
ENUM_FIELDS = {"integrity": ["intact", "partial", "broken", "absent"]}
FIELD_ALIASES = {"mk_f": "marketing", "hr_f": "human_review"}

def _get_score(parsed, abbr, long_key):
    v = parsed.get(abbr)
    if v is None: v = parsed.get(long_key)
    if isinstance(v, dict): return v.get("s") if v.get("s") is not None else v.get("score")
    if isinstance(v, (int, float)): return v
    return None

def _get_flag(parsed, short, long_key):
    v = parsed.get(short)
    if v is None: v = parsed.get(f"{short}_f")
    if v is None: v = parsed.get(long_key)
    if isinstance(v, bool): return v
    if isinstance(v, dict): return v.get("f") if "f" in v else v.get("flag")
    return None

def _get_integrity(parsed):
    if isinstance(parsed.get("ig"), str): return parsed["ig"]
    lc = parsed.get("lc") or parsed.get("logical_chain")
    if isinstance(lc, dict): return lc.get("ig") or lc.get("integrity")
    return None

@st.cache_data
def load_data():
    lines = pathlib.Path("data/full/output.jsonl").read_text().strip().split("\n")
    records = []
    for l in lines:
        r = json.loads(l)
        if not r.get("ok"):
            continue
        parsed = r.get("parsed") or {}
        if not isinstance(parsed, dict): continue
        row = {"paper_id": r.get("paper_id"), "title": r.get("title", ""), "venue": r.get("venue", ""), "fuse": parsed.get("fuse")}
        for abbr, key in SCORE_FIELDS.items():
            row[abbr] = _get_score(parsed, abbr, key)
        row["marketing"]    = _get_flag(parsed, "mk", "marketing_detected")
        row["human_review"] = _get_flag(parsed, "hr", "human_review_required")
        row["integrity"]    = _get_integrity(parsed)
        records.append(row)
    return pd.DataFrame(records)

df = load_data()
N_TOTAL = 30983

# ── 规则求值 ─────────────────────────────────────────────────
def eval_cond(df, c):
    f, op, v = c["field"], c["op"], c["value"]
    f = FIELD_ALIASES.get(f, f)
    if f not in df.columns:
        return pd.Series([False]*len(df))
    col = df[f]
    if op == "lt":  return col.fillna(999)  < v
    if op == "lte": return col.fillna(999)  <= v
    if op == "gt":  return col.fillna(-999) > v
    if op == "gte": return col.fillna(-999) >= v
    if op == "eq":  return col.notna() & (col == v)
    if op == "neq": return col.notna() & (col != v)
    if op == "in":  return col.isin(v if isinstance(v, list) else [v])
    return pd.Series([False]*len(df))

def eval_rule(df, rule):
    masks = [eval_cond(df, c) for c in rule.get("conditions", [])]
    if not masks: return pd.Series([False]*len(df))
    r = masks[0]
    for m in masks[1:]:
        r = (r & m) if rule.get("internal_logic", "AND") == "AND" else (r | m)
    return ~r if rule.get("negate") else r

def eval_config(df, config):
    active = [r for r in config.get("rules", []) if r.get("enabled", True)]
    if not active: return pd.Series([False]*len(df))
    hits = [eval_rule(df, r) for r in active]
    fused = hits[0]
    for h in hits[1:]:
        fused = (fused & h) if config.get("inter_logic", "OR") == "AND" else (fused | h)
    if config.get("force_keep_hr"):
        fused &= ~(df["human_review"].fillna(False) == True)
    if config.get("rescue_rules"):
        rescue = pd.Series([False] * len(df))
        for r in config["rescue_rules"]:
            if not r.get("enabled", True):
                continue
            rescue |= eval_rule(df, r)
        fused &= ~rescue
    return fused

# ── 加载规则文件 ─────────────────────────────────────────────
RULE_FILES = {
    "jes":        "03_filter/rules/rules_jes.json",
    "danchaofan": "03_filter/rules/rules_danchaofan.json",
    "claude":     "03_filter/rules/rules_claude.json",
}
LABELS = {"jes": "Jes", "danchaofan": "Danchaofan", "claude": "Claude"}
COLORS = {"jes": "#2196F3", "danchaofan": "#FF9800", "claude": "#9C27B0"}

configs, fused_masks = {}, {}
for key, path in RULE_FILES.items():
    p = pathlib.Path(path)
    if p.exists():
        configs[key] = json.loads(p.read_text())
        fused_masks[key] = eval_config(df, configs[key])

# ── 标题 & 总览 ──────────────────────────────────────────────
st.title("熔断规则横向对比")
st.caption(f"样本 {len(df)} 篇 · 推算总池 {N_TOTAL:,} 篇")

# 三列总览指标
header_cols = st.columns(len(configs))
for i, (key, mask) in enumerate(fused_masks.items()):
    keep = (~mask).sum()
    with header_cols[i]:
        st.markdown(f"### {LABELS[key]}")
        st.metric("全量分析", f"{keep} 篇", f"{keep/len(df):.1%}")
        st.metric("熔断丢弃", f"{mask.sum()} 篇", f"{mask.sum()/len(df):.1%}")
        st.metric("推算3万保留", f"{int(N_TOTAL*keep/len(df)):,} 篇")

st.divider()

# ── 各规则详情 ───────────────────────────────────────────────
st.subheader("规则明细")
detail_cols = st.columns(len(configs))
for i, (key, cfg) in enumerate(configs.items()):
    with detail_cols[i]:
        st.markdown(f"**{LABELS[key]}** · 规则间: `{cfg.get('inter_logic', 'OR')}`")
        for rule in cfg["rules"]:
            if not rule.get("enabled", True): continue
            cond_strs = []
            for c in rule["conditions"]:
                cond_strs.append(f"`{c['field']} {c['op']} {c['value']}`")
            logic = f" **{rule.get('internal_logic', 'AND')}** ".join(cond_strs)
            neg = " ~~取反~~" if rule.get("negate") else ""
            hit = eval_rule(df, rule).sum()
            st.markdown(f"- **{rule['name']}**{neg} → {hit}篇  \n  {logic}")

st.divider()

# ── Venn风格：论文集合交叉 ───────────────────────────────────
st.subheader("熔断集合交叉分析")
keys = list(fused_masks.keys())
if len(keys) == 3:
    a, b, c = fused_masks[keys[0]], fused_masks[keys[1]], fused_masks[keys[2]]
    sets = {
        "三方均熔断":      (a & b & c).sum(),
        f"仅{LABELS[keys[0]]}熔断":  (a & ~b & ~c).sum(),
        f"仅{LABELS[keys[1]]}熔断":  (~a & b & ~c).sum(),
        f"仅{LABELS[keys[2]]}熔断":  (~a & ~b & c).sum(),
        f"{LABELS[keys[0]]}+{LABELS[keys[1]]}":  (a & b & ~c).sum(),
        f"{LABELS[keys[0]]}+{LABELS[keys[2]]}":  (a & ~b & c).sum(),
        f"{LABELS[keys[1]]}+{LABELS[keys[2]]}":  (~a & b & c).sum(),
        "三方均保留":       (~a & ~b & ~c).sum(),
    }
    vcols = st.columns(4)
    for i, (label, count) in enumerate(sets.items()):
        vcols[i % 4].metric(label, f"{count} 篇", f"{count/len(df):.1%}")

st.divider()

# ── 分布对比图 ───────────────────────────────────────────────
st.subheader("各维度分布对比（绿=保留 红=熔断）")
for key, mask in fused_masks.items():
    st.markdown(f"**{LABELS[key]}**")
    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    bins = np.arange(-0.25, 10.75, 0.5)
    for i, abbr in enumerate(SCORE_FIELDS):
        ax = axes.flatten()[i]
        ax.hist(df.loc[~mask, abbr].dropna(), bins=bins, color="#4CAF50", alpha=0.8, label="保留")
        ax.hist(df.loc[mask,  abbr].dropna(), bins=bins, color="#f44336", alpha=0.6, label="熔断")
        ax.set_title(abbr, fontsize=9)
        ax.set_xlim(-0.5, 10.5)
        ax.tick_params(labelsize=7)
        if i == 0: ax.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── 保留论文差异明细 ─────────────────────────────────────────
st.subheader("各方保留而其他方熔断的论文")
for key, mask in fused_masks.items():
    others = [k for k in fused_masks if k != key]
    # 此方保留，但至少一个其他方熔断
    other_fused = fused_masks[others[0]]
    for ok in others[1:]:
        other_fused = other_fused | fused_masks[ok]
    exclusive_keep = (~mask) & other_fused
    if exclusive_keep.sum() > 0:
        with st.expander(f"{LABELS[key]} 保留但其他方熔断的 {exclusive_keep.sum()} 篇"):
            show = ["title", "venue", "mr", "tn", "er", "integrity"]
            st.dataframe(df[exclusive_keep][show].reset_index(drop=True), use_container_width=True)
