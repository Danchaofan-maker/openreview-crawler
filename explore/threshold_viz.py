import json, pathlib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="论文过滤阈值", layout="wide")

@st.cache_data
def load_data():
    lines = pathlib.Path("data/llm_v4pro_thinking_N600_seed42.jsonl").read_text().strip().split("\n")
    records = []
    for l in lines:
        r = json.loads(l)
        parsed = r.get("parsed") or {}
        if not isinstance(parsed, dict):
            continue
        row = {"paper_id": r.get("paper_id"), "title": r.get("title", "")}
        score_fields = [
            ("mr", "mathematical_rigor"),
            ("tn", "theoretical_novelty"),
            ("md", "mathematical_depth"),
            ("ar", "assumption_realism"),
            ("er", "empirical_reliance"),
            ("tea", "theory_experiment_alignment"),
            ("cc", "compute_complexity"),
            ("ei", "epistemological_intent"),
            ("sg", "scope_generality"),
            ("cs", "confidence_score"),
        ]
        for abbr, key in score_fields:
            val = parsed.get(key, {})
            row[abbr] = val.get("score") if isinstance(val, dict) else None
        mk = parsed.get("marketing_detected", {})
        row["marketing"] = mk.get("flag") if isinstance(mk, dict) else None
        hr = parsed.get("human_review_required", {})
        row["human_review"] = hr.get("flag") if isinstance(hr, dict) else None
        lc = parsed.get("logical_chain", {})
        row["integrity"] = lc.get("integrity") if isinstance(lc, dict) else None
        records.append(row)
    return pd.DataFrame(records)

df = load_data()
N_TOTAL = 30983

SCORE_FIELDS = {
    "mr": "mathematical_rigor", "tn": "theoretical_novelty",
    "md": "mathematical_depth", "ar": "assumption_realism",
    "er": "empirical_reliance", "tea": "theory_experiment_alignment",
    "cc": "compute_complexity", "ei": "epistemological_intent",
    "sg": "scope_generality", "cs": "confidence_score",
}

# ---------- 熔断规则定义 ----------
# 每条规则: {"name": str, "conditions": list of (field, op, value)}
# op: "lt" | "gt" | "eq" | "in" (for integrity)
DEFAULT_RULES = [
    {
        "name": "纯benchmark / 无理论",
        "conditions": [("er", "gt", 7.5), ("mr", "lt", 3.0)],
        "logic": "AND",
    },
    {
        "name": "数学严谨度极低",
        "conditions": [("mr", "lt", 2.0)],
        "logic": "AND",
    },
    {
        "name": "逻辑链缺失",
        "conditions": [("integrity", "in", ["absent", "broken"])],
        "logic": "AND",
    },
    {
        "name": "营销包装",
        "conditions": [("marketing", "eq", True)],
        "logic": "AND",
    },
    {
        "name": "置信度过低",
        "conditions": [("cs", "lt", 3.0)],
        "logic": "AND",
    },
]

def eval_condition(df, field, op, value):
    col = df[field]
    if op == "lt":
        return col.fillna(999) < value
    elif op == "gt":
        return col.fillna(-999) > value
    elif op == "eq":
        return col.fillna(False) == value
    elif op == "in":
        return col.isin(value)
    return pd.Series([False] * len(df))

def eval_rule(df, rule):
    masks = [eval_condition(df, f, op, v) for f, op, v in rule["conditions"]]
    result = masks[0]
    for m in masks[1:]:
        if rule["logic"] == "AND":
            result = result & m
        else:
            result = result | m
    return result

# ---------- 侧边栏 ----------
st.sidebar.header("熔断规则（命中任意一条→熔断）")

active_rules = []
for i, rule in enumerate(DEFAULT_RULES):
    enabled = st.sidebar.checkbox(f"规则{i+1}: {rule['name']}", value=True, key=f"rule_{i}")
    if enabled:
        active_rules.append(rule)

st.sidebar.divider()
st.sidebar.header("强制保留")
force_keep_human_review = st.sidebar.checkbox("human_review=True 强制保留", value=True)

# ---------- 计算 ----------
# 每篇论文命中哪条规则
rule_hits = pd.DataFrame({
    f"规则{i+1}_{r['name']}": eval_rule(df, r)
    for i, r in enumerate(DEFAULT_RULES)
})

fused_mask = pd.Series([False] * len(df))
for rule in active_rules:
    fused_mask |= eval_rule(df, rule)

if force_keep_human_review:
    fused_mask &= ~(df["human_review"].fillna(False) == True)

keep_mask = ~fused_mask
keep_ratio = keep_mask.sum() / len(df)

# ---------- 顶部指标 ----------
st.title("论文熔断规则可视化")
st.caption(f"样本 {len(df)} 篇 · 总池 {N_TOTAL:,} 篇")

cols = st.columns(4)
cols[0].metric("全量分析", f"{keep_mask.sum()} 篇", f"{keep_ratio:.1%}")
cols[1].metric("熔断输出", f"{fused_mask.sum()} 篇", f"{1-keep_ratio:.1%}")
cols[2].metric("推算3万篇全量", f"{int(N_TOTAL * keep_ratio):,} 篇")
cols[3].metric("推算3万篇熔断", f"{int(N_TOTAL * (1-keep_ratio)):,} 篇")

# ---------- 各规则命中统计 ----------
st.subheader("各规则命中情况")
rule_cols = st.columns(len(DEFAULT_RULES))
for i, rule in enumerate(DEFAULT_RULES):
    hit = eval_rule(df, rule).sum()
    rule_cols[i].metric(f"规则{i+1}", f"{hit} 篇", rule["name"])

# ---------- 分布图（按熔断/全量着色）----------
st.subheader("各维度分布")
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for i, (abbr, name) in enumerate(SCORE_FIELDS.items()):
    ax = axes[i]
    col_keep = df.loc[keep_mask, abbr].dropna()
    col_fuse = df.loc[fused_mask, abbr].dropna()
    bins = np.arange(-0.25, 10.75, 0.5)
    ax.hist(col_keep, bins=bins, color="#4CAF50", alpha=0.8, label="全量")
    ax.hist(col_fuse, bins=bins, color="#f44336", alpha=0.6, label="熔断")
    ax.set_title(f"{abbr} ({name[:16]})", fontsize=9)
    ax.set_xlim(-0.5, 10.5)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
st.pyplot(fig)

# ---------- 熔断论文明细 ----------
with st.expander(f"熔断的 {fused_mask.sum()} 篇（命中规则）"):
    fused_df = df[fused_mask].copy()
    for i, rule in enumerate(DEFAULT_RULES):
        fused_df[f"R{i+1}"] = eval_rule(df, rule)[fused_mask].values
    show_cols = ["title", "mr", "er", "cs", "integrity", "marketing"] + [f"R{i+1}" for i in range(len(DEFAULT_RULES))]
    st.dataframe(fused_df[show_cols].reset_index(drop=True), use_container_width=True)

with st.expander(f"全量分析的 {keep_mask.sum()} 篇"):
    show_cols = ["title"] + list(SCORE_FIELDS.keys()) + ["integrity", "marketing", "human_review"]
    st.dataframe(df[keep_mask][show_cols].reset_index(drop=True), use_container_width=True)
