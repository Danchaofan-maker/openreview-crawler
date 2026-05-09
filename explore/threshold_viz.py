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
        fields = [
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
        for abbr, key in fields:
            val = parsed.get(key, {})
            row[abbr] = val.get("score") if isinstance(val, dict) else None
        # 布尔字段
        mk = parsed.get("marketing_detected", {})
        row["marketing"] = mk.get("flag") if isinstance(mk, dict) else None
        hr = parsed.get("human_review_required", {})
        row["human_review"] = hr.get("flag") if isinstance(hr, dict) else None
        # logical_chain integrity
        lc = parsed.get("logical_chain", {})
        row["integrity"] = lc.get("integrity") if isinstance(lc, dict) else None
        records.append(row)
    return pd.DataFrame(records)

df = load_data()
N_TOTAL = 30983

st.title("论文过滤阈值可视化")
st.caption(f"当前样本 {len(df)} 篇 · 总池 {N_TOTAL:,} 篇")

FIELDS = {
    "mr":  "mathematical_rigor",
    "tn":  "theoretical_novelty",
    "md":  "mathematical_depth",
    "ar":  "assumption_realism",
    "er":  "empirical_reliance",
    "tea": "theory_experiment_alignment",
    "cc":  "compute_complexity",
    "ei":  "epistemological_intent",
    "sg":  "scope_generality",
}

st.sidebar.header("数值阈值（低于此值→丢弃）")
thresholds = {}
for abbr, name in FIELDS.items():
    col = df[abbr].dropna()
    thresholds[abbr] = st.sidebar.slider(
        name, 0.0, 10.0, 0.0, step=0.5,
        help=f"min={col.min():.1f} mean={col.mean():.1f} max={col.max():.1f}"
    )

st.sidebar.header("布尔过滤")
filter_marketing = st.sidebar.checkbox("丢弃 marketing_detected=True", value=False)

st.sidebar.header("logical_chain integrity")
INTEGRITY_ORDER = ["intact", "partial", "broken", "absent"]
integrity_counts = df["integrity"].value_counts()
keep_integrity = st.sidebar.multiselect(
    "保留以下等级",
    options=INTEGRITY_ORDER,
    default=INTEGRITY_ORDER,
    format_func=lambda x: f"{x} ({integrity_counts.get(x, 0)}篇)"
)

# 过滤逻辑
mask_keep = pd.Series([True] * len(df))
for abbr, thresh in thresholds.items():
    if thresh > 0:
        mask_keep &= df[abbr].fillna(0) >= thresh
if filter_marketing:
    mask_keep &= df["marketing"].fillna(False) == False
# human_review_required=True 强制保留，覆盖其他过滤条件
mask_keep |= df["human_review"].fillna(False) == True
if keep_integrity:
    mask_keep &= df["integrity"].isin(keep_integrity)

n_keep = mask_keep.sum()
n_drop = len(df) - n_keep
keep_ratio = n_keep / len(df)

n_marketing = df["marketing"].fillna(False).sum()
n_human_review = df["human_review"].fillna(False).sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("保留", f"{n_keep} 篇", f"{keep_ratio:.1%}")
col2.metric("丢弃", f"{n_drop} 篇", f"{1-keep_ratio:.1%}")
col3.metric("推算到3万篇保留", f"{int(N_TOTAL * keep_ratio):,} 篇")
col4.metric("marketing=True", f"{n_marketing} 篇", f"{n_marketing/len(df):.1%}")
col5.metric("human_review=True", f"{n_human_review} 篇", f"{n_human_review/len(df):.1%}")

# 分布图
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
axes = axes.flatten()

for i, (abbr, name) in enumerate(FIELDS.items()):
    ax = axes[i]
    col = df[abbr].dropna()
    bins = np.arange(-0.25, 10.75, 0.5)
    ax.hist(col[mask_keep[col.index]], bins=bins, color="#4CAF50", alpha=0.8, label="保留")
    ax.hist(col[~mask_keep[col.index]], bins=bins, color="#f44336", alpha=0.6, label="丢弃")
    thresh = thresholds[abbr]
    if thresh > 0:
        ax.axvline(thresh, color="red", linestyle="--", linewidth=1.5)
    ax.set_title(f"{abbr} ({name[:18]})", fontsize=9)
    ax.set_xlim(-0.5, 10.5)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
st.pyplot(fig)

# 保留论文列表
with st.expander(f"保留的 {n_keep} 篇论文"):
    show_cols = ["title"] + list(FIELDS.keys()) + ["integrity", "marketing", "human_review"]
    st.dataframe(df[mask_keep][show_cols].reset_index(drop=True), use_container_width=True)
