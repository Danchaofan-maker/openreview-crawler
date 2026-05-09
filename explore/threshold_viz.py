import json, pathlib, uuid
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="论文熔断规则配置", layout="wide")

CONFIG_PATH = pathlib.Path("explore/rules.json")

def save_config():
    CONFIG_PATH.write_text(json.dumps({
        "rules": st.session_state.rules,
        "inter_logic": st.session_state.inter_logic,
        "force_keep_hr": st.session_state.force_keep_hr,
    }, ensure_ascii=False, indent=2))

def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return None

# ── 数据加载 ────────────────────────────────────────────────
def _get_score(parsed, abbr, long_key):
    """兼容三格式: v0.7 长名+score / v0.8 完整 abbr+s / v0.8 紧凑 abbr=数字"""
    v = parsed.get(abbr)
    if v is None: v = parsed.get(long_key)
    if isinstance(v, dict): return v.get("s") if v.get("s") is not None else v.get("score")
    if isinstance(v, (int, float)): return v
    return None

def _get_flag(parsed, short, long_key):
    v = parsed.get(short)
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
    lines = pathlib.Path("data/llm_v4pro_thinking_N600_seed42.jsonl").read_text().strip().split("\n")
    records = []
    for l in lines:
        r = json.loads(l)
        parsed = r.get("parsed") or {}
        if not isinstance(parsed, dict):
            continue
        row = {"paper_id": r.get("paper_id"), "title": r.get("title", ""), "fuse": parsed.get("fuse")}
        for abbr, key in SCORE_FIELDS.items():
            row[abbr] = _get_score(parsed, abbr, key)
        row["marketing"]    = _get_flag(parsed, "mk", "marketing_detected")
        row["human_review"] = _get_flag(parsed, "hr", "human_review_required")
        row["integrity"]    = _get_integrity(parsed)
        records.append(row)
    return pd.DataFrame(records)

SCORE_FIELDS = {
    "mr": "mathematical_rigor", "tn": "theoretical_novelty",
    "md": "mathematical_depth", "ar": "assumption_realism",
    "er": "empirical_reliance", "tea": "theory_experiment_alignment",
    "cc": "compute_complexity", "ei": "epistemological_intent",
    "sg": "scope_generality", "cs": "confidence_score",
}
BOOL_FIELDS  = {"marketing": "marketing_detected", "human_review": "human_review_required"}
ENUM_FIELDS  = {"integrity": ["intact", "partial", "broken", "absent"]}
ALL_FIELDS   = list(SCORE_FIELDS) + list(BOOL_FIELDS) + list(ENUM_FIELDS)
FIELD_LABELS = {**SCORE_FIELDS, **BOOL_FIELDS, "integrity": "logical_chain.integrity"}

df = load_data()
N_TOTAL = 30983

# ── Session state 初始化 ─────────────────────────────────────
def new_condition():
    return {"id": str(uuid.uuid4()), "field": "mr", "op": "lt", "value": 3.0}

def new_rule():
    return {
        "id": str(uuid.uuid4()),
        "name": f"规则{len(st.session_state.rules)+1}",
        "enabled": True,
        "negate": False,
        "internal_logic": "AND",
        "conditions": [new_condition()],
    }

if "rules" not in st.session_state:
    saved = load_config()
    if saved:
        st.session_state.rules       = saved["rules"]
        st.session_state.inter_logic = saved.get("inter_logic", "OR")
        st.session_state.force_keep_hr = saved.get("force_keep_hr", True)
    else:
        st.session_state.rules = [
            {"id": str(uuid.uuid4()), "name": "数学严谨度极低", "enabled": True, "negate": False,
             "internal_logic": "AND", "conditions": [{"id": str(uuid.uuid4()), "field": "mr", "op": "lt", "value": 2.0}]},
            {"id": str(uuid.uuid4()), "name": "纯benchmark", "enabled": True, "negate": False,
             "internal_logic": "AND", "conditions": [
                 {"id": str(uuid.uuid4()), "field": "er", "op": "gt", "value": 7.5},
                 {"id": str(uuid.uuid4()), "field": "mr", "op": "lt", "value": 3.0},
             ]},
            {"id": str(uuid.uuid4()), "name": "逻辑链缺失", "enabled": True, "negate": False,
             "internal_logic": "AND", "conditions": [{"id": str(uuid.uuid4()), "field": "integrity", "op": "in", "value": ["absent", "broken"]}]},
        ]
        st.session_state.inter_logic   = "OR"
        st.session_state.force_keep_hr = True

# ── 条件求值 ─────────────────────────────────────────────────
def eval_cond(df, cond):
    f, op, v = cond["field"], cond["op"], cond["value"]
    col = df[f]
    if op == "lt":  return col.fillna(999)  < v
    if op == "lte": return col.fillna(999)  <= v
    if op == "gt":  return col.fillna(-999) > v
    if op == "gte": return col.fillna(-999) >= v
    if op == "eq":  return col.fillna(object()) == v
    if op == "neq": return col.fillna(object()) != v
    if op == "in":  return col.isin(v if isinstance(v, list) else [v])
    return pd.Series([False]*len(df))

def eval_rule(df, rule):
    masks = [eval_cond(df, c) for c in rule["conditions"]]
    if not masks:
        return pd.Series([False]*len(df))
    result = masks[0]
    for m in masks[1:]:
        result = (result & m) if rule["internal_logic"] == "AND" else (result | m)
    return ~result if rule.get("negate") else result

def eval_all(df, rules, inter_logic, force_keep_hr):
    active = [r for r in rules if r["enabled"]]
    if not active:
        return pd.Series([False]*len(df))
    hits = [eval_rule(df, r) for r in active]
    fused = hits[0]
    for h in hits[1:]:
        fused = (fused & h) if inter_logic == "AND" else (fused | h)
    if force_keep_hr:
        fused &= ~(df["human_review"].fillna(False) == True)
    return fused

# ── 条件编辑器 ───────────────────────────────────────────────
def condition_editor(rule_idx, cond, cond_idx, prefix):
    cols = st.columns([2, 1.5, 2.5, 0.5])
    field = cols[0].selectbox("字段", ALL_FIELDS,
        index=ALL_FIELDS.index(cond["field"]),
        format_func=lambda x: FIELD_LABELS.get(x, x),
        key=f"{prefix}_field")
    cond["field"] = field

    if field in SCORE_FIELDS:
        op_opts = {"<": "lt", "≤": "lte", ">": "gt", "≥": "gte", "=": "eq"}
        op_label = cols[1].selectbox("运算", list(op_opts),
            index=list(op_opts.values()).index(cond["op"]) if cond["op"] in op_opts.values() else 0,
            key=f"{prefix}_op")
        cond["op"] = op_opts[op_label]
        val = cols[2].slider("值", 0.0, 10.0,
            float(cond["value"]) if isinstance(cond["value"], (int, float)) else 0.0,
            step=0.5, key=f"{prefix}_val")
        cond["value"] = val
    elif field in BOOL_FIELDS:
        cond["op"] = "eq"
        val = cols[1].radio("值", [True, False],
            index=0 if cond.get("value") is True else 1,
            horizontal=True, key=f"{prefix}_val")
        cond["value"] = val
        cols[2].empty()
    else:  # integrity
        cond["op"] = "in"
        opts = ENUM_FIELDS["integrity"]
        val = cols[2].multiselect("属于", opts,
            default=cond["value"] if isinstance(cond["value"], list) else opts[:2],
            key=f"{prefix}_val")
        cond["value"] = val
        cols[1].empty()

    if cols[3].button("✕", key=f"{prefix}_del"):
        st.session_state.rules[rule_idx]["conditions"].pop(cond_idx)
        st.rerun()

# ── 规则编辑面板 ─────────────────────────────────────────────
st.title("熔断规则配置")

# 规则间逻辑 + 全局设置
top = st.columns([2, 2, 3])
st.session_state.inter_logic = top[0].radio(
    "规则间逻辑", ["OR（任一命中→熔断）", "AND（全部命中→熔断）"],
    index=0 if st.session_state.inter_logic == "OR" else 1,
    horizontal=True
).split("（")[0]
st.session_state.force_keep_hr = top[1].checkbox("human_review=True 强制保留", value=st.session_state.force_keep_hr)

rules_to_delete = []
for ri, rule in enumerate(st.session_state.rules):
    with st.expander(f"{'✅' if rule['enabled'] else '⬜'} {rule['name']}", expanded=True):
        h1, h2, h3, h4, h5 = st.columns([2.5, 1.2, 1.2, 1.2, 0.8])
        rule["name"] = h1.text_input("规则名", rule["name"], key=f"rname_{rule['id']}")
        rule["enabled"] = h2.checkbox("启用", rule["enabled"], key=f"ren_{rule['id']}")
        rule["negate"] = h3.checkbox("取反(NOT)", rule["negate"], key=f"rneg_{rule['id']}")
        rule["internal_logic"] = h4.radio("条件间", ["AND", "OR"],
            index=0 if rule["internal_logic"] == "AND" else 1,
            horizontal=True, key=f"rlog_{rule['id']}")
        if h5.button("删除规则", key=f"rdel_{rule['id']}"):
            rules_to_delete.append(ri)

        for ci, cond in enumerate(rule["conditions"]):
            condition_editor(ri, cond, ci, f"c_{rule['id']}_{cond['id']}")

        if st.button("＋ 添加条件", key=f"radd_{rule['id']}"):
            rule["conditions"].append(new_condition())
            st.rerun()

for ri in sorted(rules_to_delete, reverse=True):
    st.session_state.rules.pop(ri)
if rules_to_delete:
    st.rerun()

btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 4])
if btn1.button("＋ 新增规则"):
    st.session_state.rules.append(new_rule())
    st.rerun()
if btn2.button("💾 保存配置"):
    save_config()
    st.success("已保存到 explore/rules.json")

config_json = json.dumps({
    "rules": st.session_state.rules,
    "inter_logic": st.session_state.inter_logic,
    "force_keep_hr": st.session_state.force_keep_hr,
}, ensure_ascii=False, indent=2)
btn3.download_button("📤 导出", config_json, "rules.json", "application/json")

uploaded = st.file_uploader("📥 导入规则配置", type="json", label_visibility="collapsed")
if uploaded:
    imported = json.loads(uploaded.read())
    st.session_state.rules         = imported["rules"]
    st.session_state.inter_logic   = imported.get("inter_logic", "OR")
    st.session_state.force_keep_hr = imported.get("force_keep_hr", True)
    save_config()
    st.rerun()

# ── 结果计算 ─────────────────────────────────────────────────
st.divider()
fused_mask = eval_all(df, st.session_state.rules, st.session_state.inter_logic, st.session_state.force_keep_hr)
keep_mask  = ~fused_mask
keep_ratio = keep_mask.sum() / len(df)

m = st.columns(4)
m[0].metric("全量分析", f"{keep_mask.sum()} 篇", f"{keep_ratio:.1%}")
m[1].metric("熔断输出", f"{fused_mask.sum()} 篇", f"{1-keep_ratio:.1%}")
m[2].metric("推算3万→全量", f"{int(N_TOTAL*keep_ratio):,} 篇")
m[3].metric("推算3万→熔断", f"{int(N_TOTAL*(1-keep_ratio)):,} 篇")

# 各规则命中数
active_rules = [r for r in st.session_state.rules if r["enabled"]]
if active_rules:
    st.subheader("各规则命中")
    rcols = st.columns(len(active_rules))
    for i, rule in enumerate(active_rules):
        hit = eval_rule(df, rule).sum()
        rcols[i].metric(rule["name"], f"{hit} 篇")

# 分布图
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for i, (abbr, name) in enumerate(SCORE_FIELDS.items()):
    ax = axes.flatten()[i]
    ax.hist(df.loc[keep_mask,  abbr].dropna(), bins=np.arange(-0.25,10.75,0.5), color="#4CAF50", alpha=0.8, label="全量")
    ax.hist(df.loc[fused_mask, abbr].dropna(), bins=np.arange(-0.25,10.75,0.5), color="#f44336", alpha=0.6, label="熔断")
    ax.set_title(f"{abbr} ({name[:16]})", fontsize=9)
    ax.set_xlim(-0.5, 10.5)
    ax.tick_params(labelsize=8)
    if i == 0:
        ax.legend(fontsize=7)
plt.tight_layout()
st.pyplot(fig)

# 明细
with st.expander(f"熔断 {fused_mask.sum()} 篇"):
    fd = df[fused_mask].copy()
    for r in active_rules:
        fd[r["name"]] = eval_rule(df, r)[fused_mask].values
    show = ["title","mr","er","cs","integrity","marketing"] + [r["name"] for r in active_rules]
    st.dataframe(fd[show].reset_index(drop=True), use_container_width=True)

with st.expander(f"全量分析 {keep_mask.sum()} 篇"):
    show = ["title"] + list(SCORE_FIELDS) + ["integrity","marketing","human_review"]
    st.dataframe(df[keep_mask][show].reset_index(drop=True), use_container_width=True)
