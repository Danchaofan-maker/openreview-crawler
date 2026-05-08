"""论文打分可视化 + 双盲人工复核 (Streamlit)

启动:
  uv run streamlit run scripts/viz.py

四个模式:
  - Dashboard 总览:全集统计、分布直方图、按维度过滤
  - 单篇详情:看 LLM 给某一篇的所有打分 + chain + reasoning
  - 双盲打分:协作者各自独立给"是否筛掉"打标,默认隐藏 LLM 分数
  - 对照视图:LLM 判定 vs 各审阅者打标,找分歧

数据源:
  data/llm_test_run.jsonl       — LLM 输出
  data/sample_20.json           — 摘要查询(扩展时可指向 raw)
  data/human_reviews/<name>.jsonl — 每位审阅者一份(自动追加)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- 配置 ----------
LLM_PATH = "data/llm_test_run.jsonl"
SAMPLE_PATH = "data/sample_20.json"
RAW_DIR = "data/raw"
HUMAN_DIR = Path("data/human_reviews")

DIMS = [
    "mathematical_rigor",
    "theoretical_novelty",
    "mathematical_depth",
    "assumption_realism",
    "empirical_reliance",
    "theory_experiment_alignment",
    "compute_complexity",
    "epistemological_intent",
    "scope_generality",
]
DIM_CN = {
    "mathematical_rigor": "数学严密度",
    "theoretical_novelty": "理论新颖度",
    "mathematical_depth": "数学深度",
    "assumption_realism": "假设现实度",
    "empirical_reliance": "经验依赖",
    "theory_experiment_alignment": "理论实验咬合",
    "compute_complexity": "算力门槛",
    "epistemological_intent": "认识论意图",
    "scope_generality": "适用广度",
}

# ---------- 数据加载 ----------
@st.cache_data(show_spinner=False)
def load_llm_rows() -> list[dict]:
    if not os.path.exists(LLM_PATH):
        return []
    rows = []
    with open(LLM_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


@st.cache_data(show_spinner=False)
def load_abstract_map() -> dict[str, dict]:
    """paper_id -> {abstract, ...metadata}。先读 sample_20,再从 raw 补全。"""
    out: dict[str, dict] = {}
    if os.path.exists(SAMPLE_PATH):
        d = json.loads(Path(SAMPLE_PATH).read_text(encoding="utf-8"))
        for s in d.get("samples", []):
            out[s["paper_id"]] = s
    if os.path.isdir(RAW_DIR):
        for fname in os.listdir(RAW_DIR):
            if not fname.endswith(".jsonl"):
                continue
            try:
                with open(os.path.join(RAW_DIR, fname), encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            p = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if p.get("paper_id") and p["paper_id"] not in out:
                            out[p["paper_id"]] = p
            except Exception:
                continue
    return out


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    records = []
    for r in rows:
        if not r.get("ok"):
            continue
        p = r.get("parsed") or {}
        rec: dict = {
            "paper_id": r.get("paper_id"),
            "title": r.get("title", ""),
            "venue": r.get("venue", ""),
            "year": r.get("year"),
            "domain_modality": p.get("domain_modality") or "",
            "integrity": (p.get("logical_chain") or {}).get("integrity") or "",
            "marketing": bool((p.get("marketing_detected") or {}).get("flag")),
            "human_review_required": bool((p.get("human_review_required") or {}).get("flag")),
            "out_of_scope_notes": p.get("out_of_scope_notes") or "",
            "confidence": (p.get("confidence_score") or {}).get("score"),
            "elapsed_s": r.get("elapsed_s"),
        }
        for d in DIMS:
            v = p.get(d)
            rec[d] = v.get("score") if isinstance(v, dict) else None
        records.append(rec)
    return pd.DataFrame(records)


# ---------- 人工 review 持久化 ----------
def review_path(reviewer: str) -> Path:
    safe = "".join(c for c in reviewer if c.isalnum() or c in "._-").strip("._-")
    return HUMAN_DIR / f"{safe or 'unknown'}.jsonl"


def load_reviews(reviewer: str) -> dict[str, dict]:
    """返回 paper_id -> 最新一条记录"""
    path = review_path(reviewer)
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["paper_id"]] = rec
            except json.JSONDecodeError:
                continue
    return out


def save_review(reviewer: str, paper_id: str, filter_out: bool, note: str):
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    path = review_path(reviewer)
    rec = {
        "paper_id": paper_id,
        "reviewer": reviewer,
        "filter_out": filter_out,
        "note": note,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def list_reviewers() -> list[str]:
    if not HUMAN_DIR.exists():
        return []
    return sorted(p.stem for p in HUMAN_DIR.glob("*.jsonl"))


# ---------- UI ----------
st.set_page_config(page_title="论文打分可视化", layout="wide", initial_sidebar_state="expanded")
st.title("论文打分可视化 · 双盲复核")

rows = load_llm_rows()
if not rows:
    st.error(f"未找到 LLM 输出文件 `{LLM_PATH}`。先跑测试脚本生成。")
    st.stop()

df = rows_to_df(rows)
abs_map = load_abstract_map()

mode = st.sidebar.radio(
    "模式",
    ["📊 Dashboard 总览", "🔬 单篇详情", "🕶️ 双盲打分", "⚖️ 对照视图"],
)

# ============================================================
# Mode 1: Dashboard
# ============================================================
if mode.startswith("📊"):
    st.header("Dashboard 总览")
    st.caption(f"数据源:`{LLM_PATH}` · 共 {len(df)} 篇成功打分 / {len(rows)} 总")

    with st.sidebar:
        st.markdown("### 过滤器")
        venues = sorted(df["venue"].dropna().unique().tolist())
        f_venues = st.multiselect("venue", venues, default=venues)
        f_integrity = st.multiselect(
            "integrity",
            ["intact", "partial", "broken", "absent"],
            default=["intact", "partial", "broken", "absent"],
        )
        years = sorted([int(y) for y in df["year"].dropna().unique().tolist()])
        if years:
            f_year = st.slider("年份范围", min(years), max(years), (min(years), max(years)))
        else:
            f_year = (None, None)
        f_rigor_min = st.slider("mathematical_rigor 最低值", 0.0, 10.0, 0.0, 0.5)
        f_human_review = st.selectbox("human_review_required", ["全部", "True", "False"])
        f_marketing = st.selectbox("marketing_detected", ["全部", "True", "False"])
        f_out_of_scope = st.checkbox("仅看含 out_of_scope_notes 的", value=False)
        search = st.text_input("标题/domain 搜索(模糊)").strip().lower()

    fdf = df[df["venue"].isin(f_venues)]
    fdf = fdf[fdf["integrity"].isin(f_integrity)]
    if f_year != (None, None):
        fdf = fdf[(fdf["year"].fillna(0) >= f_year[0]) & (fdf["year"].fillna(0) <= f_year[1])]
    fdf = fdf[fdf["mathematical_rigor"].fillna(0) >= f_rigor_min]
    if f_human_review != "全部":
        fdf = fdf[fdf["human_review_required"] == (f_human_review == "True")]
    if f_marketing != "全部":
        fdf = fdf[fdf["marketing"] == (f_marketing == "True")]
    if f_out_of_scope:
        fdf = fdf[fdf["out_of_scope_notes"].str.len() > 0]
    if search:
        fdf = fdf[
            fdf["title"].str.lower().str.contains(search, na=False)
            | fdf["domain_modality"].str.lower().str.contains(search, na=False)
        ]

    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("过滤后", len(fdf), delta=f"全集 {len(df)}")
    c2.metric("intact 占比", f"{(fdf['integrity'] == 'intact').mean()*100:.0f}%" if len(fdf) else "—")
    c3.metric("rigor 均值", f"{fdf['mathematical_rigor'].mean():.1f}" if len(fdf) else "—")
    c4.metric("novelty 均值", f"{fdf['theoretical_novelty'].mean():.1f}" if len(fdf) else "—")
    c5.metric("human_review", int(fdf["human_review_required"].sum()) if len(fdf) else 0)
    c6.metric("marketing", int(fdf["marketing"].sum()) if len(fdf) else 0)

    if len(fdf) == 0:
        st.warning("当前过滤器下无数据。")
    else:
        st.divider()
        st.subheader("9 维度分布")
        dim_cols = st.columns(3)
        for i, d in enumerate(DIMS):
            with dim_cols[i % 3]:
                vals = fdf[d].dropna()
                if len(vals):
                    fig = px.histogram(
                        vals, nbins=11, range_x=[-0.5, 10.5],
                        title=f"{DIM_CN[d]}  (n={len(vals)} avg={vals.mean():.1f})",
                    )
                    fig.update_layout(height=220, showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption(f"{DIM_CN[d]} 无数据")

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("integrity 分布")
            integ_counts = fdf["integrity"].value_counts().reset_index()
            integ_counts.columns = ["integrity", "count"]
            fig = px.pie(integ_counts, names="integrity", values="count", hole=0.4)
            fig.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.subheader("Top 15 学术子领域")
            doms = fdf["domain_modality"].value_counts().head(15).reset_index()
            doms.columns = ["domain", "count"]
            fig = px.bar(doms, x="count", y="domain", orientation="h")
            fig.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20), yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("rigor × novelty 散点")
        fig = px.scatter(
            fdf, x="mathematical_rigor", y="theoretical_novelty",
            color="integrity", hover_data=["paper_id", "title", "venue"],
            range_x=[-0.5, 10.5], range_y=[-0.5, 10.5],
        )
        fig.update_layout(height=420, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader(f"论文列表 ({len(fdf)})")
        show_cols = ["paper_id", "venue", "year", "title", "integrity"] + DIMS + ["domain_modality"]
        st.dataframe(fdf[show_cols], use_container_width=True, height=400)

        csv = fdf[["paper_id"]].to_csv(index=False).encode("utf-8")
        st.download_button("📥 导出当前 paper_id 列表 (csv)", csv, "filtered_paper_ids.csv", "text/csv")

# ============================================================
# Mode 2: 单篇详情
# ============================================================
elif mode.startswith("🔬"):
    st.header("单篇详情")
    paper_id = st.selectbox("paper_id", df["paper_id"].tolist(), format_func=lambda x: f"{x} — {df[df.paper_id==x].iloc[0]['title'][:80]}")
    row = next((r for r in rows if r.get("paper_id") == paper_id), None)
    if row is None:
        st.error("未找到")
        st.stop()
    p = row.get("parsed") or {}

    st.subheader(row.get("title", ""))
    st.caption(f"{row.get('venue', '')}  ·  {row.get('year', '')}  ·  耗时 {row.get('elapsed_s')}s  ·  reasoning {row.get('reasoning_chars', 0)} 字")

    if paper_id in abs_map:
        with st.expander("📄 原始摘要", expanded=False):
            st.write(abs_map[paper_id].get("abstract", "(无)"))

    # Radar
    st.subheader("9 维度雷达图")
    scores = []
    for d in DIMS:
        v = p.get(d)
        s = v.get("score") if isinstance(v, dict) else None
        scores.append(s if s is not None else 0)
    fig = go.Figure(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=[DIM_CN[d] for d in DIMS] + [DIM_CN[DIMS[0]]],
        fill="toself",
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), showlegend=False, height=440)
    st.plotly_chart(fig, use_container_width=True)

    # 各维度 score + rationale
    st.subheader("9 维度评分与理由")
    for d in DIMS:
        v = p.get(d)
        if isinstance(v, dict):
            s = v.get("score")
            rationale = v.get("rationale", "")
            label = "null" if s is None else str(s)
            st.markdown(f"**{DIM_CN[d]}** ({d}) — `{label}`")
            st.caption(rationale)
        else:
            st.markdown(f"**{DIM_CN[d]}** ({d}) — `null`")

    # logical_chain
    st.subheader("logical_chain 数学骨架")
    lc = p.get("logical_chain") or {}
    integ = lc.get("integrity", "?")
    color = {"intact": "🟢", "partial": "🟡", "broken": "🔴", "absent": "⚫"}.get(integ, "⚪")
    st.markdown(f"### {color} integrity: `{integ}`")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Premise (前提)**")
        st.info(lc.get("premise") or "(空)")
        st.markdown("**Tools (工具)**")
        tools = lc.get("tools") or []
        if tools:
            st.write(tools)
        else:
            st.caption("(空)")
    with cc2:
        st.markdown("**Derivation Outline (推导骨架)**")
        st.info(lc.get("derivation_outline") or "(空)")
        st.markdown("**Conclusion (结论)**")
        st.info(lc.get("conclusion") or "(空)")

    # 元字段
    st.subheader("辅助 + 元字段")
    cm1, cm2, cm3 = st.columns(3)
    with cm1:
        st.markdown("**domain_modality**")
        st.code(p.get("domain_modality", ""))
        st.markdown("**confidence**")
        cs = p.get("confidence_score") or {}
        st.write(f"{cs.get('score')} — {cs.get('rationale', '')}")
    with cm2:
        st.markdown("**marketing_detected**")
        m = p.get("marketing_detected") or {}
        st.write(f"{m.get('flag')} — {m.get('rationale', '')}")
        st.markdown("**human_review_required**")
        h = p.get("human_review_required") or {}
        st.write(f"{h.get('flag')} — {h.get('rationale', '')}")
    with cm3:
        st.markdown("**out_of_scope_notes**")
        st.write(p.get("out_of_scope_notes") or "(空)")

    with st.expander("🧠 reasoning_content (内部 CoT 日志)"):
        st.text(row.get("reasoning", "") or "(空)")

    with st.expander("📦 完整原始 JSON 输出"):
        st.json(p)

# ============================================================
# Mode 3: 双盲打分
# ============================================================
elif mode.startswith("🕶️"):
    st.header("🕶️ 双盲打分")
    st.caption("⚠️ 此模式下隐藏 LLM 分数与他人打标,仅根据摘要独立判断「是否筛掉」。")

    if "reviewer" not in st.session_state:
        st.session_state.reviewer = ""
    if "blind_idx" not in st.session_state:
        st.session_state.blind_idx = 0
    if "skip_reviewed" not in st.session_state:
        st.session_state.skip_reviewed = True

    existing = list_reviewers()
    cn1, cn2 = st.columns([2, 1])
    with cn1:
        reviewer_input = st.text_input(
            "你的标识(英文/拼音,提交后保存到 `data/human_reviews/<标识>.jsonl`)",
            value=st.session_state.reviewer or (existing[0] if existing else ""),
        )
    with cn2:
        st.session_state.skip_reviewed = st.checkbox("跳过已打过的", value=st.session_state.skip_reviewed)
    if existing:
        st.caption(f"已有审阅者档案:{', '.join(existing)}")

    if not reviewer_input.strip():
        st.warning("请先输入你的标识开始打分。")
        st.stop()

    reviewer = reviewer_input.strip()
    st.session_state.reviewer = reviewer
    my_reviews = load_reviews(reviewer)

    # 当前论文集
    paper_ids = df["paper_id"].tolist()
    if st.session_state.skip_reviewed:
        unreviewed = [pid for pid in paper_ids if pid not in my_reviews]
    else:
        unreviewed = paper_ids
    total_reviewed = len(my_reviews)

    cstat1, cstat2, cstat3 = st.columns(3)
    cstat1.metric("已打标", f"{total_reviewed} / {len(paper_ids)}")
    cstat2.metric("筛掉", sum(1 for r in my_reviews.values() if r.get("filter_out")))
    cstat3.metric("保留", sum(1 for r in my_reviews.values() if not r.get("filter_out")))

    if not unreviewed:
        st.success("✅ 全部已打标完成")
        st.stop()

    # 当前 idx clamp 到 unreviewed 范围
    if st.session_state.blind_idx >= len(unreviewed):
        st.session_state.blind_idx = 0
    cur_pid = unreviewed[st.session_state.blind_idx]
    cur_row = next((r for r in rows if r.get("paper_id") == cur_pid), None)
    if cur_row is None:
        st.error("当前条目找不到")
        st.stop()

    st.divider()
    st.markdown(f"**[{st.session_state.blind_idx + 1} / {len(unreviewed)}]**  paper_id: `{cur_pid}`")
    st.subheader(cur_row.get("title", ""))
    st.caption(f"{cur_row.get('venue', '')}  ·  {cur_row.get('year', '')}")

    abstract = (abs_map.get(cur_pid) or {}).get("abstract", "(摘要未在缓存中找到)")
    st.markdown("### 摘要")
    st.write(abstract)

    # 历史记录(自己之前打过的可见)
    if cur_pid in my_reviews:
        prev = my_reviews[cur_pid]
        st.info(f"你之前已打过:filter_out=**{prev.get('filter_out')}**,note: {prev.get('note', '')}")

    note = st.text_area("备注(可选,会保留下来)", key=f"note_{cur_pid}", height=80)

    bc1, bc2, bc3, bc4 = st.columns([1, 1, 1, 1])
    if bc1.button("❌ 筛掉 (filter_out=True)", use_container_width=True, type="primary"):
        save_review(reviewer, cur_pid, True, note)
        st.session_state.blind_idx += 1
        st.rerun()
    if bc2.button("✅ 保留 (filter_out=False)", use_container_width=True, type="primary"):
        save_review(reviewer, cur_pid, False, note)
        st.session_state.blind_idx += 1
        st.rerun()
    if bc3.button("⏭️ 跳过(不写入)", use_container_width=True):
        st.session_state.blind_idx += 1
        st.rerun()
    if bc4.button("⏮️ 上一篇", use_container_width=True):
        st.session_state.blind_idx = max(0, st.session_state.blind_idx - 1)
        st.rerun()

# ============================================================
# Mode 4: 对照视图
# ============================================================
elif mode.startswith("⚖️"):
    st.header("⚖️ 对照视图:LLM × 各审阅者")
    existing = list_reviewers()
    if not existing:
        st.info("暂无审阅者档案。先去「双盲打分」生成。")
        st.stop()

    reviewer_data = {name: load_reviews(name) for name in existing}
    paper_rows = {r["paper_id"]: r for r in rows}

    # ---- 配置:可调 LLM filter hint ----
    with st.sidebar:
        st.markdown("### LLM filter hint 阈值")
        hint_rigor_max = st.slider("rigor 上限(<= 即 filter_out)", 0.0, 10.0, 3.0, 0.5)
        hint_absent = st.checkbox("integrity=absent 即 filter_out", value=True)
        hint_marketing = st.checkbox("marketing=true 即 filter_out", value=False)
        st.caption("LLM_hint = (integrity=absent ∧ 上勾) ∨ (rigor < 阈值) ∨ (marketing ∧ 上勾)")

    def llm_hint(r) -> bool:
        rigor = r["mathematical_rigor"] or 0
        if hint_absent and r["integrity"] == "absent":
            return True
        if rigor < hint_rigor_max:
            return True
        if hint_marketing and r.get("marketing"):
            return True
        return False

    # ---- 构建主表 ----
    rows_table = []
    for _, r in df.iterrows():
        pid = r["paper_id"]
        rec = {
            "paper_id": pid,
            "title": r["title"],
            "venue": r["venue"],
            "year": r["year"],
            "LLM_integrity": r["integrity"],
            "LLM_rigor": r["mathematical_rigor"],
            "LLM_novelty": r["theoretical_novelty"],
            "LLM_marketing": r["marketing"],
            "LLM_hint_filter_out": llm_hint(r),
        }
        for name in existing:
            v = reviewer_data[name].get(pid)
            rec[f"{name}_filter_out"] = v["filter_out"] if v else None
            rec[f"{name}_note"] = (v.get("note", "") if v else "") or ""
        rows_table.append(rec)
    cdf = pd.DataFrame(rows_table)

    # ---- 一致性统计 ----
    def cohens_kappa(a_vals, b_vals):
        """Binary Cohen's kappa for two arrays of bool/None."""
        pairs = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
        if not pairs:
            return None, 0
        n = len(pairs)
        po = sum(1 for a, b in pairs if a == b) / n
        p_a_pos = sum(1 for a, _ in pairs if a) / n
        p_b_pos = sum(1 for _, b in pairs if b) / n
        pe = p_a_pos * p_b_pos + (1 - p_a_pos) * (1 - p_b_pos)
        if pe >= 1:
            return 1.0, n
        kappa = (po - pe) / (1 - pe)
        return kappa, n

    # 3 方两两 kappa
    pairs = [("LLM_hint", existing[0])]
    if len(existing) >= 2:
        pairs.append(("LLM_hint", existing[1]))
        pairs.append((existing[0], existing[1]))

    st.subheader("📈 概览")
    overview_cols = st.columns(len(pairs) + 1)
    overview_cols[0].metric("论文数", len(cdf))
    for i, (a, b) in enumerate(pairs):
        ac = "LLM_hint_filter_out" if a == "LLM_hint" else f"{a}_filter_out"
        bc = "LLM_hint_filter_out" if b == "LLM_hint" else f"{b}_filter_out"
        kappa, n = cohens_kappa(cdf[ac].tolist(), cdf[bc].tolist())
        kappa_str = f"{kappa:+.2f}" if kappa is not None else "n/a"
        # 简单一致性
        agree = sum(1 for x, y in zip(cdf[ac], cdf[bc]) if x is not None and y is not None and x == y)
        a_label = "LLM" if a == "LLM_hint" else a
        b_label = "LLM" if b == "LLM_hint" else b
        overview_cols[i + 1].metric(
            f"{a_label} ↔ {b_label}",
            f"κ={kappa_str}",
            delta=f"{agree}/{n} 同意",
            delta_color="off",
        )

    # 各审阅者打标分布
    st.markdown("**各审阅者打标分布**")
    dist_cols = st.columns(len(existing) + 1)
    llm_filter_n = int(cdf["LLM_hint_filter_out"].sum())
    dist_cols[0].metric("LLM_hint filter_out", f"{llm_filter_n} / {len(cdf)}", f"{llm_filter_n/len(cdf)*100:.0f}%")
    for i, name in enumerate(existing):
        col = f"{name}_filter_out"
        n_fo = int(cdf[col].fillna(False).sum())
        n_total = int(cdf[col].notna().sum())
        dist_cols[i + 1].metric(f"{name} filter_out", f"{n_fo} / {n_total}", f"{n_fo/max(n_total,1)*100:.0f}%")

    st.divider()

    # ---- 三方一致性分类 ----
    def classify(row):
        l = row["LLM_hint_filter_out"]
        humans = [row.get(f"{n}_filter_out") for n in existing]
        humans = [h for h in humans if h is not None]
        if not humans:
            return "未审阅"
        all_humans_agree = len(set(humans)) == 1
        if all_humans_agree and humans[0] == l:
            return "✅ 全一致"
        if all_humans_agree and humans[0] != l:
            return "🔴 LLM 孤立"
        if not all_humans_agree:
            if l in humans:
                return "🟡 人际分歧 (LLM 与一人同)"
            else:
                return "🟣 三方分歧"
        return "其他"

    cdf["分类"] = cdf.apply(classify, axis=1)
    class_counts = cdf["分类"].value_counts()

    st.subheader("🗂️ 一致性分类")
    cat_cols = st.columns(len(class_counts) or 1)
    for i, (cat, n) in enumerate(class_counts.items()):
        cat_cols[i].metric(cat, n)

    # ---- Tabs:每类一个面板 ----
    categories = ["✅ 全一致", "🔴 LLM 孤立", "🟡 人际分歧 (LLM 与一人同)", "🟣 三方分歧", "未审阅"]
    available = [c for c in categories if c in class_counts]
    if available:
        tabs = st.tabs(available)
        for tab, cat in zip(tabs, available):
            with tab:
                sub = cdf[cdf["分类"] == cat].sort_values("paper_id")
                if len(sub) == 0:
                    st.caption("(空)")
                    continue

                # 紧凑表格
                show_cols = ["paper_id", "venue", "title", "LLM_integrity", "LLM_rigor",
                             "LLM_marketing", "LLM_hint_filter_out"] + \
                            [f"{n}_filter_out" for n in existing]
                st.dataframe(sub[show_cols], use_container_width=True, height=min(40 + 35 * len(sub), 400))

                # 逐篇可展开详情
                if cat != "✅ 全一致":  # 全一致就不需要细看
                    st.markdown("**逐篇详情**(点开看摘要、LLM 理由、双方 note)")
                    for _, row in sub.iterrows():
                        pid = row["paper_id"]
                        llm_row = paper_rows.get(pid, {})
                        p = llm_row.get("parsed") or {}
                        with st.expander(f"`{pid}` · {row['venue']} {row['year']} · {row['title'][:80]}"):
                            # 摘要
                            abstract = (abs_map.get(pid) or {}).get("abstract", "(摘要未在缓存找到)")
                            st.markdown("**摘要**")
                            st.write(abstract)

                            ccol1, ccol2, ccol3 = st.columns(3)
                            with ccol1:
                                st.markdown("**LLM 判定**")
                                st.write(f"hint filter_out: **{row['LLM_hint_filter_out']}**")
                                st.write(f"integrity: `{row['LLM_integrity']}`")
                                st.write(f"rigor: {row['LLM_rigor']}")
                                st.write(f"novelty: {row['LLM_novelty']}")
                                st.write(f"marketing: {row['LLM_marketing']}")
                                lc = p.get("logical_chain") or {}
                                if lc.get("conclusion"):
                                    st.caption(f"结论: {lc['conclusion']}")
                            with ccol2:
                                if existing:
                                    name = existing[0]
                                    v = row.get(f"{name}_filter_out")
                                    st.markdown(f"**{name} 判定**")
                                    st.write(f"filter_out: **{v}**")
                                    note = row.get(f"{name}_note") or ""
                                    if note.strip():
                                        st.caption(f"note: {note}")
                            with ccol3:
                                if len(existing) >= 2:
                                    name = existing[1]
                                    v = row.get(f"{name}_filter_out")
                                    st.markdown(f"**{name} 判定**")
                                    st.write(f"filter_out: **{v}**")
                                    note = row.get(f"{name}_note") or ""
                                    if note.strip():
                                        st.caption(f"note: {note}")

                            # LLM 9 维度 rationale 折叠
                            with st.expander("LLM 9 维度 rationale"):
                                for d in DIMS:
                                    v = p.get(d)
                                    if isinstance(v, dict):
                                        st.markdown(f"- **{DIM_CN[d]}** = `{v.get('score')}` — {v.get('rationale','')}")

    st.divider()

    # ---- 完整表导出 ----
    with st.expander("📋 完整对照表 (所有论文)"):
        st.dataframe(cdf, use_container_width=True, height=400)
    csv = cdf.to_csv(index=False).encode("utf-8")
    st.download_button("📥 导出对照表 (csv)", csv, "comparison.csv", "text/csv")
