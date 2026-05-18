#!/usr/bin/env python3
"""
UMAP 降维可视化 — 把 ~19k 篇论文的 9 维打分压到 2D
观察 intact/partial/broken 是否在特征空间里自然分离

输出: 03_filter/space_umap.html  (交互式 plotly)
用法: uv run 03_filter/space_umap.py [--sample N]
"""

import json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap

PROJECT   = Path(__file__).parent.parent
DATA_FILE = PROJECT / "data/full/output.jsonl"
OUT_HTML  = PROJECT / "03_filter/space_umap.html"

# 维度顺序和可靠度权重（来自 repeatability 实验）
FIELDS = ["mr", "tn", "md", "ar", "er", "tea", "cc", "ei", "sg", "cs"]
WEIGHTS = {
    "mr":  1.00,  # std=0.515 最稳定
    "cs":  1.00,  # std=0.623
    "md":  0.90,  # std=0.701
    "cc":  0.90,  # std=0.708
    "er":  0.85,  # std=0.813
    "tn":  0.85,  # std=0.803
    "sg":  0.70,  # std=0.985
    "ei":  0.60,  # std=1.254
    "tea": 0.55,  # std=1.276
    "ar":  0.55,  # std=1.269
}

IG_COLORS = {
    "intact":  "#2ecc71",
    "partial": "#3498db",
    "broken":  "#e74c3c",
    "absent":  "#95a5a6",
}


def load_data(sample_n: int | None) -> pd.DataFrame:
    rows = []
    with open(DATA_FILE) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("ok") or not r.get("parsed"):
                continue
            p = r["parsed"]
            ig = p.get("ig")
            if ig not in ("intact", "partial", "broken", "absent"):
                continue
            scores = {}
            for f in FIELDS:
                v = p.get(f)
                if isinstance(v, (int, float)) and 0 <= v <= 10:
                    scores[f] = float(v)
            min_dims = 4 if ig == "absent" else 7
            if len(scores) < min_dims:
                continue
            row = {
                "paper_id": r["paper_id"],
                "title":    r.get("title", "")[:80],
                "venue":    r.get("venue", ""),
                "year":     r.get("year"),
                "ig":       ig,
                "mk_f":     p.get("mk_f", False),
                "hr_f":     p.get("hr_f", False),
                **scores,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} papers (intact={sum(df.ig=='intact')} "
          f"partial={sum(df.ig=='partial')} broken={sum(df.ig=='broken')} "
          f"absent={sum(df.ig=='absent')})")

    if sample_n and sample_n < len(df):
        pieces = []
        for ig_val in df["ig"].unique():
            g = df[df["ig"] == ig_val]
            n = max(1, int(sample_n * len(g) / len(df)))
            pieces.append(g.sample(min(len(g), n), random_state=42))
        df = pd.concat(pieces).reset_index(drop=True)
        print(f"Sampled {len(df)} papers")

    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    mat = []
    for f in FIELDS:
        if f in df.columns:
            col = df[f].fillna(df[f].median()).values
            mat.append(col * WEIGHTS[f])
    return np.column_stack(mat)


def run_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=42,
        low_memory=True,
        verbose=True,
    )
    return reducer.fit_transform(X)


def make_plot(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("按逻辑链完整性 (ig)", "按营销包装 (mk_f)"),
        horizontal_spacing=0.08,
    )

    # Panel 1: color by ig
    for ig, color in IG_COLORS.items():
        mask = df["ig"] == ig
        sub = df[mask]
        fig.add_trace(go.Scatter(
            x=sub["umap_x"], y=sub["umap_y"],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.6),
            name=ig,
            text=sub.apply(lambda r:
                f"<b>{r['title']}</b><br>"
                f"venue={r['venue']} year={r['year']}<br>"
                f"ig={r['ig']} mk={r['mk_f']}<br>"
                f"mr={r.get('mr','?')} tn={r.get('tn','?')} md={r.get('md','?')}<br>"
                f"ar={r.get('ar','?')} er={r.get('er','?')} cs={r.get('cs','?')}",
                axis=1),
            hoverinfo="text",
            legendgroup=ig,
        ), row=1, col=1)

    # Panel 2: color by marketing
    for mk, color, name in [(False, "#3498db", "无营销"), (True, "#e74c3c", "营销包装")]:
        mask = df["mk_f"] == mk
        sub = df[mask]
        fig.add_trace(go.Scatter(
            x=sub["umap_x"], y=sub["umap_y"],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.5),
            name=name,
            text=sub.apply(lambda r:
                f"<b>{r['title']}</b><br>"
                f"venue={r['venue']}  ig={r['ig']}<br>"
                f"mr={r.get('mr','?')} tn={r.get('tn','?')}",
                axis=1),
            hoverinfo="text",
            legendgroup=name,
            showlegend=True,
        ), row=1, col=2)

    fig.update_layout(
        title="论文特征空间 UMAP 投影 (9维打分 → 2D)<br>"
              f"<sup>n={len(df)} | 维度加权 | metric=euclidean | n_neighbors=30</sup>",
        height=700,
        template="plotly_dark",
        legend=dict(itemsizing="constant"),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=8000,
                        help="stratified sample size (default 8000, 0=all)")
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    args = parser.parse_args()

    sample_n = args.sample if args.sample > 0 else None

    print("Loading data...")
    df = load_data(sample_n)

    print("Building feature matrix...")
    X = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape}")

    print("Running UMAP...")
    coords = run_umap(X, n_neighbors=args.neighbors, min_dist=args.min_dist)
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]

    print("Generating plot...")
    fig = make_plot(df)
    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print(f"Saved: {OUT_HTML}")

    # quick summary: are intact papers clustered?
    from scipy.spatial.distance import cdist
    intact_coords  = coords[df["ig"] == "intact"]
    partial_coords = coords[df["ig"] == "partial"]
    if len(intact_coords) > 10 and len(partial_coords) > 10:
        intact_centroid  = intact_coords.mean(axis=0)
        partial_centroid = partial_coords.mean(axis=0)
        centroid_dist = np.linalg.norm(intact_centroid - partial_centroid)
        intact_spread  = np.std(cdist(intact_coords,  [intact_centroid]))
        partial_spread = np.std(cdist(partial_coords, [partial_centroid]))
        print(f"\nIntact  centroid: {intact_centroid.round(2)}  spread={intact_spread:.2f}")
        print(f"Partial centroid: {partial_centroid.round(2)}  spread={partial_spread:.2f}")
        print(f"Centroid distance: {centroid_dist:.2f}")
        print(f"Separation ratio: {centroid_dist / (intact_spread + partial_spread):.2f}  "
              f"(>1 = 有分离, <0.5 = 严重混叠)")


if __name__ == "__main__":
    main()
