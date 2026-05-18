from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.mixture import GaussianMixture

from diptest import diptest

try:
    from explore.field_labels import FIELD_ZH
except ImportError:
    FIELD_ZH = {}


DATA_PATH = Path("data/full/output.jsonl")
HTML_PATH = Path("explore/distribution_analysis.html")
JSON_PATH = Path("explore/distribution_conclusions.json")

SCORE_FIELDS = ["mr", "tn", "md", "ar", "er", "cc", "ei", "sg", "cs"]
INTEGRITY_ORDER = ["intact", "partial", "absent"]
FIELD_NAMES = {
    "mr": "数学严谨度",
    "tn": "理论新颖度",
    "md": "数学深度",
    "ar": "假设现实度",
    "er": "经验验证依赖度",
    "cc": "算力门槛",
    "ei": "认识论意图",
    "sg": "适用范围广度",
    "cs": "置信度",
}
FIELD_NAMES.update({k: FIELD_ZH.get(k, v) for k, v in FIELD_NAMES.items()})


@dataclass
class GmmResult:
    means: list[float]
    stds: list[float]
    weights: list[float]
    valley: float | None
    separation_sigma: float | None


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0 or value > 10:
        return None
    return value


def load_dataframe(path: Path = DATA_PATH) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error") or not record.get("ok"):
                continue
            parsed = record.get("parsed")
            if not isinstance(parsed, dict):
                continue
            row: dict[str, Any] = {
                "paper_id": record.get("paper_id") or parsed.get("id"),
                "title": record.get("title", ""),
                "venue": record.get("venue"),
                "year": record.get("year"),
                "ig": parsed.get("ig"),
            }
            for field in SCORE_FIELDS:
                row[field] = finite_float(parsed.get(field))
            rows.append(row)
    return pd.DataFrame(rows)


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = max(std, 1e-6)
    z = (x - mean) / std
    return np.exp(-0.5 * z * z) / (std * math.sqrt(2 * math.pi))


def fit_gmm(values: np.ndarray) -> GmmResult:
    x = values.reshape(-1, 1)
    model = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    model.fit(x)

    means = model.means_.ravel()
    stds = np.sqrt(model.covariances_.reshape(2))
    weights = model.weights_.ravel()
    order = np.argsort(means)
    means = means[order]
    stds = stds[order]
    weights = weights[order]

    pooled_std = float(np.sqrt(np.mean(stds**2)))
    separation_sigma = None if pooled_std <= 0 else float((means[1] - means[0]) / pooled_std)
    valley = find_gmm_valley(means, stds, weights)

    return GmmResult(
        means=[round(float(v), 4) for v in means],
        stds=[round(float(v), 4) for v in stds],
        weights=[round(float(v), 4) for v in weights],
        valley=None if valley is None else round(float(valley), 4),
        separation_sigma=None if separation_sigma is None else round(separation_sigma, 4),
    )


def find_gmm_valley(means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> float | None:
    lo, hi = float(means[0]), float(means[1])
    if not lo < hi:
        return None
    grid = np.linspace(lo, hi, 3001)
    density = (
        weights[0] * normal_pdf(grid, means[0], stds[0])
        + weights[1] * normal_pdf(grid, means[1], stds[1])
    )
    derivative = np.diff(density)
    local_min_idx = np.where((derivative[:-1] < 0) & (derivative[1:] > 0))[0] + 1
    if len(local_min_idx):
        idx = int(local_min_idx[np.argmin(density[local_min_idx])])
    else:
        idx = int(np.argmin(density))
    return float(grid[idx])


def describe_shape(
    field: str,
    std: float,
    skewness: float,
    dip_p: float,
    gmm: GmmResult,
) -> tuple[str, float | None, str, str]:
    separation = gmm.separation_sigma or 0.0
    is_bimodal = dip_p < 0.05 and separation > 1.5
    if is_bimodal and gmm.valley is not None:
        return (
            "bimodal",
            gmm.valley,
            "gmm_valley",
            "dip test rejects unimodality and GMM components are well separated",
        )

    if field == "cs" or std < 1.15:
        return (
            "compressed_normal",
            None,
            "not_recommended",
            f"std={std:.2f}, insufficient discriminative power",
        )

    if skewness > 0.75:
        return (
            "unimodal_right_skew",
            None,
            "p75_percentile",
            "right-skewed or long-tailed; percentile threshold is more stable than an absolute cut",
        )
    if skewness < -0.75:
        return (
            "unimodal_left_skew",
            None,
            "p25_percentile",
            "left-skewed; percentile threshold is more stable than an absolute cut",
        )
    return (
        "unimodal",
        None,
        "p75_percentile",
        "no confirmed bimodality; use a percentile cut if the dimension is needed",
    )


def analyze_field(df: pd.DataFrame, field: str) -> dict[str, Any]:
    values = df[field].dropna().astype(float).to_numpy()
    if len(values) < 10:
        raise ValueError(f"{field}: insufficient valid values")

    dip, dip_p = diptest(values)
    gmm = fit_gmm(values)
    percentiles = {
        f"p{p}": round(float(np.percentile(values, p)), 4)
        for p in [10, 25, 50, 75, 85, 90]
    }
    std = float(np.std(values, ddof=1))
    skewness = float(stats.skew(values, bias=False))
    kurtosis = float(stats.kurtosis(values, fisher=True, bias=False))
    shape, threshold, method, reason = describe_shape(field, std, skewness, float(dip_p), gmm)
    if threshold is None and method.endswith("_percentile"):
        percentile_key = method.removesuffix("_percentile")
        threshold = percentiles.get(percentile_key)

    return {
        "shape": shape,
        "count": int(len(values)),
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
        "std": round(std, 4),
        "skewness": round(skewness, 4),
        "excess_kurtosis": round(kurtosis, 4),
        "percentiles": percentiles,
        "dip_test_d": round(float(dip), 6),
        "dip_test_p": round(float(dip_p), 6),
        "gmm_components": [
            {"mean": gmm.means[i], "std": gmm.stds[i], "weight": gmm.weights[i]}
            for i in range(2)
        ],
        "gmm_valley": gmm.valley,
        "gmm_separation_sigma": gmm.separation_sigma,
        "recommended_threshold": None if threshold is None else round(float(threshold), 4),
        "threshold_method": method,
        "reason": reason,
    }


def mixture_density(field_result: dict[str, Any], x: np.ndarray) -> np.ndarray:
    density = np.zeros_like(x, dtype=float)
    for component in field_result["gmm_components"]:
        density += component["weight"] * normal_pdf(x, component["mean"], component["std"])
    return density


def make_distribution_figure(df: pd.DataFrame, results: dict[str, dict[str, Any]]) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=[
            f"{field} · {FIELD_NAMES[field]}<br>{results[field]['shape']}, p={results[field]['dip_test_p']:.3g}"
            for field in SCORE_FIELDS
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.13,
    )
    xgrid = np.linspace(0, 10, 501)
    for idx, field in enumerate(SCORE_FIELDS):
        row = idx // 3 + 1
        col = idx % 3 + 1
        values = df[field].dropna().astype(float)
        fig.add_trace(
            go.Histogram(
                x=values,
                histnorm="probability density",
                xbins={"start": 0, "end": 10, "size": 0.5},
                marker_color="#4C78A8",
                opacity=0.68,
                showlegend=False,
                hovertemplate=f"{field}: %{{x}}<br>density=%{{y:.3f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=xgrid,
                y=mixture_density(results[field], xgrid),
                mode="lines",
                line={"color": "#F58518", "width": 2},
                showlegend=False,
                hovertemplate="GMM density=%{y:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        threshold = results[field]["recommended_threshold"]
        if threshold is not None:
            fig.add_vline(
                x=threshold,
                line_width=1.5,
                line_dash="dash",
                line_color="#E45756",
                row=row,
                col=col,
            )
        fig.update_xaxes(range=[0, 10], dtick=2, row=row, col=col)
    fig.update_layout(
        title="Score Distributions with Two-Component GMM Fits",
        height=980,
        bargap=0.04,
        template="plotly_white",
    )
    return fig


def make_correlation_figures(df: pd.DataFrame) -> tuple[go.Figure, go.Figure, dict[str, Any]]:
    score_df = df[SCORE_FIELDS].astype(float)
    pearson = score_df.corr(method="pearson")
    spearman = score_df.corr(method="spearman")
    labels = [f"{f}<br>{FIELD_NAMES[f]}" for f in SCORE_FIELDS]

    pearson_fig = px.imshow(
        pearson,
        x=labels,
        y=labels,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        text_auto=".2f",
        title="Pearson Correlation Matrix",
    )
    pearson_fig.update_layout(template="plotly_white", height=650)

    spearman_fig = px.imshow(
        spearman,
        x=labels,
        y=labels,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        text_auto=".2f",
        title="Spearman Correlation Matrix",
    )
    spearman_fig.update_layout(template="plotly_white", height=650)

    high_pairs = []
    for i, left in enumerate(SCORE_FIELDS):
        for right in SCORE_FIELDS[i + 1 :]:
            pr = float(pearson.loc[left, right])
            sr = float(spearman.loc[left, right])
            if abs(pr) > 0.6 or abs(sr) > 0.6:
                high_pairs.append(
                    {
                        "fields": [left, right],
                        "pearson": round(pr, 4),
                        "spearman": round(sr, 4),
                    }
                )
    corr_json = {
        "pearson": pearson.round(4).to_dict(),
        "spearman": spearman.round(4).to_dict(),
        "highly_correlated_pairs": high_pairs,
    }
    return pearson_fig, spearman_fig, corr_json


def make_integrity_figure(df: pd.DataFrame) -> go.Figure:
    plot_df = df[df["ig"].isin(INTEGRITY_ORDER)].copy()
    long_df = plot_df.melt(
        id_vars=["ig"],
        value_vars=SCORE_FIELDS,
        var_name="dimension",
        value_name="score",
    ).dropna()
    long_df["dimension_label"] = long_df["dimension"].map(
        lambda f: f"{f} · {FIELD_NAMES[f]}"
    )
    fig = px.violin(
        long_df,
        x="dimension_label",
        y="score",
        color="ig",
        category_orders={"ig": INTEGRITY_ORDER},
        box=True,
        points=False,
        color_discrete_map={
            "intact": "#54A24B",
            "partial": "#F58518",
            "absent": "#E45756",
        },
        title="Score Distribution by Integrity Group",
    )
    fig.update_layout(template="plotly_white", height=720, xaxis_tickangle=-35)
    fig.update_yaxes(range=[0, 10])
    return fig


def integrity_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for group in INTEGRITY_ORDER:
        group_df = df[df["ig"] == group]
        summary[group] = {"count": int(len(group_df))}
        for field in SCORE_FIELDS:
            values = group_df[field].dropna().astype(float)
            summary[group][field] = {
                "count": int(len(values)),
                "mean": None if values.empty else round(float(values.mean()), 4),
                "median": None if values.empty else round(float(values.median()), 4),
            }
    return summary


def strategy_table(results: dict[str, dict[str, Any]]) -> str:
    rows = []
    for field in SCORE_FIELDS:
        result = results[field]
        threshold = result["recommended_threshold"]
        threshold_text = "not recommended" if threshold is None else f"{threshold:.3g}"
        rows.append(
            "<tr>"
            f"<td><strong>{field}</strong></td>"
            f"<td>{FIELD_NAMES[field]}</td>"
            f"<td>{result['shape']}</td>"
            f"<td>{result['dip_test_p']:.3g}</td>"
            f"<td>{result['gmm_separation_sigma']:.3g}</td>"
            f"<td>{threshold_text}</td>"
            f"<td>{result['threshold_method']}</td>"
            f"<td>{result['reason']}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Field</th><th>Name</th><th>Shape</th><th>Dip p</th>"
        "<th>GMM separation</th><th>Threshold</th><th>Method</th><th>Reason</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_html(
    df: pd.DataFrame,
    results: dict[str, dict[str, Any]],
    corr: dict[str, Any],
    figures: list[go.Figure],
) -> str:
    group_counts = df["ig"].value_counts(dropna=False).to_dict()
    confirmed = [field for field, result in results.items() if result["shape"] == "bimodal"]
    not_recommended = [
        field for field, result in results.items() if result["threshold_method"] == "not_recommended"
    ]
    fig_html = "\n".join(
        pio.to_html(fig, include_plotlyjs=True if i == 0 else False, full_html=False)
        for i, fig in enumerate(figures)
    )
    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #222; }
      h1, h2 { margin-top: 32px; }
      .summary { display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; }
      .metric { border: 1px solid #ddd; border-radius: 6px; padding: 12px 14px; background: #fafafa; }
      .metric .value { font-size: 24px; font-weight: 650; margin-top: 4px; }
      table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 14px; }
      th, td { border: 1px solid #ddd; padding: 8px 10px; vertical-align: top; }
      th { background: #f4f4f4; text-align: left; }
      code { background: #f3f3f3; padding: 2px 4px; border-radius: 3px; }
    </style>
    """
    high_pairs = corr["highly_correlated_pairs"]
    high_pair_items = "".join(
        f"<li><code>{a}</code> / <code>{b}</code>: Pearson={p['pearson']}, Spearman={p['spearman']}</li>"
        for p in high_pairs
        for a, b in [p["fields"]]
    )
    if not high_pair_items:
        high_pair_items = "<li>No pair exceeded |r| &gt; 0.6.</li>"

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Distribution Analysis</title>
  {css}
</head>
<body>
  <h1>Distribution Analysis for LLM Paper Scores</h1>
  <p>Input: <code>{DATA_PATH}</code>. Valid records: {len(df):,}. Generated by <code>explore/distribution_analysis.py</code>.</p>
  <div class="summary">
    <div class="metric"><div>Confirmed bimodal dimensions</div><div class="value">{", ".join(confirmed) or "none"}</div></div>
    <div class="metric"><div>Not recommended as filter axis</div><div class="value">{", ".join(not_recommended) or "none"}</div></div>
    <div class="metric"><div>High-correlation pairs</div><div class="value">{len(high_pairs)}</div></div>
    <div class="metric"><div>Integrity counts</div><div class="value">{json.dumps(group_counts, ensure_ascii=False)}</div></div>
  </div>

  <h2>Recommended Filtering Strategy</h2>
  {strategy_table(results)}

  <h2>Correlation Redundancy</h2>
  <ul>{high_pair_items}</ul>

  {fig_html}
</body>
</html>
"""


def write_outputs() -> None:
    df = load_dataframe()
    if df.empty:
        raise RuntimeError(f"No valid rows loaded from {DATA_PATH}")

    results = {field: analyze_field(df, field) for field in SCORE_FIELDS}
    pearson_fig, spearman_fig, corr = make_correlation_figures(df)
    integrity_fig = make_integrity_figure(df)
    distribution_fig = make_distribution_figure(df, results)

    conclusions: dict[str, Any] = {
        field: results[field]
        for field in SCORE_FIELDS
    }
    conclusions["_meta"] = {
        "input_path": str(DATA_PATH),
        "valid_records": int(len(df)),
        "score_fields": SCORE_FIELDS,
        "integrity_counts": {
            str(k): int(v) for k, v in df["ig"].value_counts(dropna=False).to_dict().items()
        },
    }
    conclusions["_correlations"] = corr
    conclusions["_integrity_summary"] = integrity_summary(df)

    JSON_PATH.write_text(
        json.dumps(conclusions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    HTML_PATH.write_text(
        render_html(df, results, corr, [distribution_fig, pearson_fig, spearman_fig, integrity_fig]),
        encoding="utf-8",
    )
    print(f"Loaded {len(df):,} valid records")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {HTML_PATH}")


if __name__ == "__main__":
    write_outputs()
