# openreview-crawler

抓取多个机器学习会议/期刊的论文元数据,并基于摘要做数学严谨性粗筛,产出统一格式的候选集。

## 目录结构

```
openreview-crawler/
├── scripts/          可重复运行的爬虫与筛选脚本
├── explore/          一次性探索脚本(摸 API,不是正式代码)
├── archive/          已废弃旧版本,留档备查
├── data/
│   ├── raw/          原始元数据,只写不改
│   ├── filtered/     筛选后的候选集
│   └── excluded/     已排除的来源/年份(留作对照)
├── pyproject.toml    依赖清单(由 uv 管理)
└── uv.lock
```

## 环境

- Python ≥ 3.10
- 包管理: [uv](https://docs.astral.sh/uv/)
- 工作目录建议放在 Linux 原生路径(避免 `/mnt/c/...` 跨系统读写慢且权限混乱)

```bash
uv sync                       # 按 pyproject.toml + uv.lock 还原环境
uv run scripts/<某脚本>.py    # 在虚拟环境内运行
```

## 数据来源与脚本

每个脚本独立运行,输出固定路径,新增来源只需在此表格追加一行。

| 脚本 | 来源 | 范围 | 抓取方式 | 输出 |
|---|---|---|---|---|
| `scripts/openreview_crawler.py` | ICLR / ICML / NeurIPS | 2023–2026(见脚本 `VENUES`) | OpenReview API | `data/raw/<venue>_metadata.jsonl` |
| `scripts/tmlr_crawler.py` | TMLR | `pdate >= 2024-01-01` | OpenReview API | `data/raw/tmlr_metadata.jsonl` |
| `scripts/jmlr_crawler.py` | JMLR | v25(2024)、v26(2025) | jmlr.org HTML | `data/raw/jmlr_metadata.jsonl` |
| `scripts/colt_crawler.py` | COLT | v247(2024)、v291(2025) | PMLR HTML | `data/raw/colt_metadata.jsonl` |
| `scripts/journal_crawler.py` | JMLR / SIMODS / SIOPT | 2024-01-01 起 | OpenAlex API | `data/raw/{jmlr,simods,siopt}_metadata.jsonl` |
| `scripts/math_filter.py` | — | 读 `data/raw/*.jsonl` | 关键词打分 | `data/filtered/math_candidates.jsonl` + `math_filter_stats.json` |

> `journal_crawler.py` 支持参数选择期刊: `uv run scripts/journal_crawler.py simods siopt`(不传参则全跑)。

## 统一元数据字段

所有 `data/raw/*.jsonl` 写入同一套字段,便于后续合并/筛选:

| 字段 | 说明 |
|---|---|
| `paper_id` | 内部 ID(OpenReview note id / `pmlr-vNNN-xxx` / `jmlr-vNN-xxx` / OpenAlex work id) |
| `arxiv_id` | arXiv ID,无则 `null` |
| `venue` | 来源标识(`ICLR_2024` / `TMLR` / `COLT_2025` / `JMLR` / ...) |
| `decision_track` | OpenReview 会议: `Oral` / `Spotlight` / `Poster` / `null`;其他来源固定 `null` |
| `title`, `abstract`, `authors` | 原始字段,作者用 `; ` 分隔 |
| `year` | 发表/接收年份(部分来源可缺) |
| `doi` | 仅 OpenAlex 来源稳定提供 |
| `crawled_at` | 抓取时间(ISO 8601) |

新增来源时,请保持字段一致;新增字段请同步本表。

## 数学严谨性筛选

`scripts/math_filter.py` 读取 `data/raw/` 下所有 `*.jsonl`,按摘要+标题文本做关键词打分:

- 强信号(+3): `theorem` / `lemma` / `proof` / `convergence rate` / `regret bound` / `np-hard` 等
- 中等信号(+1): `lower bound` / `minimax` / `convex` / `hilbert` / `measure theory` 等
- 弱信号(≥2 个才 +1): `algorithm` / `optimization` / `estimator` 等

输出:
- `data/filtered/math_candidates.jsonl` — `score >= 3` 的候选,附 `math_score` 与命中模式 `math_hits`
- `data/filtered/math_filter_stats.json` — 各来源的总数与不同阈值通过数

阈值与模式集中在脚本顶部 `STRONG_PATTERNS` / `MEDIUM_PATTERNS` / `WEAK_PATTERNS`,需要调整直接改这三个列表。

## 典型流程

```bash
# 1. 拉取各来源(可分别运行,互不影响)
uv run scripts/openreview_crawler.py
uv run scripts/tmlr_crawler.py
uv run scripts/jmlr_crawler.py
uv run scripts/colt_crawler.py
uv run scripts/journal_crawler.py

# 2. 合并筛选
uv run scripts/math_filter.py
```

## 维护约定

- **`data/raw/` 只增不改**:重新抓取直接覆盖整个 `*.jsonl` 文件,不在原文件上手工编辑。
- **排除某来源**:把对应 `*.jsonl` 移到 `data/excluded/`(例:`colt_pre2024_metadata.jsonl`),`math_filter.py` 自然不会读到。
- **废弃旧脚本**:移到 `archive/`,不直接删除,便于追溯当时方案。
- **新增爬虫的最小步骤**:
  1. 在 `scripts/` 下新建脚本,输出 `data/raw/<source>_metadata.jsonl`,字段对齐上表。
  2. 在本 README 的"数据来源与脚本"表格追加一行。
  3. 不需要改 `math_filter.py`(它自动遍历 `data/raw/*.jsonl`)。
- **依赖变更**:用 `uv add <pkg>` / `uv remove <pkg>`,不要手改 `pyproject.toml` 后忘记更新 `uv.lock`。
