# ml-theory-lens

**研究课题**：当代人工智能顶尖文献中严密数学推导占比的量化评估与基础数学工具聚类分析。

---

## 动机

这个项目试图回答一个简单但在噪声里很难看清的问题：在每年涌入顶会的数万篇论文里，真正依赖严格数学工具进行论证的有多少，他们在用什么数学。

我们悬置对具体应用场景和下游任务的讨论，只盯着推导链本身——不是"这篇论文解决了什么问题"，而是"这篇论文的结论是被证出来的还是被跑出来的"。最终目标是一份关于当代理论基础的量化图谱，能客观说明学界在探讨底层逻辑时最常依赖的数学分支，为我们自己的理论研究方向提供校准依据。

---

## 流水线

```
01_collect/   →   02_score/   →   03_filter/   →   04_read/
  爬取元数据       LLM 多维打分      规则筛选           浏览与标注
```

### 01_collect — 数据入口

自动化爬取 ICLR / ICML / NeurIPS / TMLR / JMLR / COLT / SIOPT / SIMODS 的论文元数据，输出到 `data/raw/`。

| 脚本 | 来源 | 输出 |
|------|------|------|
| `openreview_crawler.py` | ICLR / ICML / NeurIPS | `data/raw/<venue>_metadata.jsonl` |
| `tmlr_crawler.py` | TMLR | `data/raw/tmlr_metadata.jsonl` |
| `jmlr_crawler.py` | JMLR | `data/raw/jmlr_metadata.jsonl` |
| `colt_crawler.py` | COLT | `data/raw/colt_metadata.jsonl` |
| `journal_crawler.py` | JMLR / SIMODS / SIOPT | `data/raw/{jmlr,simods,siopt}_metadata.jsonl` |

统一元数据字段：`paper_id` / `arxiv_id` / `venue` / `title` / `abstract` / `authors` / `year` / `doi` / `crawled_at`

### 02_score — LLM 多维打分

用 DeepSeek V4-Pro（prompt v0.8）对摘要做 9 个维度的打分，同时生成逻辑链分析。当前已完成全量 ~31k 篇的评分（`data/full/output.jsonl`，108MB）。

**评分维度**（0–10）：

| 字段 | 含义 |
|------|------|
| `mr` | 数学严谨度 |
| `tn` | 理论新颖度 |
| `md` | 数学深度 |
| `ar` | 假设现实度 |
| `er` | 经验验证依赖度 |
| `tea` | 理论实验咬合度 |
| `cc` | 算力门槛 |
| `ei` | 认识论意图 |
| `sg` | 适用范围广度 |
| `cs` | 置信度（模型自评） |

**逻辑链完整性**（`ig`）：`intact` / `partial` / `broken` / `absent`

关键发现：intact 论文占比从 2024 年的 12.9% 单调下降至 2026 年的 7.3%，绝对产量接近饱和（~1,100 篇/年）——供给端受限于人，不受工具加速影响。

### 03_filter — 规则筛选

从打分结果中按规则集筛出精读语料库（目标 1,000–2,000 篇）。每个人维护自己的规则文件，最终取交集。

- `03_filter/rules/rules_claude.json` — Claude 的规则集，当前产出 2,354 篇
- `03_filter/rules/rules_jes.json` — jes 的规则集（待写）
- `03_filter/rules/rules_danchaofan.json` — danchaofan 的规则集（待写）

规则格式见任意现有 JSON 文件，可直接在 `04_read/` 前端的 preset 下拉里加载预览效果。

**待完成**：基于 `03_filter/distribution_conclusions.json` 的分布分析结论，优化各规则集阈值；对精读语料库做数学工具聚类（第三阶段）。

### 04_read — 浏览与标注

本地 HTTP 服务器 + 单页前端，支持规则预览、收藏、双盲打分。

```bash
uv run 04_read/serve.py
# 浏览器打开 http://localhost:8080
```

---

## 快速开始

```bash
# 环境
uv sync

# 从头跑（已有 data/raw/ 可跳过第一步）
uv run 01_collect/openreview_crawler.py
uv run 02_score/score_paper_c3.py

# 直接浏览已有全量结果
uv run 04_read/serve.py
```

---

## 数据约定

- `data/raw/` 只增不改，重新抓取直接覆盖整个文件
- `data/full/output.jsonl` 是全量打分结果，不手工编辑
- 废弃脚本移到对应子目录的 `deprecated/`，不直接删除

---

## ourinsights/

横跨流水线的观察和判断放这里，不属于任何单一阶段。当前：

- `market_structure_notes.md` — 投稿量 delta/gamma 分析，intact 供给端饱和假说，时间窗口判断
