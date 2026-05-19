# 泛智能顶尖文献纯粹数学骨架观测与基底坐标重建高维映射域

**研究课题**：当代人工智能顶尖文献中严密数学推导链的系统性量化内容分析与基础数学工具分布观测。

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

从打分结果中按规则集筛出精读语料库（目标 1,000–2,000 篇）。五套独立筛选器，结构上类似 MAGI——保留分歧信息，不强行合并为单一共识。

**JSON 规则集**（各自独立，双盲设计）：

| 规则集 | 产出 | 设计思路 |
|--------|------|----------|
| `rules_jes.json` | ~6,200 篇（宽松预筛，供 MMR 输入） | 保留规则 + OR，rescue outlier |
| `rules_claude.json` | ~1,870 篇 | 排除规则 + OR，ig 分层 + mr 硬切 |
| `rules_codex.json` | ~1,680 篇 | 多条件复合排除 + 7 类 rescue |
| `rules_danchaofan.json` | ~2,030 篇 | 多维度联合排除 |

**Python 筛选器**（超出超矩形的三个机制）：

- `python_filter.py` — 组内百分位排名 + 多信号投票（mr/tn/md 合为一票）+ 维度一致性检测，产出 ~2,030 篇

**语料库分层**：

| 层次 | 论文数 | 含义 |
|------|--------|------|
| 5-way 交集（JSON×4 + python） | 851 篇 | 高置信核心，所有方法一致同意 |
| 4-way JSON 交集 | 961 篇 | JSON 规则全票通过 |
| `combined_corpus.jsonl` | 1,500 篇 | MMR 最终精读语料库（FLD×0.7 + PCA×0.3，λ=0.5） |

```bash
# 运行任意筛选器
uv run python -c "
import json, sys; sys.path.insert(0, '03_filter')
from mmr_select import apply_rules
papers = [json.loads(l)['parsed'] for l in open('data/full/output.jsonl') if json.loads(l).get('ok')]
cfg = json.loads(open('03_filter/rules/rules_claude.json').read())
kept = apply_rules(papers, cfg)
print(len(kept))
"

# 重新生成 MMR 语料库
uv run 03_filter/combined_score.py --n 1500 --lam 0.5
```

设计哲学见 `ourinsights/filter_design_philosophy.md`。

### 04_read — 浏览与标注

本地 HTTP 服务器 + 单页前端，支持规则预览、收藏、双盲打分。

```bash
uv run 03_filter/serve.py
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

横跨流水线的观察和判断放这里，不属于任何单一阶段。

- `market_structure_notes.md` — 投稿量 delta/gamma 分析，intact 供给端饱和假说，时间窗口判断
- `space_analysis_notes.md` — 特征空间分析：双峰结构、维度相关性、集合论筛选的适用边界
- `selection_methodology.md` — 筛选方法论：FLD/PCA/MMR 设计、综合评分推导、Monte Carlo ensemble
- `filter_design_philosophy.md` — 筛选设计哲学：三个超出超矩形机制的认识论依据、MAGI 结构的意义
