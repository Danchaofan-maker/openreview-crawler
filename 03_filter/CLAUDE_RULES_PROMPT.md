# 任务：独立设计 rules_claude.json

## 研究背景

这个项目试图回答：在每年涌入顶会的数万篇论文里，真正依赖严格数学工具进行论证的有多少，他们在用什么数学。

研究悬置对应用场景的讨论，只盯推导链本身——"这篇论文的结论是被证出来的还是被跑出来的"。最终目标是建立一份关于当代理论基础的量化图谱，为理论研究方向提供校准依据。

为此需要从约 31k 篇已打分论文中筛出 **1,000–2,000 篇**值得精读的数学理论论文。你的任务是独立设计这个筛选规则集。

---

## 第一性原理：从认识论出发

在看任何数据之前，先想清楚这个问题的本质：

**什么是"值得精读的数学理论论文"？**

一篇论文的结论要么是被证出来的，要么是被跑出来的。我们要的是前者。但"被证出来"本身是一个光谱：
- 严格到什么程度才算？（假设合理性、证明完整性）
- 哪类数学工具代表真正的理论深度？（集合论+不等式只是入门，测度论、算子理论、代数拓扑才是区分度更高的信号）
- 理论新颖性和数学严格性是同一件事吗？

另一个问题：这份语料库是用来做什么的？
- 不是"最优论文排行榜"——那会导致 1500 篇风格完全相同的统计学习理论论文
- 是"对当前 ML 理论空间的覆盖采样"——需要多样性，需要捕捉不同类型的理论价值

从这两个出发点推导你的规则逻辑，而不是从数据分布反推。

---

## 深入研究后再动手

按顺序阅读，每一份都有独特信息：

1. `FIELDS.md` — 字段速查
2. `02_score/prompts/prompt_c3.md` — **最重要**：每个字段的精确定义、评分锚点、边界案例。特别注意 tea 字段的特殊性和 ig 分类的语义边界
3. `ourinsights/space_analysis_notes.md` — 数据空间分析：双峰结构、维度相关性、集合论筛选的适用边界
4. `ourinsights/market_structure_notes.md` — 市场结构：intact 供给端趋势，理解你在筛什么样的稀缺资源
5. `03_filter/space_report.html` — 分布报告：各维度分布图、ig 分组均值、相关矩阵
6. `data/sample_N600_seed42.json` — 600 篇样本，**强烈建议**：随机抽取 10–20 篇看真实论文数据，感受各字段在真实情况下是什么水平
7. `data/full/output.jsonl` — 全量数据，可用 Python 做任何分析

探索时要主动思考：
- 哪类论文是噪音，哪类是信号，边界在哪里
- 哪些好论文会被主信号（mr/tn/md 这个高度相关的簇）系统性低估
- ml 理论论文用到的数学工具有多广——不仅是集合论和不等式，还有什么
- 一个只靠单一维度的规则会犯什么错误
- 阈值切割（在欧氏空间划超矩形）是唯一可用的机制吗？维度间的**关系**（比如 mr 和 er 的联动、同类论文中的相对位置）能否成为更精确的信号？
- 多个弱信号同时指向同一结论，和单个强信号单独指向，哪种排除更可靠？

---

## 不要读以下文件（双盲设计）

- `03_filter/rules/rules_jes.json`
- `03_filter/rules/rules_danchaofan.json`
- `ourinsights/selection_methodology.md`
- `03_filter/mmr_corpus.jsonl` / `pca_corpus.jsonl` / `combined_corpus.jsonl`
- `03_filter/RULES_ENGINE_AUDIT.md`
- `03_filter/rules_codex.json`

---

## 规则格式

```json
{
  "inter_logic": "OR",
  "force_keep_hr": true,
  "keep_na": true,
  "rescue_rules": [...],
  "rules": [
    {
      "name": "规则名（说清楚排除的是什么类型的论文）",
      "enabled": true,
      "negate": false,
      "internal_logic": "AND",
      "conditions": [
        {"field": "mr", "op": "lte", "value": 3}
      ]
    }
  ]
}
```

- `inter_logic`：规则间逻辑，OR = 任一命中即排除
- `internal_logic`：条件间逻辑，AND = 全部满足才命中
- `keep_na`：字段缺失时视为不命中（建议 true；tea 字段约 60% 为 null，这是正常的）
- `rescue_rules`：格式同 rules，命中则强制保留，覆盖排除规则
- op：lt / lte / gt / gte / eq / neq / in
- 字段：mr / tn / md / ar / er / tea / cc / ei / sg / cs / integrity / marketing / human_review
  - integrity 取值：intact / partial / broken / absent
  - marketing / human_review 为布尔值

---

## 验证

```bash
uv run python -c "
import json, sys
sys.path.insert(0, '03_filter')
from mmr_select import apply_rules
cfg = json.loads(open('03_filter/rules/rules_claude.json').read())
papers = []
with open('data/full/output.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('ok') and r.get('parsed'):
            papers.append(r['parsed'])
kept = apply_rules(papers, cfg)
print(f'{len(papers)} → {len(kept)} 篇 ({len(kept)/len(papers):.1%})')
from collections import Counter
print(dict(Counter(p.get('ig') for p in kept)))
"
```

目标：**1,000–2,000 篇**，ig 分布合理（不应该全是 intact）。

---

## 输出

将规则写入 `03_filter/rules/rules_claude.json`，运行验证确认结果，并为每条规则写一句设计理由。
