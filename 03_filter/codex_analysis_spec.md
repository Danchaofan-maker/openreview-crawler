# 分布分析任务 spec
*给 Codex 的独立分析任务*

---

## 背景

我们对 ~31k 篇 ML 顶会/期刊论文做了 LLM 多维打分（prompt v0.8），输出存储在
`data/full/output.jsonl`。每条记录的 `parsed` 字段包含 9 个 0-10 的数值维度。

**核心假说（需验证）**：

- **假说 1**：工程/benchmark 论文和理论论文在某些维度上形成可分的双峰，切分点
  即为筛选阈值
- **假说 2**：不同维度的分布形态不同（双峰 / 长尾 / 正态偏置），需要针对性的
  统计方法来确定切分点；错误地把单峰当双峰处理会产生错误的筛选规则

---

## 已知的分布形态（快速目测）

| 维度 | 中文含义 | 初步形态 | 备注 |
|------|---------|---------|------|
| `mr` | 数学严谨度 | **疑似双峰** | peak@1-2, valley@3-4, secondary peak@7 |
| `tn` | 理论新颖度 | 右偏单峰 | peak@3-4，平缓右尾 |
| `md` | 数学深度 | 强左偏长尾 | peak@1，指数衰减 |
| `ar` | 假设现实度 | 左偏单峰 | peak@5，有明显低分簇 |
| `er` | 经验验证依赖度 | 左偏单峰 | peak@7-8，ceiling 效应（13.4%≥9） |
| `cc` | 算力门槛 | 强单峰正态 | peak@3，std=1.4，分辨率低 |
| `ei` | 认识论意图 | **疑似双峰** | peak@1, valley@4-5, secondary peak@7 |
| `sg` | 适用范围广度 | 右偏单峰 | peak@2，平缓右尾 |
| `cs` | 置信度 | 高度压缩正态 | peak@7，std=1.03，几乎无分辨率 |

---

## 分析目标

### Goal 1：确认/否定双峰假说

对每个维度：

1. 用 **Hartigan's dip test** 检验单峰 vs 多峰（`diptest` 库）
2. 用 **Gaussian Mixture Model（GMM，n=2）** 拟合，输出两个成分的均值/方差/权重
3. 判据：dip test p < 0.05 且 GMM 两峰间距 > 1.5σ → 确认双峰
4. 对确认双峰的维度，找 GMM 的**谷底（valley）坐标**作为候选切分点

### Goal 2：识别长尾维度，提出正确的阈值方法

对单峰/长尾维度：

1. 计算 **skewness** 和 **excess kurtosis**
2. 对右偏长尾维度（`md`、`ei` 低分区），推荐用**百分位数阈值**（如 p75/p85）
   而非绝对值阈值
3. 对高度压缩的维度（`cs`，std=1.03），明确说明该维度**不适合作为筛选轴**

### Goal 3：维度间相关性矩阵

1. 计算所有维度对的 Pearson 和 Spearman 相关系数
2. 识别高度相关的维度对（|r| > 0.6），这些维度包含冗余信息，
   筛选规则里只需保留一个
3. 输出热力图

### Goal 4：按 integrity 分组的分布对比

对 `ig ∈ {intact, partial, absent}` 三组，分别绘制所有维度的分布：
- intact 组的分布形态是否与整体不同？
- 这能否反向验证哪些维度真正区分了"有推导链的论文"

---

## 输出要求

1. **一个 HTML 报告**（保存到 `explore/distribution_analysis.html`），包含：
   - 每个维度的分布直方图 + GMM 拟合曲线 + dip test 结果
   - 维度间相关矩阵热力图
   - integrity 分组对比图
   - 每个维度的**推荐筛选策略**（双峰切点 / 百分位数 / 不建议使用）

2. **一个 JSON 结论文件**（保存到 `explore/distribution_conclusions.json`），格式：
   ```json
   {
     "mr": {
       "shape": "bimodal",
       "dip_test_p": 0.001,
       "gmm_valley": 3.8,
       "recommended_threshold": 3.8,
       "threshold_method": "gmm_valley"
     },
     "tn": {
       "shape": "unimodal_right_skew",
       "recommended_threshold": 5.5,
       "threshold_method": "p75_percentile"
     },
     "cs": {
       "shape": "compressed_normal",
       "recommended_threshold": null,
       "threshold_method": "not_recommended",
       "reason": "std=1.03, insufficient discriminative power"
     }
   }
   ```

---

## 数据读取方式

```python
import json

with open('data/full/output.jsonl') as f:
    records = [json.loads(l) for l in f
               if not json.loads(l).get('error') and json.loads(l).get('ok')]

# 提取分数（过滤异常值 >10）
def get_score(records, field):
    return [float(r['parsed'][field]) for r in records
            if isinstance(r['parsed'].get(field), (int, float))
            and 0 <= r['parsed'][field] <= 10]

# integrity 分组
intact  = [r for r in records if r['parsed'].get('ig') == 'intact']
partial = [r for r in records if r['parsed'].get('ig') == 'partial']
absent  = [r for r in records if r['parsed'].get('ig') == 'absent']
```

## 依赖库

```
scipy          # dip test (scipy.stats), GMM (sklearn.mixture.GaussianMixture)
sklearn
numpy
matplotlib / plotly   # 图表（推荐 plotly，输出 HTML）
pandas
```

运行入口：`uv run explore/distribution_analysis.py`
