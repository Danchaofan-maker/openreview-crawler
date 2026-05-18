# 精选方法论笔记
*2026-05-18 | rules_jes.json 设计哲学 + 综合评分方向*

---

## 规则设计的核心哲学

### 只排"能拍胸脯说不要"的

规则的可靠边界：只在多个独立信号同时指向同一结论时才排除。单一信号可以噪声很大，但三个独立维度同时指向"无骨架"的概率极低。这是 R2（mk_f + mr + er）的设计依据。

反面教训：用单一维度排除论文（如只用 ig=partial）会系统性误伤。R1 最初要求 ig=absent + tn<3 + md<2 + er>7 四条件，后来发现 absent 本身是定义层面的保证，多余条件是多余保守主义。

**层次：**
- 定义层面排除：ig=absent（无任何形式化论断，概念上确定）
- 多信号验证排除：R2（mk_f 是直接证据，mr/er 是数值确认）
- 分类层排除：ig in [partial, broken]（默认排，然后 rescue outlier）

### 排除方向 vs 保留方向的逻辑差异

- **排除规则 + OR**：叠加相关维度导致过度排除，同一底层特质被惩罚多次
- **保留/rescue 规则 + OR**：救 outlier——`mr>=7 OR tn>=9 OR cc=10` 能捞出被主信号低估的不同类型论文

规则集是纯排除逻辑（OR 触发排除），rescue_rules 是纯保留逻辑（OR 触发保留）。两者分离，互不干扰。

### rescue_rules 的本质是手动多样性机制

每一条 rescue 规则背后是一类被主打分信号系统性低估的论文：
- `mr>=7`：推导严格但 ig 分类为 partial/broken（标注不完整不等于无价值）
- `cc=10`：极端算力论文，工程边界探索，与数学理论是不同维度的价值
- `tn>=9`：理论新意极高但推导未完成
- `ar>=9 AND sg>=9`：极高现实性 + 极高泛化性，不同类型的重要性
- `hr_f`：人工标记需要关注的论文

这不是补漏洞，是在承认：任何单一打分轴都会系统性忽略某些类型的价值。

---

## 综合评分：为什么排名不够

### 纯排名的失效模式

FLD / Mahalanobis 距离排名会把"最像 intact 原型"的论文排在最前。结果：语料库里可能 80% 都是相同风格的统计学习理论论文。精读语料库不是最优论文列表，是对"当前 ML 理论空间"的覆盖采样。

**这就是为什么一直在救 outlier**——直觉上在做多样性保证。

### MMR：形式化多样性

Maximum Marginal Relevance（Carbonell & Goldstein, 1998）：

```
score(d) = λ · relevance(d) - (1-λ) · max_{d'∈S} similarity(d, d')
```

每轮选择"足够相关但和已选集合最不同"的论文。

**在我们的 case 里：**
- `relevance(d)` = FLD 分数（线性，一次计算，直接反映"理论强度"）
- `similarity(d, d')` = Mahalanobis 距离（考虑维度相关结构，避免 mr=9 的两篇论文仅因为都高分就被认为"不同"）
- λ 偏低（~0.5）：更偏多样性，因为目标是覆盖，不是提纯

### FLD 分数的推导

`w = Σ⁻¹ × (μ_intact - μ_absent)`

权重来自数据，不是拍脑袋。协方差矩阵自动处理维度相关性；intact 和 absent 均值差自动处理判别力。μ_intact 和 μ_absent 已在 `03_filter/space_report.html` 的 `IG_MEANS` 里。

### cs 的位置

cs 不进 FLD 也不进 similarity——cs 是关于"模型对摘要判断有多把握"的元信号。对 cs 低的论文，FLD 分数本身可信度低，可以用 cs 对最终分做轻微折扣：`final = fld_score × f(cs)`。

---

### MMR 的路径依赖问题与 Monte Carlo 解法

贪心 MMR 是路径依赖的：第一步永远选 FLD 最高分，前 ~50 步的选择顺序对初始值敏感，导致"第几名"不稳定。

但这是**排名**不稳定，不是**集合**不稳定——覆盖收敛后，哪 1500 篇进入集合是相对稳定的。

**解法：Monte Carlo ensemble + Borda count**

给 FLD 分加小幅高斯噪声（`ε = fld.std() × 0.1`），独立跑 N 次 MMR（N=50-100），每次每篇论文得到一个名次。最终按平均名次排序。

等价于对 MMR 路径分布取期望值，N→∞ 时路径随机性被平均掉，剩下系统性信号。N 个独立运行可以完全并行（`joblib.Parallel`），计算开销线性可控。

```python
for run in range(N):
    noisy_fld = fld_scores + noise_scale * randn(n)
    selected = mmr_select(X, noisy_fld, ...)
    ranks[selected, run] = [1..target_n]

mean_rank = ranks.mean(axis=1)
final = argsort(mean_rank)[:target_n]
```

**FLD 的计算陷阱**：FLD 权重必须从**全量论文**（30k）计算，不能从规则过滤后的子集计算。过滤后 absent 论文几乎被清空，对照组变成 rescue 救回来的 outlier，权重会严重偏移（md 出现负权重）。

**协方差矩阵条件数**：45.5，健康（<1000），无需 ridge 正则化。

---

## TODO

- [x] 实现 FLD 权重计算
- [x] 实现 MMR 选择脚本（`03_filter/mmr_select.py`）
- [x] 加入 Monte Carlo ensemble（N=100，joblib 并行，~30s on 32 cores）
- [ ] 确定 λ（当前 0.5，人工浏览语料库后调整）
- [ ] 与 danchaofan 的 rules 合并后重新跑
