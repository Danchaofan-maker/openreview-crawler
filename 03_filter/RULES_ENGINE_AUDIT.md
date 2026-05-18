# 规则引擎审查任务

## 背景

这个项目对 ~31k 篇 ML 论文做规则过滤，筛出精读语料库。规则引擎是核心逻辑——过滤结果直接影响后续 FLD+MMR 选择的 1500 篇论文。

**刚发现一个 NA 处理 bug（已修复），怀疑还有其他不一致问题。需要完整审查。**

---

## 规则格式（JSON）

```json
{
  "inter_logic": "OR",          // 规则间逻辑：OR=任一命中即排除
  "force_keep_hr": true,        // hr_f=true 的论文强制保留
  "keep_na": true,              // 字段缺失时视作"不命中"（倾向保留）
  "rescue_rules": [...],        // 救回规则：命中则强制保留（覆盖排除）
  "rules": [                    // 排除规则：命中则排除
    {
      "name": "...",
      "enabled": true,
      "negate": false,
      "internal_logic": "AND",  // 条件间逻辑：AND=全部满足才命中
      "conditions": [
        {"field": "mr", "op": "lte", "value": 3}
      ]
    }
  ]
}
```

**字段名映射**（规则文件用 UI 名，数据对象用短名）：
| 规则文件字段 | 数据对象字段 | 含义 |
|---|---|---|
| `integrity` | `ig` | 逻辑链完整性 |
| `marketing` | `mk_f` | 数学营销标记 |
| `human_review` | `hr_f` | 人工核验标记 |
| 其余字段 | 同名 | mr/tn/md/ar/er/tea/cc/ei/sg/cs |

**tea 字段特殊性**：只有论文同时含理论结果和数值实验时才有值，纯理论/纯实证为 null（~60% 为 null，这是正常的，不是 bug）。

---

## 三套规则引擎（需要全部审查）

### 1. `03_filter/mmr_select.py`（对象引擎）

核心函数：`_get_val` → `_eval_cond` → `_eval_rule` → `apply_rules`

运行环境：直接操作 Python dict（parsed 对象）

**已知修复（最近）**：
- 补充了 `marketing`/`human_review` 字段别名映射
- 修复了 AND 规则中 None 处理 bug：原来 None 被过滤后 AND 条件意外缩减（如 `er>4 AND tea<6.5` 对 tea=null 的论文变成了只有 `er>4`）。修复后 AND 规则中 None = False（字段缺失 = 条件不成立 = AND 中断）

**当前逻辑**：
```python
def _eval_rule(p, rule):
    results = [_eval_cond(p, c) for c in rule.get("conditions", [])]
    logic = rule.get("internal_logic", "AND")
    if logic == "AND":
        resolved = [False if r is None else r for r in results]
    else:  # OR: None 不参与
        resolved = [r for r in results if r is not None]
    if not resolved: return False
    hit = all(resolved) if logic == "AND" else any(resolved)
    return (not hit) if rule.get("negate") else hit
```

### 2. `03_filter/serve.py`（对象引擎）

核心函数：`_field_val` → `eval_cond` → `eval_rule` → `eval_config`

运行环境：直接操作 Python dict（parsed 对象）

**NA 处理方式**（与 mmr_select.py 不同）：
```python
def eval_cond(parsed, c, keep_na=True):
    v = _field_val(parsed, field)
    if v is None:
        return not keep_na  # keep_na=True → False（不命中）
```

None 在 `eval_cond` 层就变成了 False，不依赖 `eval_rule` 的聚合逻辑。

### 3. `03_filter/rules_compare.py` 和 `03_filter/threshold_viz.py`（DataFrame 引擎）

运行环境：pandas DataFrame（列名已是 UI 名，不需要字段映射）

**NA 处理方式**（与以上两者都不同）：
```python
# threshold_viz.py
if op == "lt":  return col.fillna(999)  < v   # 缺失视为极大值
if op == "gt":  return col.fillna(-999) > v   # 缺失视为极小值
```

fillna 策略：lt/lte 填 999（缺失 = 不命中），gt/gte 填 -999（缺失 = 不命中）。

`rules_compare.py` 的 NA 处理方式不同：
```python
if op == "lt":  return col.fillna(999)  < v
if op == "gte": return col.fillna(-999) >= v
```
与 threshold_viz.py 相同，但没有 rescue_rules 支持（待确认）。

---

## 审查清单

### A. 语义一致性（三套引擎行为是否一致）

对以下每个场景，三套引擎应该给出相同的保留/排除结论：

1. **AND 规则，字段全部有值**：基础情况，应该一致
2. **AND 规则，其中一个字段为 null**：
   - mmr_select.py: AND 中断 → 不命中 → 保留
   - serve.py: eval_cond 返回 False → AND 中断 → 不命中 → 保留
   - DataFrame 引擎: fillna(999/−999) → 根据算子方向决定
   - **问题**：fillna 策略对 AND 规则的 null 处理是否和其他两者一致？
3. **OR 规则，其中一个字段为 null**：
   - mmr_select.py: None 被过滤，其余条件决定
   - serve.py: eval_cond 返回 False，其余条件决定
   - 两者应该一致，但 DataFrame 引擎的 fillna 策略可能不同
4. **rescue_rules 是否在所有引擎里实现**：
   - mmr_select.py: ✅ apply_rules 里有
   - serve.py: ✅ eval_config 里有
   - rules_compare.py: 需确认
   - threshold_viz.py: ❌ eval_all 里没有 rescue_rules
5. **force_keep_hr 是否一致**：检查三套引擎都正确处理
6. **negate 标志**：检查三套引擎行为一致
7. **空 conditions 列表**：三套引擎对空条件规则的处理是否一致

### B. 字段映射完整性

在 mmr_select.py 和 serve.py 中，确认以下字段别名都有映射：
- `marketing` → `mk_f`
- `human_review` → `hr_f`
- `integrity` → `ig`

DataFrame 引擎不需要映射（列名已经是 UI 名），但要确认列名和规则字段名完全匹配。

### C. 边界情况

- 规则的 `enabled: false` 是否被所有引擎跳过
- `inter_logic` 默认值不存在时的 fallback
- `internal_logic` 默认值不存在时的 fallback
- conditions 为空列表时各引擎的行为

---

## 测试方法

写一个对比测试：同一批论文同一套规则，跑 mmr_select.py 和 serve.py 两个引擎，结果必须完全一致。

参考数据：
- 规则文件：`03_filter/rules/rules_jes.json`、`03_filter/rules/rules_danchaofan.json`
- 数据：`data/full/output.jsonl`（30983 篇，其中约 12272 篇有 tea 值）
- 已知基准：rules_jes.json 过滤后应保留 **6236 篇**

```python
# 建议的测试框架
def compare_engines(rules_path, data_path):
    cfg = json.loads(Path(rules_path).read_text())
    papers = load_papers(data_path)
    
    # mmr_select 引擎
    kept_mmr = {p['id'] for p in apply_rules(papers, cfg)}
    
    # serve 引擎
    kept_serve = {p['paper_id'] for p in papers if not eval_config(p['parsed'], cfg)}
    
    diff = kept_mmr.symmetric_difference(kept_serve)
    assert len(diff) == 0, f"{len(diff)} 篇结果不一致"
```

---

## 输出要求

1. 对每个审查项给出结论：一致 / 不一致 / 潜在问题
2. 发现的 bug 直接修复，并说明修复前后行为差异
3. threshold_viz.py 缺少 rescue_rules 是否需要补充（这个工具是交互式规则探索器，用户在里面设计规则，不需要最终过滤语义，可以不加——但要确认）
4. 如果 DataFrame 引擎的 fillna 策略和对象引擎的 None=False 语义存在不一致，说明影响范围（DataFrame 引擎只是可视化工具，不影响最终语料库，可能可以接受）
