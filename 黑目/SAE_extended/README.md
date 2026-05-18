# SAE 论文扩展严格版

基于黑目少女 2026-05-17 原稿之扩展工作。当前为 v3 完备版。

## 文件

- `main.tex` — 完整 LaTeX 源码（中文，需 ctex 包 + xelatex 编译）

## 版本演进

| 版本 | 行数 | 主要变化 |
|---|---|---|
| v1（原稿 SAE.pdf） | — | 4 节散文 + 量纲推理 + 8 篇引用 |
| v2 | 526 行 | 形式化定义、命题/引理环境、L1 不充分证明、局限性章节 |
| **v3（当前）** | ~900 行 | TopK 不充分证明、辅助变量逃生通道、可证伪标度律、实验协议、相关工作、完整附录证明、线性 SAE 工作样本 |

## v3 相对 v2 的具体补全

| 维度 | v2 状态 | v3 处理 |
|---|---|---|
| **符号系统** | 散落在各定义中 | § 2 给统一符号表 |
| **TopK 是否打破不可识别性** | 开放问题 | **命题 5.5 给出部分否定**：保 TopK 等价变换族 |
| **跨越 Locatello 门槛的途径** | 未讨论 | § 5.7 引入辅助变量（iVAE 路线），给构造性补集 |
| **聚类引理证明** | 正文要点 | **附录 A 完整证明**（4 步推导） |
| **ρ 与实测的关系** | "高度定性一致" | **§ 6.7 五条可证伪标度律** + § 7 实验协议 |
| **相关工作定位** | 缺失 | § 8 把本文嵌入 5 条文献链 |
| **可验证锚点** | 缺失 | **附录 B 线性 SAE 工作样本**：显式计算 \|I\|₂, Δμ, C₀ |
| **下游实践建议** | 仅"接受流动本体论" | § 9.A 四条具体建议（多种子报告、可识别性等级、辅助变量、玩具模型预验） |
| **κ 因子的处理** | 沿用 v1 | **去除 κ**，理由见评述 6.6：κ 属无条件层面、定义模糊、三定理代数链本身已闭合 |

## v3 中诚实承认未完成的部分

1. **TopK 在保 TopK 等价族之外是否打破识别性** — 完整刻画超出本文范围（评述 5.6）
2. **常数 C₁, C₂ 的精确数值依赖** — 未独立追踪原始三定理的证明常数（附录 A 评述）
3. **标度律的具体回归系数预测** — 仅给方向，未给斜率（局限性 § 10.3）
4. **toy model 数值验证** — 协议已具体到可执行，但本文未执行

## 编译方式

### 方案 A：Overleaf（推荐，零安装）

1. 打开 https://www.overleaf.com → New Project → Blank Project
2. 把 `main.tex` 全部复制粘贴进去
3. 左上 Menu → Compiler → 选 **XeLaTeX**
4. Recompile

### 方案 B：本地安装 TeX Live（WSL Ubuntu）

```bash
sudo apt-get update
sudo apt-get install -y texlive-xetex texlive-lang-chinese texlive-fonts-recommended
cd /home/rog/projects/openreview-crawler/黑目/SAE_extended
xelatex main.tex
xelatex main.tex   # 第二遍以确定交叉引用与目录
```

约 1.5 GB 磁盘占用。

### 方案 C：Docker（如果你装了 Docker Desktop）

```bash
docker run --rm -v "$(pwd)":/data -w /data texlive/texlive xelatex main.tex
```

## 在 openreview-crawler 评分维度下的预估

| 维度 | v1（原稿） | v3（当前） |
|---|---|---|
| mr 数学严谨 | 3–4 | 6–7（引理有完整证明、定义形式化） |
| tn 理论新颖 | 4–5 | 4–5（核心是论证补全，非新现象） |
| md 数学深度 | 4 | 6（引入压缩感知、iVAE、Welch 界等正规工具） |
| ar 假设现实 | 6 | 6（与 v1 同） |
| er 经验验证 | 0（无实验） | 3（实验协议但未执行） |
| ig 逻辑链完整性 | partial | **intact**（每个跳步都被严格论证或显式标注） |

v3 的目标是把 v1 从 "ig=partial" 推到 "ig=intact"——
所有跳步要么被严格证明，要么被显式标注为开放问题。
