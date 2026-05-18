# SAE 论文扩展严格版

基于黑目少女 2026-05-17 原稿之扩展工作。

## 文件

- `main.tex` — 完整 LaTeX 源码（中文，需 ctex 包 + xelatex 编译）

## 相对原稿的补全

| 原稿不足 | 本扩展版的处理 |
|---|---|
| 概念未形式化（"特征"、"独特性"是修辞概念） | § 2 给出 4 个形式定义：隐式生成过程、L-NL SAE、特征重合率、精确/稳健可识别性 |
| Locatello 适用前提未证（直接声称"SAE 没有归纳偏置"） | 命题 5.1 显式证明 L1 稀疏不足以打破不可识别性；推论 5.2 给出**弱版本**的无条件不可识别性结论 |
| 三定理"乘"成下界，无推导 | 引理 6.1 给出实际推导（量阶下界），并诚实标注常数 C₁, C₂ 未独立追踪 |
| ρ 公式是量纲拼接，不是公式 | 推论 6.3 从引理 6.1 出发显式构造，给出"自由空间"假设 η 的角色 |
| 数学语言与文学修辞混合 | 保留原文叙事意图，将每一处"必然"、"无情"等修辞主张转化为带显式假设的命题或评述 |
| 无局限性讨论 | § 7 严格区分**严格证明 / 有条件结论 / 启发性陈述**三档 |

## 关键诚实点

本扩展版没有"修复所有问题"，而是把**原稿暗藏的修辞滑坡显式化**：

1. § 5 的不可识别性结论是**弱版本**（"存在等价类"），不是"所有偏置都无效"
2. § 6 的下界是**量阶下界**，不是带具体常数的精确下界
3. ρ 公式与 30% 实测之间的对应被明确标注为**方向性**，非数值预言

这是把原稿从 essay 升级为"开放问题论文"的最大限度。要进一步达到完整定理论文水准，
还需要：(a) Toy Models 上的数值验证；(b) C₁/C₂/C₃ 的解析追踪；
(c) TopK 等更强偏置下的可识别性分析。

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
cd /mnt/c/Users/rog/Desktop/SAE_extended
xelatex main.tex
xelatex main.tex   # 第二遍以确定交叉引用与目录
```

约 1.5 GB 磁盘占用。

### 方案 C：Docker（如果你装了 Docker Desktop）

```bash
docker run --rm -v "$(pwd)":/data -w /data texlive/texlive xelatex main.tex
```
