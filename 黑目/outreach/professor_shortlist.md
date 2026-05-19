# SAE non-uniqueness outreach shortlist

Generated from `/home/jes/cuhk_sz/professor_matcher` on 2026-05-19 using:

- semantic queries over the local Chroma index
- exact keyword scans over `data/all_professors_enriched.json`
- manual inspection of research areas, bio snippets, publication snippets, and email fields

## Recommended first email

### 1. 樊继聪 Jicong Fan

- Email: fanjicong@cuhk.edu.cn
- URL: https://sds.cuhk.edu.cn/teacher/331
- Why: strongest local fit for the representation/factorization side. The matcher corpus lists AI/ML, matrix/tensor methods, clustering, graph learning, anomaly detection, AutoML, and an IJCAI 2025 paper titled "Explainable Graph Representation Learning via Graph Pattern Analysis". Exact keyword scans also hit `explainable`, `autoencoder`, `dictionary`, and `representation learning`.
- Ask angle: whether framing SAE feature instability as an identifiability/factorization problem is mathematically meaningful, and whether the sparse/dictionary-learning part is salvageable.

## Strong alternatives

### 2. 孙若愚 Ruoyu Sun

- Email: sunruoyu@cuhk.edu.cn
- URL: https://sds.cuhk.edu.cn/teacher/628
- Why: strongest fit for deep learning theory, nonconvex optimization, generative models, and the implicit-bias gap exposed by the toy experiments. The matcher ranked him first for "deep learning theory nonconvex optimization representation learning generative models identifiability lower bounds".
- Ask angle: whether the conditional lower-bound/Pareto-bound story is defensible, and whether the failed scaling-law predictions point to SGD implicit bias rather than pure identifiability.

### 3. 李阳 Yang Li

- Email: yangl@cuhk.edu.cn
- URL: https://sai.cuhk.edu.cn/teacher/234
- Why: trustworthy AI, reliable and interpretable machine learning, transferability quantification, continual learning, simple/interpretable model design.
- Ask angle: whether this is relevant to interpretability as a reliability problem, not just as pure theory.

### 4. 王本友 Benyou Wang

- Email: wangbenyou@cuhk.edu.cn
- URL: https://sds.cuhk.edu.cn/teacher/571
- Why: NLP, information retrieval, applied ML, pre-trained language models, parameter compression, and a NAACL best explainable-paper award. More LLM/NLP-adjacent than dictionary-learning-adjacent.
- Ask angle: whether the SAE instability argument matters for LLM interpretability/NLP model analysis.

### 5. 刘圳 Zhen Liu

- Email: zhenliu@cuhk.edu.cn
- URL: https://sds.cuhk.edu.cn/teacher/1951
- Why: machine learning, generative models, representation learning, foundation models; Mila/Schölkopf-adjacent background in the listed bio/publications.
- Ask angle: whether the Locatello/nonlinear-ICA/disentanglement framing is a reasonable bridge to modern representation learning.

### 6. 代忠祥 Zhongxiang Dai

- Email: daizhongxiang@cuhk.edu.cn
- URL: https://sds.cuhk.edu.cn/teacher/1746
- Why: ML theory and applications, LLMs, bandits, robustness/initialization guarantees. Useful if the message is about robustness of learned features across seeds.
- Ask angle: whether the experimental protocol and seed-robustness question can be turned into a clean ML problem.

## Lower-priority but plausible

### 尹峰 Feng Yin

- Email: yinfeng@cuhk.edu.cn
- URL: https://sai.cuhk.edu.cn/teacher/97
- Why: Bayesian ML, optimization, statistical signal processing, sparsity-aware modeling, GP latent-variable-model collapse.
- Ask angle: sparsity-aware latent variable modeling and non-identifiability.

### 李永春 Yongchun Li

- Email: yongchunli@cuhk.edu.cn
- URL: https://sds.cuhk.edu.cn/teacher/2149
- Why: interpretable/fair ML, sparse PCA, sparse truncated SVD, maximum-entropy sampling, optimization guarantees.
- Ask angle: sparse factorization and interpretability from optimization/statistics.

### JENTZEN, Arnulf

- Email: ajentzen@cuhk.edu.cn
- URL: https://sds.cuhk.edu.cn/teacher/452
- Why: mathematical deep learning, stochastic approximation, SGD/Adam convergence and failure modes.
- Ask angle: proof-level feedback on optimization/implicit-bias claims. This is mathematically strong but less directly SAE/interpretability-aligned.

### 吴保元 Baoyuan Wu

- Email: wubaoyuan@cuhk.edu.cn
- URL: https://sai.cuhk.edu.cn/teacher/121
- Why: AI safety/trustworthy AI, ML, CV, top AI venues.
- Ask angle: interpretability/safety relevance. Less directly matched to identifiability theory.

## Do not lead with these

- 贺品嘉: strong LLM/trustworthy AI and AI-for-SE profile, but less close to SAE identifiability or representation theory.
- 陈廷欢: ranked high by semantic queries but profile is mainly VLSI/CAD/deep-learning accelerators, likely a false positive for this task.
- 冀晓强, 甘培润, 刘梦琳: semantic hits from broad AI terms, but local records do not show enough connection to SAE identifiability.
