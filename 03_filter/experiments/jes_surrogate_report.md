# Jes Final Surrogate Explanation

- Target corpus: `03_filter/combined_corpus_jes_current.jsonl`
- Prefilter rules: `03_filter/rules/rules_jes.json`
- Target positives: `1500`
- Jes prefilter size: `6236`

## Predictability Upper Bound

| model | feature set | test top-k P | test top-k R | test top-k F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| random_forest_raw | 18 | 0.562 | 0.562 | 0.562 | 0.974 | 0.620 |
| extra_trees_raw | 18 | 0.529 | 0.529 | 0.529 | 0.966 | 0.550 |
| random_forest_derived | 23 | 0.633 | 0.633 | 0.633 | 0.983 | 0.724 |
| extra_trees_derived | 23 | 0.600 | 0.600 | 0.600 | 0.976 | 0.650 |

Top-k means the classifier is forced to select the same number of papers as the split contains positives.

## Feature Importance

| feature | importance |
|---|---:|
| jes_prefilter | 0.3756 |
| fld_norm | 0.0830 |
| mr | 0.0750 |
| combined_score | 0.0725 |
| ig_intact | 0.0711 |
| pca_norm | 0.0498 |
| tn | 0.0390 |
| er | 0.0346 |
| md | 0.0337 |
| sg | 0.0329 |
| ig_absent | 0.0231 |
| ei | 0.0226 |
| cluster_center_dist | 0.0191 |
| human_review | 0.0159 |
| cc | 0.0119 |

## Distilled Rule Set

Selected surrogate union: `kept=1829`, `hit=1176`, `precision=0.643`, `recall=0.784`, `F1=0.707`.

These rules explain membership tendencies. They are not equivalent to MMR because MMR is set-dependent.

### Rule 1: tree_222_leaf_189

- Own: kept=846, hit=549, precision=0.649, recall=0.366
- Marginal: kept=846, hit=549, precision=0.649
- Conditions:
  - `cluster_center_dist > 1.17`
  - `er <= 3.28`
  - `mr > 6.84`

### Rule 2: tree_034_leaf_156

- Own: kept=395, hit=300, precision=0.759, recall=0.200
- Marginal: kept=195, hit=140, precision=0.718
- Conditions:
  - `jes_prefilter = true`
  - `marketing = false`
  - `cluster_center_dist > 1.42`
  - `ei <= 8.68`
  - `md > 4.15`
  - `pca_norm > 0.344`

### Rule 3: tree_628_leaf_17

- Own: kept=70, hit=51, precision=0.729, recall=0.034
- Marginal: kept=67, hit=49, precision=0.731
- Conditions:
  - `integrity != intact`
  - `jes_prefilter = true`
  - `combined_score > 0.263`
  - `fld_norm <= 0.563`
  - `mr <= 6.61`
  - `pca_norm > 0.471`

### Rule 4: tree_692_leaf_93

- Own: kept=367, hit=253, precision=0.689, recall=0.169
- Marginal: kept=231, hit=130, precision=0.563
- Conditions:
  - `integrity != absent`
  - `jes_prefilter = true`
  - `tea_missing = true`
  - `fld_norm > 0.801`

### Rule 5: tree_082_leaf_136

- Own: kept=209, hit=175, precision=0.837, recall=0.117
- Marginal: kept=82, hit=63, precision=0.768
- Conditions:
  - `integrity != partial`
  - `jes_prefilter = true`
  - `cluster_center_dist > 1.28`
  - `cluster_log_size <= 5.85`
  - `combined_score > 0.629`
  - `er > 2.52`
  - `pca_norm > 0.481`

### Rule 6: tree_433_leaf_58

- Own: kept=92, hit=61, precision=0.663, recall=0.041
- Marginal: kept=64, hit=36, precision=0.562
- Conditions:
  - `integrity = intact`
  - `jes_prefilter = true`
  - `tea_missing = false`
  - `cluster_center_dist > 1.76`
  - `tn <= 7.99`

### Rule 7: tree_377_leaf_154

- Own: kept=183, hit=115, precision=0.628, recall=0.077
- Marginal: kept=73, hit=36, precision=0.493
- Conditions:
  - `integrity = partial`
  - `jes_prefilter = true`
  - `cluster_center_dist > 1.3`
  - `cluster_log_size <= 6.03`
  - `er > 2.25`
  - `fld_norm > 0.544`
  - `tea <= 6.89`

### Rule 8: tree_749_leaf_109

- Own: kept=28, hit=25, precision=0.893, recall=0.017
- Marginal: kept=14, hit=13, precision=0.929
- Conditions:
  - `human_review = true`
  - `cluster_center_dist > 2.12`
  - `combined_score > 0.26`
  - `pca_norm <= 0.556`

### Rule 9: tree_036_leaf_128

- Own: kept=72, hit=56, precision=0.778, recall=0.037
- Marginal: kept=34, hit=21, precision=0.618
- Conditions:
  - `jes_prefilter = true`
  - `tea_missing = false`
  - `cluster_center_dist > 1.11`
  - `combined_score > 0.558`
  - `cs > 7.11`
  - `fld_norm > 0.656`
  - `7.55 < mr <= 8.45`
  - `sg <= 6.77`

### Rule 10: tree_211_leaf_29

- Own: kept=58, hit=46, precision=0.793, recall=0.031
- Marginal: kept=30, hit=23, precision=0.767
- Conditions:
  - `integrity in ['intact', 'partial']`
  - `jes_prefilter = true`
  - `marketing = false`
  - `cluster_center_dist <= 1.67`
  - `mr > 8.05`
  - `sg <= 9.35`
  - `tea > -0.302`

### Rule 11: tree_360_leaf_122

- Own: kept=186, hit=143, precision=0.769, recall=0.095
- Marginal: kept=44, hit=23, precision=0.523
- Conditions:
  - `integrity = intact`
  - `cc <= 1.76`
  - `cluster_center_dist > 0.91`
  - `cluster_log_size <= 5.21`

### Rule 12: tree_081_leaf_83

- Own: kept=37, hit=30, precision=0.811, recall=0.020
- Marginal: kept=10, hit=9, precision=0.900
- Conditions:
  - `integrity != intact`
  - `jes_prefilter = true`
  - `marketing = false`
  - `cluster_center_dist > 0.814`
  - `cluster_log_size <= 4.73`
  - `ei > 5.12`
  - `er <= 7.66`
  - `tn <= 8.95`

### Rule 13: tree_717_leaf_131

- Own: kept=67, hit=45, precision=0.672, recall=0.030
- Marginal: kept=41, hit=24, precision=0.585
- Conditions:
  - `integrity = intact`
  - `cc > 0.698`
  - `cluster_log_size <= 5.55`
  - `cs > 7.24`
  - `mr <= 8.84`
  - `0.728 < pca_norm <= 0.8`

### Rule 14: tree_628_leaf_56

- Own: kept=44, hit=34, precision=0.773, recall=0.023
- Marginal: kept=13, hit=9, precision=0.692
- Conditions:
  - `integrity != intact`
  - `fld_norm > 0.563`
  - `md <= 4.75`
  - `mr > 7.64`

### Rule 15: tree_876_leaf_124

- Own: kept=41, hit=28, precision=0.683, recall=0.019
- Marginal: kept=22, hit=12, precision=0.545
- Conditions:
  - `integrity = intact`
  - `1.04 < cluster_center_dist <= 1.65`
  - `cs <= 8.08`
  - `er > 1.25`
  - `0.621 < fld_norm <= 0.818`
  - `pca_norm <= 0.646`
  - `sg > 6.5`

### Rule 16: tree_079_leaf_77

- Own: kept=51, hit=36, precision=0.706, recall=0.024
- Marginal: kept=22, hit=12, precision=0.545
- Conditions:
  - `integrity = intact`
  - `jes_prefilter = true`
  - `0.839 < cluster_center_dist <= 2.15`
  - `cluster_log_size > 5.3`
  - `combined_score > 0.374`
  - `er > 0.57`
  - `mr > 7.67`
  - `sg <= 7.19`
  - `tn <= 6.86`

### Rule 17: tree_190_leaf_115

- Own: kept=56, hit=42, precision=0.750, recall=0.028
- Marginal: kept=9, hit=8, precision=0.889
- Conditions:
  - `jes_prefilter = true`
  - `tea_missing = true`
  - `ar > 1.94`
  - `cluster_center_dist > 1.19`
  - `combined_score > 0.221`
  - `cs > 7.34`
  - `mr <= 7.13`
  - `sg > 6.05`

### Rule 18: tree_604_leaf_108

- Own: kept=39, hit=26, precision=0.667, recall=0.017
- Marginal: kept=17, hit=10, precision=0.588
- Conditions:
  - `integrity != intact`
  - `ar <= 6.5`
  - `cluster_center_dist > 1.08`
  - `cs > 7.52`
  - `er > 1.9`
  - `fld_norm > 0.71`
  - `md > 6.62`
  - `mr > 4.81`
  - `pca_norm > 0.472`

### Rule 19: tree_571_leaf_94

- Own: kept=39, hit=30, precision=0.769, recall=0.020
- Marginal: kept=15, hit=9, precision=0.600
- Conditions:
  - `integrity != absent`
  - `jes_prefilter = true`
  - `ar > 3.28`
  - `combined_score <= 0.841`
  - `ei > 8.04`
  - `0.376 < pca_norm <= 0.685`
  - `tea <= 4.63`

