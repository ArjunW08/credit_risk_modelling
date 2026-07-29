# Exploratory Data Analysis

This document summarises the exploratory data analysis (EDA) and feature-engineering steps carried out in the Jupyter notebooks before model training. The work was done across three notebooks:

- [data_ingestion.ipynb](notebooks/data_ingestion.ipynb) — loading raw Excel files into SQLite
- [dataset_analysis.ipynb](notebooks/dataset_analysis.ipynb) — cleaning, feature screening, and encoding
- [model_training.ipynb](notebooks/model_training.ipynb) — baseline model comparison and hyperparameter search

---

## 1. Data Sources

| Table | Rows | Columns | Description |
|-------|------|---------|-------------|
| `internal_product` | 51 336 | 26 | Internal credit-account data (trade-line counts, product types, account ages) |
| `cibil_score` | 51 336 | 62 | CIBIL bureau data (delinquency, enquiries, demographics, credit score, target) |
| `feature_target_description` | 88 | 3 | Metadata describing every feature and the target variable |

Both datasets share a common join key: **`PROSPECTID`**.

The modelling target is **`Approved_Flag`** with four classes:

| Class | Meaning | Count |
|-------|---------|-------|
| P1 | Suitable for approval | 4 908 |
| P2 | Approval with caution | 25 452 |
| P3 | Risky to approve | 6 440 |
| P4 | High risk to approve | 5 264 |

> **Note:** The target is **imbalanced** — P2 accounts for ~60 % of samples, while P1 is only ~12 %. This influenced the choice of evaluation metric (macro F1-score) during model tuning.

---

## 2. Missing-Value Handling

The data uses the sentinel value **`-99999`** to represent missing values rather than standard NaN/null. A two-pass strategy was applied:

### Pass 1 — Drop columns with excessive missingness

Columns where more than **10 000** rows (≈19 %) contained `-99999` were dropped entirely. Eight columns were removed from the CIBIL dataset:

| Dropped Column | Reason |
|----------------|--------|
| `time_since_first_deliquency` | >10 000 sentinel values |
| `time_since_recent_deliquency` | >10 000 sentinel values |
| `max_delinquency_level` | >10 000 sentinel values |
| `max_deliq_6mts` | >10 000 sentinel values |
| `max_deliq_12mts` | >10 000 sentinel values |
| `CC_utilization` | >10 000 sentinel values |
| `PL_utilization` | >10 000 sentinel values |
| `max_unsec_exposure_inPct` | >10 000 sentinel values |

### Pass 2 — Drop remaining sentinel rows

Any remaining rows still containing `-99999` in any column were removed. This reduced the dataset from 51 336 to roughly 42 066 clean rows per table.

---

## 3. Data Merging

The two cleaned tables were merged via an **inner join** on `PROSPECTID`:

```
combined = merge(internal_data, cibil_data, on='PROSPECTID', how='inner')
```

**Result:** 42 064 rows × 79 columns (after dropping the `PROSPECTID` key).

---

## 4. Feature Screening

Features were separated into **categorical** and **numerical** groups and screened independently.

### 4a. Categorical Features — Chi-Square Test

A chi-square test of independence was run between each categorical column and `Approved_Flag` (α = 0.05).

| Feature | p-value | Decision |
|---------|---------|----------|
| `MARITALSTATUS` | 3.58 × 10⁻²³³ | **Keep** |
| `EDUCATION` | 2.69 × 10⁻³⁰ | **Keep** |
| `GENDER` | 1.91 × 10⁻⁵ | **Keep** |
| `last_prod_enq2` | 0.0 | **Keep** |
| `first_prod_enq2` | 7.85 × 10⁻²⁸⁷ | **Keep** |

All five categorical features showed a statistically significant association with the target, so none were dropped.

### 4b. Numerical Features — VIF + ANOVA

Numerical features went through a two-stage filter:

#### Stage 1 — Variance Inflation Factor (VIF ≤ 6.0)

VIF measures multicollinearity. Columns were evaluated **sequentially** (not in parallel, to avoid associated features dropping each other). Columns with VIF > 6 were removed from the feature matrix immediately before evaluating the next.

Key VIF insights:
- **VIF = 1** — no multicollinearity
- **VIF 1–5** — low multicollinearity
- **VIF 5–10** — moderate multicollinearity
- **VIF > 10** — high multicollinearity (dropped)

Notable drops at this stage:

| Dropped | VIF | Reason |
|---------|-----|--------|
| `Total_TL` | ∞ | Perfect collinearity |
| `Tot_Active_TL` | 11.3 | High |
| `pct_active_tl` | ∞ | Perfect collinearity |
| `AGE` | 22.1 | High |
| `Credit_Score` | 12.4 | High |
| `tot_enq` | 16.2 | High |

#### Stage 2 — ANOVA F-test (p < 0.05)

The surviving numerical features were tested against the four-class target using one-way ANOVA. Features with p ≥ 0.05 were dropped:

| Dropped | p-value | Reason |
|---------|---------|--------|
| `num_lss_12mts` | 0.3549 | Not associated with target |
| `pct_currentBal_all_TL` | 0.6083 | Not associated with target |

**Final result:** 79 columns → **44 columns** (35 numerical columns dropped, 0 categorical dropped).

---

## 5. Categorical Feature Distributions

After screening, the remaining categorical features had the following value distributions:

### Marital Status
| Value | Count |
|-------|-------|
| Married | 30 886 |
| Single | 11 178 |

### Gender
| Value | Count |
|-------|-------|
| M | 37 345 |
| F | 4 719 |

### Education
| Value | Count |
|-------|-------|
| GRADUATE | 14 140 |
| 12TH | 11 703 |
| SSC | 7 241 |
| UNDER GRADUATE | 4 572 |
| OTHERS | 2 291 |
| POST-GRADUATE | 1 898 |
| PROFESSIONAL | 219 |

### First Product Enquiry
| Value | Count |
|-------|-------|
| others | 20 640 |
| ConsumerLoan | 11 075 |
| PL | 4 431 |
| AL | 2 641 |
| CC | 1 988 |
| HL | 1 289 |

### Last Product Enquiry
| Value | Count |
|-------|-------|
| ConsumerLoan | 16 480 |
| others | 13 653 |
| PL | 7 553 |
| CC | 2 195 |
| AL | 1 353 |
| HL | 830 |

---

## 6. Feature Encoding

### Ordinal Encoding — EDUCATION

Education levels carry an inherent order and were mapped to integers:

| Original Value | Encoded Value | Rationale |
|----------------|---------------|-----------|
| SSC | 1 | Basic secondary |
| OTHERS | 1 | Treated as baseline (to be verified by business) |
| 12TH | 2 | Higher secondary |
| GRADUATE | 3 | Tertiary |
| UNDER GRADUATE | 3 | Grouped with graduate |
| PROFESSIONAL | 3 | Grouped with graduate |
| POST-GRADUATE | 4 | Advanced degree |

### One-Hot Encoding — Nominal Features

The following categorical features were one-hot encoded using `pd.get_dummies`:

- `MARITALSTATUS` → `MARITALSTATUS_Married`, `MARITALSTATUS_Single`
- `GENDER` → `GENDER_F`, `GENDER_M`
- `first_prod_enq2` → 6 columns (AL, CC, ConsumerLoan, HL, PL, others)
- `last_prod_enq2` → 6 columns (AL, CC, ConsumerLoan, HL, PL, others)

**Final encoded dataset:** 42 064 rows × 56 columns (including `Approved_Flag`). This was persisted to the `credit_risk_load_data` table in the SQLite database.

---

## 7. Baseline Model Comparison

Three classifiers were trained on an 80/20 train-test split to establish baselines:

| Model | Accuracy | Notes |
|-------|----------|-------|
| Decision Tree | 71.2 % | `max_depth=20`, `min_samples_split=10` |
| Random Forest | 76.7 % | `n_estimators=200` |
| **XGBoost** | **77.6 %** | `multi:softmax`, 4 classes |

### Per-Class Performance (XGBoost Baseline)

| Class | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| P1 | 0.820 | 0.765 | 0.792 |
| P2 | 0.824 | 0.913 | 0.866 |
| P3 | 0.465 | 0.291 | 0.358 |
| P4 | 0.726 | 0.736 | 0.731 |

> **Key insight:** P3 (risky) has notably low recall (29 %) — the model struggles to identify this intermediate risk segment. P2 (the majority class) dominates performance.

---

## 8. Hyperparameter Tuning

XGBoost was selected as the best baseline and tuned using `RandomizedSearchCV` with 60 iterations and 5-fold cross-validation:

### Search Space

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 50–400 |
| `learning_rate` | 0.01–0.30 |
| `max_depth` | 3–22 |
| `subsample` | 0.6–1.0 |
| `colsample_bytree` | 0.6–1.0 |
| `gamma` | 0–5 |

### Best Parameters Found

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 356 |
| `learning_rate` | 0.068 |
| `max_depth` | 5 |
| `subsample` | 0.872 |
| `colsample_bytree` | 0.782 |
| `gamma` | 3.926 |

**Best CV accuracy:** 78.2 %

The tuned model is saved at `models/predict_loan_possibility_model.pkl`.

---

## 9. Key Takeaways and Open Items

### Insights

1. **Sentinel values are pervasive.** The `-99999` convention masks true missingness and required aggressive column/row filtering, losing ~18 % of rows.
2. **High multicollinearity.** 27 of 66 numerical features had VIF > 6, indicating substantial redundancy in the raw feature set.
3. **Class imbalance.** P2 is 4–5× more frequent than P1/P3/P4, inflating overall accuracy while hiding poor recall on minority classes.
4. **P3 is hard to predict.** All three models struggled with the "risky" segment (P3), suggesting it shares feature patterns with both P2 and P4.
5. **Moderate gamma preferred.** The tuned `gamma=3.926` (minimum loss reduction to split) indicates the model benefits from regularisation to prevent overfitting.

### Open Items

- [ ] Investigate class-weighting or oversampling (SMOTE) to improve P3 recall.
- [ ] Add chart-driven business interpretation (distributions, correlation heatmap, feature importance).
- [ ] Evaluate additional metrics: macro/weighted F1, AUC-ROC per class.
- [ ] Consider feature engineering: interaction terms, ratio features, binning.
- [ ] Validate the "OTHERS" → 1 education mapping with the business team.
- [ ] Assess whether `PROSPECTID` (retained by ANOVA at p=0.042) should be excluded as a non-informative identifier.
