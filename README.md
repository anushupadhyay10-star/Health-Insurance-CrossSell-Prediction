# Health Insurance Cross-Sell Prediction

Predict which health-insurance customers will also respond to a **vehicle-insurance** offer, and turn
those predictions into a profitable contact list.

- Data: `data/train.csv` (381,109 rows, labelled) and `data/test.csv` (127,037 rows, unlabelled)
- Target: `Response` (1 = interested). Positive rate **12.26 %** (7.2 : 1 imbalance)

## Project layout

```
src/eda.py            Step 1  exploratory analysis       -> outputs/figures/01-07, outputs/tables/*
src/pipeline.py       Step 2  feature engineering + leakage-safe preprocessing
src/models.py         Step 3  model definitions (LR, RF, XGBoost, LightGBM, CatBoost)
src/train_compare.py  Steps 3-4  training, comparison, evaluation -> figures 08-12, model_comparison.csv
src/business.py       Step 5  ranking, budget, profit simulation  -> figures 13-15, campaign_target_list.csv
```

Run order: `python src/eda.py && python src/train_compare.py && python src/business.py`
(about 3-4 minutes on 2 CPU cores). `pip install -r requirements.txt` first.

## 1. EDA - key findings

| Check | Result |
|---|---|
| Target | 12.26 % positive. A "predict nobody" model scores 87.7 % accuracy, so accuracy is useless here |
| Missing values | None in train or test |
| Duplicates | 0 exact duplicates; 269 rows (0.07 %) identical apart from `id`, none with conflicting labels - harmless, kept |
| Premium outliers | 10,320 rows (2.7 %) outside the IQR fences [1,912 ; 61,892], max 540,165. Skew 1.77 raw. 64,877 customers (17 %) sit exactly at the floor value 2,630 |
| Strongest signals | `Vehicle_Damage` (23.8 % respond if yes vs 0.5 % if no), `Previously_Insured` (0.09 % if yes vs 22.6 % if no), `Vehicle_Age` (4.4 % <1 yr, 17.4 % 1-2 yr, 29.4 % >2 yr), `Age` (peaks at 30-50, ~21 %; under 25 only 3.5 %) |
| Weak signals | `Vintage` (corr -0.001), `Annual_Premium` (corr 0.02), `Driving_License` (99.8 % have one) |
| High-cardinality | `Region_Code` (53 levels; region 28 = 18.7 % response), `Policy_Sales_Channel` (155 levels; channels 152/160 ~2-3 %, channels 26/124/156 ~19-22 %) |
| Drift | Train and test means match closely |

## 2. Preprocessing and features

All learned steps sit inside sklearn Pipelines, so they are fitted on the training split only.
`id` (and `Response`) are dropped inside `FeatureEngineer`, so they can never become predictors.

- `Gender`, `Vehicle_Damage` -> 0/1; `Vehicle_Age` -> ordinal 0/1/2
- `Annual_Premium` -> `log1p` (skew 1.77 -> -1.47: the floor spike makes the log skew negative, trees are unaffected either way)
- `Premium_Floor` flag, `Damage_NotInsured` interaction
- `Region_Code`, `Policy_Sales_Channel`: one-hot with rare levels pooled (Logistic Regression), cross-fitted
  `TargetEncoder` (RF, XGBoost, LightGBM), native categorical handling (CatBoost)
- Scaling only for Logistic Regression

## 3-4. Models and evaluation

Stratified split of train.csv: 64 % fit / 16 % validation (model choice + threshold) / 20 % untouched test.
No class re-weighting, so probabilities stay usable for the business step. Threshold = F1-optimal on validation.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8490 | 0.3383 | 0.321 | 0.759 | 0.451 | 0.774 |
| Random Forest | 0.8566 | 0.3730 | 0.328 | 0.756 | 0.457 | 0.780 |
| XGBoost | 0.8565 | 0.3690 | 0.323 | 0.775 | 0.456 | 0.774 |
| LightGBM | 0.8563 | 0.3695 | 0.330 | 0.745 | 0.457 | 0.783 |
| **CatBoost** | **0.8585** | **0.3746** | 0.329 | 0.768 | **0.461** | 0.780 |

(hold-out test, no-skill PR-AUC = 0.123)

- CatBoost wins on validation PR-AUC and is selected. The four tree models are within ~0.002 ROC-AUC of each
  other, so the choice among them is not decisive; the real gap is trees vs Logistic Regression (PR-AUC +0.03-0.04).
- At the default 0.5 threshold every model has recall of only 0.1-2 % (accuracy 87.7 %, identical to "predict nobody"),
  which is why thresholds/budgets are chosen on the ranking instead.
- Top CatBoost features: `Previously_Insured`, the `Damage_NotInsured` interaction, `Age`, `Policy_Sales_Channel`.
- Figures: `08_roc_pr_curves`, `09_confusion_matrices`, `10_model_comparison_bars`, `11_threshold_tradeoff`, `12_feature_importance`.

## 5. Business decision (assumptions are editable at the top of `src/business.py`)

| Assumption | Value |
|---|---|
| Cost per contact | 300 (same currency as premium) |
| Revenue per conversion | 10 % of that customer's `Annual_Premium` (first-year margin) |
| Campaign population | 127,037 customers (test.csv); hold-out results scaled up |

Break-even response probability is roughly 9.8 % at the mean premium. Customers are ranked by CatBoost probability.
The contact share (41 %) was chosen to maximise profit on the **validation** set, then measured on the **test** set:

| Strategy | Contacted | Conversions | Buyers reached | Profit | ROI |
|---|---|---|---|---|---|
| Contact everyone | 127,037 | 15,570 | 100 % | 11.2 M | 0.29 |
| Random 41 % | 52,503 | ~6,435 | 41 % | 4.6 M | 0.29 |
| Top 10 % by model | 12,703 | 5,112 | 33 % | 13.7 M | 3.59 |
| Top 20 % by model | 25,408 | 9,028 | 58 % | 22.4 M | 2.94 |
| Top 30 % by model | 38,111 | 12,328 | 79 % | 28.5 M | 2.50 |
| **Chosen: top 41 %** | **52,503** | **14,608** | **94 %** | **30.9 M** | **1.96** |
| Top 50 % by model | 63,519 | 15,387 | 99 % | 29.7 M | 1.56 |

- Contacting the top 41 % reaches 94 % of buyers, nearly triples profit versus contacting everyone (30.9 M vs 11.2 M),
  and is ~6.7x a random contact list of the same size.
- Expected profit from predicted probabilities (30.8 M) matches realised test profit (30.9 M), so the probabilities are well calibrated.
- The profit curve is flat near the optimum (contacting 35-45 % of customers stays within ~2.5 % of the maximum profit), so the budget can be set
  by call-centre capacity without losing much.
- Sensitivity (`15_sensitivity`): at cost 800 and margin 5 % the campaign is barely worth running (optimal share 1.7 %);
  at cost 100 and margin 20 % the optimum rises to ~53 % of customers.
- `outputs/campaign_target_list.csv` ranks all 127,037 customers in test.csv (model refit on all labelled data):
  52,503 flagged for contact, ~14,700 expected conversions, ~31.0 M expected profit.

## Caveats

- Cost and margin figures are illustrative assumptions, not measured business data; replace them with real numbers.
- Revenue is counted only for converted customers, with no churn, cannibalisation or cost of the sales process after contact.
- Single stratified hold-out split with fixed hyper-parameters (no tuning, no repeated CV), so small differences between the
  four tree models should not be over-interpreted.
- The dataset is historical and the response labels come from one past campaign; a live A/B test (model list vs random list) is the proper confirmation.
