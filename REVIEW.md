# GBA-1 Analysis — Methodology Review

**Scope:** `code_gba1 - Final.ipynb`, the 9 trained models under `Model_ML_GBA1/`, and `Data_GBA1_FIX-Clean.csv`.
**Goal of the work:** predict 10 self-report psychometric targets (Big-Five Openness `BFP.O_4item`; creativity scores `WC_All`, `C_Semantic_*`, `C_Likert_*` across the JE / DS / AoL subscales) from in-game behavior of **856 participants**.

This is a constructive review for the team — the pipeline is well-structured; the issues below are mostly about making the reported numbers honest and the repo reproducible.

---

## 1. What's good

- **Sound pipeline skeleton:** 80/20 holdout → 5-fold CV on train → `RandomizedSearchCV`/`GridSearchCV` tuning → final holdout eval. `random_state=42` throughout → reproducible.
- **Leakage-aware intent:** the custom `OutlierHandler` (cell 213) fits IQR bounds on **train only**, explicitly to avoid test leakage.
- **Breadth:** 10 model families (linear, regularized, SVR, KNN, RF, XGBoost, CatBoost, TabNet, MLP, PLS) × 6 feature-selection methods.
- **Deployment-minded:** each model saved with metadata JSON + an inference script. In the *selected* models, holdout RMSE ≈ CV RMSE (no gross train-set overfitting).

## 2. Key findings

**There is a real but weak signal.** Best Pearson r per target is 0.17–0.39 with significant p-values — game behavior carries genuine information about self-reported creativity/openness, but explains roughly 2–15% of variance at best.

**The headline R² values are optimistically biased by selection on the holdout.** Each target evaluated 144–216 configurations against the *same* 172-row holdout, then reported the max. The distribution of holdout R² across configs:

| target | best R² | median R² | configs with R² < 0 |
|---|---|---|---|
| likert_ds | 0.010 | −0.025 | **139 / 144** |
| likert_all | 0.081 | −0.005 | 128 / 216 |
| semantic_ds | 0.027 | −0.006 | 84 / 144 |
| likert_aol | 0.052 | −0.007 | 87 / 144 |
| semantic_je | 0.076 | 0.030 | 64 / 216 |
| wc_all | 0.087 | 0.058 | 16 / 144 |
| bpfo | 0.100 | 0.070 | 13 / 144 |
| semantic_all | 0.121 | 0.077 | 27 / 216 |
| semantic_aol | 0.149 | 0.097 | 19 / 216 |

When the median config scores ~0 and you report the max of ~200 noisy tries, the "best" is partly luck.
- **`likert_ds` is essentially noise** (139/144 negative; best 0.010) and should not be presented as a working model.
- The **DS subscales and most Likert targets** sit at/below noise.
- The defensible tier is **semantic_aol / semantic_all / bpfo / wc_all** — weak but plausible.

**The 50+ engineered features add almost nothing over raw game score.** For BPFO, a 2-feature model (`sum_weighted_score`, `avg_weighted_score`) reaches holdout R² = 0.093, matching the 15-feature SVR "winner" (0.0875). `sum_weighted_score` is the explicit anchor of nearly every composite (cell 137), so the `Mutant_* / Psych_* / Feat_* / Comp_*` features largely re-express the same performance signal.

## 3. Methodological problems (ranked)

1. **Feature-selection leakage.** FS rankings (Spearman/MI/RF/etc.) are computed on the *full* `final_df` (cell 209), holdout included, then hardcoded into training. FS must run inside the CV fold.
2. **Selection-on-test.** Picking the winning model/FS combo by holdout R² over ~200 configs turns the holdout into a second training set. Use a three-way split (train/dev/test) or nested CV; touch the true test once.
3. **BPFO target was median-imputed.** Cell 12: 121 of 856 `BFP.O_4item` values (~14%) were missing and filled with the median. ~34 of the 172 BPFO holdout rows therefore carry a constant (median) label — trivially fit, which deflates RMSE and inflates apparent reliability. Refit on the 735 genuine rows, or drop missing-target rows rather than impute the dependent variable. (Only BPFO was affected; the other 9 targets were complete.)
4. **PCA fit on full data** (cell 137) — minor unsupervised leakage; move it inside the pipeline.
5. **Arbitrary composite weights** — `Comp_GBA_*_Anchored` use hand-picked multipliers (×2, ×1.5, ×3, ×5) with no justification or sensitivity analysis. (Verified these are built from *gameplay*, not the targets — no direct target leakage there.)

## 4. Repo hygiene / reproducibility

- **Hardcoded Windows paths** in the inference cell: `C:\Users\willi\Documents\...` — won't run elsewhere. Parameterize.
- **Committed scratch:** `catboost_info/` (incl. a `tmp/` of `.tmp` files) and `.DS_Store` — now `.gitignore`d and removed from tracking.
- **9.6 MB notebook with embedded outputs** — recommend `nbstripout` before committing.
- **Dead/duplicated cells:** training pipeline pasted twice (226/227), `show_sample_predictions` redefined 4×, a stray `dawda` cell (222), mislabeled comments (cell 238 says "PEARSON > 0.15" inside the Spearman section).
- **Manual copy-paste of feature lists** from FS output into training cells — error-prone and already drifting (comments say ">0.15" while code uses ">0.2").

## 5. Recommendations

1. Put FS + scaling + PCA inside a `Pipeline`, inside CV. This single change makes the reported numbers honest.
2. Use a three-way split or nested CV; report spread (mean ± sd) across configs, not just the argmax.
3. Lead with the weak-signal story honestly; drop or clearly flag noise-level targets (esp. `likert_ds`).
4. Add baselines to every target: predict-the-mean, and a raw `sum_weighted_score`-only model. If engineered features can't beat raw score, that *is* the finding.
5. Fix the BPFO target handling (refit on the 735 genuine rows).
6. Repo cleanup: `.gitignore` (done), `nbstripout`, parameterized paths.

---

*Reviewed 2026-06-01. Findings derived from the committed notebook, model metadata, and per-config `training_results.csv` tables.*
