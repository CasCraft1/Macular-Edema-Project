Dataset link: https://drive.google.com/file/d/1CcnKC69FamykjuU00WxIt7kY38yLml0f/view?usp=drive_link

# Macular Edema Project Overview

This repository contains our data preparation scripts, Checkpoint 2 statistical tests, and Checkpoint 3 predictive modeling pipelines for **Diabetic Macular Edema (DME)** using the **OLIVES (TREX-DME)** ophthalmic dataset.

## Project Goal

Diabetic Macular Edema happens when leaking blood vessels cause fluid to build up in the center of the retina (the macula), threatening vision. Treatment relies on repeated anti-VEGF eye injections. Under the **Treat-and-Extend (TREX)** clinical protocol, doctors space out visits once an eye dries out, defined by **Central Subfield Thickness (CST) <= 325 um**, and wait until fluid rebounds (**CST >= 325 um** or **>= 20% above the eye's lowest recorded thickness, called its nadir**) before injecting again.

Because TREX is **reactive**, doctors wait for swelling to return before intervening. We test whether routine clinical history and Optical Coherence Tomography (OCT) eye scans already captured at every appointment can predict a rebound **one visit in advance**, especially when an eye currently looks dry (`Dry-to-Rebound Early Warning`).

## Repository Structure

### `main` Branch (Data Prep, Checkpoint 2 Stats & Tabular/Multimodal Modeling)

| File | What It Is |
| :--- | :--- |
| `OlivesData.py` | Data-cleaning and Checkpoint 2 statistical analysis script. |
| `olives_visit_level_clean.csv` | Clean visit-level OLIVES dataset used directly by `checkpoint3_modeling.py` and `dataset_script.py`. |
| `dataset_script.py` | Local preprocessing script that filters `olives_visit_level_clean.csv` to TREX-DME, loads each visit's 3D OCT `.npy` scan volume, expands each 2D B-scan slice into a row, and writes `dataset.csv` (quick preview) and `dataset.pk1` (pickled NumPy arrays; stored externally due to file size). |
| `dataset.csv` | Slice-level TREX-DME preview table (39 columns: identifiers, clinical measurements `CST` and `BCVA`, 16 OCT biomarker fractions graded at baseline/final visits, causal longitudinal/target columns, and truncated pixel text). |
| `image_visualization_test.py` | Quick 15-line sanity check that loads `dataset.pk1` and displays slice 0 in grayscale with Matplotlib to verify the pickled OCT pixel arrays. |
| `checkpoint2_statistical_tests.csv` | Complete 27-test statistical ledger from Checkpoint 2 across all 96 OLIVES eyes (1,593 visits), covering normality, loading-phase CST/BCVA changes, group comparisons, correlations, and OCT biomarker stratification. |
| `checkpoint3_modeling.py` | Main Checkpoint 3 modeling pipeline. Builds causal per-eye features, runs patient-grouped 5-fold cross-validation across a 3-stage ablation ladder (`M1` -> `M2` -> `M3`), computes pooled out-of-fold feature importance, and exports the results CSVs and 3-panel figure. |
| `checkpoint3_model_comparison.csv` | Out-of-fold 5-fold CV performance table (28 rows) comparing all models across 4 tasks: All Visits Rebound, Dry-to-Rebound Early Warning, Next-Visit CST Regression, and Next-Visit BCVA Regression. |
| `checkpoint3_feature_importance.csv` | Pooled out-of-fold permutation importance (`Importance_Mean_AUC_Drop` +/- `Importance_SD`) for all 11 features in the winning Dry-to-Rebound model (`M2`). |
| `checkpoint3_results.png` | 3-panel Checkpoint 3 summary figure: (1) Dry-to-Rebound ROC curves (`M1` vs. `M2` vs. `M3`), (2) Top 8 out-of-fold M2 predictors, and (3) Proactive TREX threshold tradeoff curve for M2. |
| `OLIVES_Dataset_Labels/` | Raw OLIVES clinical and biomarker label spreadsheets/CSVs (`full_labels/` and `ml_centric_labels/`) plus study summaries. |
| `References/` | Reference paper (`12967_2024_Article_5141.pdf`), supplementary table (`12967_2024_5141_MOESM2_ESM.xlsx`), and `ML Notes.odt`. |
| `LICENSE` | Repository license file. |

*Note: `olives_visit_level_clean.csv` is included in the repository; `dataset.pk1` (the large OCT volume pickle) is kept outside GitHub via Google Drive (link above) and is only needed locally to run `M3`.*

### `CNN/` Folder (PyTorch 3D CNN Multimodal Arm on `main` & `CNN` Branch)

The [`CNN/` folder](https://github.com/CasCraft1/Macular-Edema-Project/tree/main/CNN) (merged into `main` from the [`CNN` branch](https://github.com/CasCraft1/Macular-Edema-Project/tree/CNN)) explores end-to-end deep learning directly on stacked 3D OCT volumes:

| File | What It Is |
| :--- | :--- |
| `CNN/models.py` | Defines `OCTDataset` (stacks 46 slices per eye-visit, normalizes `/ 255`) and `OCT_CNN` (3-layer 3D CNN producing a 32-dim image embedding concatenated with 6 clinical trajectory features), trained with MSE loss for `CST_t_plus_1` regression. |
| `CNN/model_classifier.py` | Defines `OCT_CNN_CLASSIFIER` (same 3D CNN + 6 tabular features backbone) trained with `BCEWithLogitsLoss` to classify `Meets_TREX_Criteria` (`CST_t_plus_1 >= 325` or `>= 1.2 * nadir_CST`) across all valid visits on an 80/20 patient-grouped split. |
| `CNN/benchmarks.py` | Evaluation and diagnostic plotting script for the saved PyTorch weights. |
| `CNN/classifier_fig.png` | ROC curve for `OCT_CNN_CLASSIFIER` on the 20% held-out patient test split (**AUC = 0.926** on All Visits). |
| `CNN/model_cst.png` | 3-panel regression diagnostic plot (Predicted vs. Actual CST, Residual Plot, and Error Distribution in um). |
| `CNN/weights.torch`, `CNN/weights_classifier.torch` | Saved PyTorch model weights for regression and classification. |
| `CNN/train_indices.npy`, `CNN/test_indices.npy`, `CNN/volumeLevels.csv` | Saved 80/20 `GroupShuffleSplit` indices and visit-level volume metadata table. |

## Background (`checkpoint2_statistical_tests.csv`)

Before building predictive models, 27 formal statistical tests across the full 96-eye OLIVES cohort (1,593 visits; 1,497 transitions) established why static one-visit snapshots are insufficient:

- **Skewed clinical measurements:** `CST` (median 276.0 um, skew 1.67), `BCVA` (median 82.0 letters, skew -1.23), and `delta_CST` (median -2.0 um, skew -1.41) all fail Shapiro-Wilk normality (`p < 1e-26`), justifying non-parametric tests alongside parametric checks.
- **Strong loading-phase response:** In TREX-DME (`n = 56` eyes), mean CST drops from **468.8 um** at baseline to **350.2 um** by Visit 3 (`p = 3.17e-11`), while visual acuity improves from **67.9** to **78.4** ETDRS letters (`p = 7.60e-11`).
- **Vision and swelling do not move in lockstep:** Concurrent CST and BCVA correlate only modestly (`Spearman rho = -0.3305`), and static baseline CST does not predict final vision (`rho = 0.1029, p = 0.4506`). However, early CST velocity (`V1 -> V3` % drop) strongly tracks long-term CST reduction (`rho = 0.6309, p = 5.59e-12`).
- **Fluid biomarkers drive both swelling and vision loss:** Across the `n = 192` graded baseline and final visits, **Intraretinal Fluid (IRF)** (`rho = -0.5882`, Active Edema Odds Ratio `37.8`, `p = 3.47e-09`) and **Subretinal Fluid (SRF)** (`rho = -0.4366`, OR `19.87`, `p = 2.24e-10`) are strongly associated with both worse vision and swelling >= 325 um. Structural markers (`IR HRF`, `Disruption of EZ`, `Disruption of RPE`) correlate with worse vision without significantly associating with >= 325 um swelling.

## Methodology Explained

### 1. Cohort & Prediction Targets

In `checkpoint3_modeling.py`, we filter to the **TREX-DME** trial (**56 eyes across 47 unique patients**). After dropping each eye's very first visit (no prior visit exists to measure change) and very last visit (no next visit exists to check the outcome), we have **821 visit-to-visit transitions**.

We evaluate two classification tasks and two continuous forecasting tasks:

- **All TREX-DME Visits (`n = 821`, 44.9% rebound at `t + 1`):** Predicts whether any visit will meet TREX rebound criteria (`rebound_t_plus_1 == 1`) at the next appointment.
- **Dry-to-Rebound Early Warning (`n = 428`, 11.2% rebound at `t + 1`):** Filters strictly to eyes that are **currently dry** (`trex_rebound_now == 0`). This is the clinically decisive task: when an eye is already swollen today, a doctor already knows to inject it; when an eye looks dry today, spotting the 11.2% about to rebound next visit is where proactive care changes treatment.
- **Next-Visit Continuous Forecasting:** Predicts exact next-visit thickness (`CST_t_plus_1` in um) and next-visit vision (`BCVA_t_plus_1` in ETDRS letters), benchmarked against a "Naive Persistence" baseline (simply guessing next visit = today's value).

### 2. How We Prevent Data Leakage

Medical machine-learning models can easily look artificially accurate if test data leaks into training. Both pipelines enforce strict safeguards:

- **Separate history per eye (`Eye_ID`):** Every historical measurement (running minimum `nadir_CST`, visit-to-visit `delta_CST`, `% change`, `% above nadir`, expanding standard deviation, and change from baseline) is calculated strictly within each individual eye using **past and current visits only**, never peeking at future visits.
- **Grouped by patient (`Patient_ID`) for train/test splits:** Nine patients contribute both a left eye (`OS`) and right eye (`OD`). Splitting by `Patient_ID` (`GroupKFold` with 5 folds on `main`; `GroupShuffleSplit` 80/20 on `CNN`) guarantees that **both eyes and all visits of a patient stay in the same fold**, so every score reflects performance on completely unseen patients.
- **Visit-level OCT pooling (no slice leakage):** Each OCT scan has ~46-49 2D B-scan slices. On `main`, slices are aggregated into one summary vector per visit (`mean` and `std` across slices) before splitting. On `CNN`, all 46 slices are stacked into a single 3D volume tensor per visit. Slices from the same appointment are never split across train and test.
- **Preprocessing inside each fold:** Median imputation and feature scaling (`StandardScaler`) are fit strictly inside each training fold's scikit-learn `Pipeline` on `main`.
- **Balanced class weights:** All three classifiers on `main` (Logistic Regression, Random Forest, HistGradientBoosting) use `class_weight="balanced"` so models cannot inflate accuracy by always guessing "no rebound" on the imbalanced Dry-to-Rebound subset (where 88.8% of dry visits stay dry).

### 3. The 3-Stage Ablation Ladder (`main`) vs. 3D CNN (`CNN`)

To measure whether adding more complex data actually improves predictions, `checkpoint3_modeling.py` tests three progressively richer feature sets using the exact same models and folds:

- **M1: Static Snapshot (2 features):** Today's `CST` and `BCVA` only.
- **M2: Longitudinal Trajectory (11 features):** Adds the eye's path over time (`nadir_CST`, `delta_CST`, `pct_change_CST`, `pct_above_nadir`, `delta_BCVA`, `rolling_cst_std`, `cst_vs_baseline`, `Visit_Order`) plus treatment schedule (`is_trex_or_gila`: `1` for spaced-out TREX/GILA, `0` for Monthly).
- **M3: Hand-Crafted Multimodal Fusion (29 features):** Adds 18 intensity and gradient statistics extracted from the central 50% crop of each OCT B-scan volume (`dataset.pk1`).
- **3D CNN Multimodal Arm (`CNN/`):** Replaces hand-crafted pixel statistics with a PyTorch 3-layer 3D CNN (`Conv3d` 1->8->16->32) that learns a 32-number embedding directly from the 46-slice OCT volume, concatenated with 6 core M2 features (`CST`, `BCVA`, `nadir_CST`, `delta_CST`, `pct_change_CST`, `pct_above_nadir`).

### 4. How to Read the Metrics

- **ROC-AUC (higher is better; 0.50 = coin flip):** If you pick one visit that *will* rebound and one that will *stay dry*, ROC-AUC is the probability the model assigns a higher risk score to the rebounding eye. It is much more reliable than plain Accuracy when rebounds are rare (11.2%).
- **PR-AUC (higher is better):** Precision-Recall AUC focuses specifically on catching the rare positive class (random baseline = `0.112` on Dry-to-Rebound).
- **Brier Score (lower is better):** Measures how well-calibrated the predicted probabilities are.
- **MAE / RMSE / R^2:** Average absolute error, outlier-penalized error, and share of variance explained for continuous CST (um) and BCVA (letters) forecasting.

## Checkpoint 3 Results

### Main Pipeline (`checkpoint3_model_comparison.csv`, 5-Fold Patient CV)

Across all four tasks on `main`, **M2 (Longitudinal Trajectory)** outperforms both static snapshots (`M1`) and hand-crafted OCT pixel fusion (`M3`):

| Task | Best Model (`main`) | Key Out-of-Fold Metrics | vs. M1 / Baseline & M3 |
| :--- | :--- | :--- | :--- |
| **All TREX-DME Visits (`n = 821`)** | M2 Random Forest | ROC-AUC **0.9250**, PR-AUC **0.9198**, Brier **0.1064** | Beats M1 (`0.9117`) & M3 (`0.9162`) |
| **Dry-to-Rebound (`n = 428`)** | M2 Logistic Regression (L2) | ROC-AUC **0.7958**, PR-AUC **0.3012**, Acc **0.7056** | Beats M1 (`0.6844`) & M3 (`0.7161`) |
| **Next-Visit CST (`um`)** | M2 Ridge Regression | MAE **25.642**, RMSE **41.777**, R^2 **0.7298** | Beats Carry-Forward (`46.784` RMSE) & M3 (`43.776`) |
| **Next-Visit BCVA (`letters`)** | M2 Ridge Regression | MAE **3.118**, RMSE **4.203**, R^2 **0.8314** | Beats Carry-Forward (`4.487` RMSE) & M3 (`4.322`) |

### Key Takeaways From `checkpoint3_results.png` & `checkpoint3_feature_importance.csv`

1. **Trajectory adds major early-warning power:** On currently dry eyes (`n = 428`), adding longitudinal history lifts ROC-AUC by **+0.112** over a single-visit snapshot (`0.684` -> `0.796`).
2. **Hand-crafted pixel stats overfit on 47 patients:** Adding 18 unsegmented central-crop OCT stats (`M3`) dropped Dry-to-Rebound AUC from `0.796` to `0.716` (`-0.080`) and worsened every regression task. Without clinician-drawn Region-of-Interest (ROI) fluid masks, coarse pixel summaries add variance rather than signal at this sample size.
3. **What actually drives early warning (Pooled Out-of-Fold Permutation Importance on M2):**
   - **Treatment schedule:** `is_trex_or_gila` (`0.100 +/- 0.024` AUC drop) is the #1 predictor. Spaced-out Treat-and-Extend eyes rebound far more often than Monthly eyes.
   - **Subclinical swelling drift:** `pct_change_CST` (`0.058 +/- 0.013`) and `pct_above_nadir` (`0.055 +/- 0.014`) are the strongest anatomical early warnings, even while an eye still reads "dry" (`<= 325 um`), creeping upward from its prior visit or personal nadir flags impending rebound.
   - **Vision lags behind fluid:** `BCVA` (`0.0008`) and `delta_BCVA` (`-0.0003`) add essentially zero predictive signal once thickness trajectory is known, confirming Checkpoint 2's functional-anatomical dissociation.
4. **Actionable clinical tradeoff (Panel 3):** Around a predicted probability cutoff of `0.54-0.55`, M2 catches roughly **~70% of upcoming rebounds early** while still **safely extending ~70% of dry visits**.

## How to Run (`main` Branch)

Requirements: Python 3 with `numpy`, `pandas`, `scikit-learn`, and `matplotlib` (all pre-installed in Anaconda).

```bash
python checkpoint3_modeling.py
```

- With `olives_visit_level_clean.csv` in the script folder, `M1` and `M2` run automatically and export `checkpoint3_model_comparison.csv`, `checkpoint3_feature_importance.csv`, and `checkpoint3_results.png`.
- If `dataset.pk1` is also present in the same folder, `M3` (visit-level OCT fusion) activates automatically.