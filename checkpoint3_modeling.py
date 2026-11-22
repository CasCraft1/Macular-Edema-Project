"""
What this script does:
  1. Loads `olives_visit_level_clean.csv` (TREX-DME cohort: 56 eyes, 47 patients)
     and engineers strictly causal (past-only) longitudinal trajectory features.
  2. Runs Patient-Grouped 5-Fold Cross-Validation (`GroupKFold` on `Patient_ID`)
     so neither fellow eyes nor repeated visits ever leak between train and test.
  3. Evaluates a 3-Stage Ablation Ladder:
       - Model 1 (Static Snapshot):      Current CST & BCVA only
       - Model 2 (Longitudinal Tabular): Adds nadir CST, delta CST, velocity,
                                         rolling volatility, visit order, & arm
       - Model 3 (Multimodal Fusion):    Automatically activates if `dataset.pk1`
                                         is present, pooling OCT B-scan features
                                         at the visit level (zero slice leakage!)
  4. Evaluates two clinically distinct classification setups + two regression tasks:
       - Task A1: Next-visit Rebound (`rebound_t_plus_1`) across all visits
       - Task A2: Dry-to-Rebound Early Warning (currently dry at visit t -> rebound at t+1)
       - Task B1: Next-visit CST forecasting (`CST_t_plus_1`, MAE / RMSE / R^2)
       - Task B2: Next-visit BCVA forecasting (`BCVA_t_plus_1`, MAE / RMSE / R^2)
  5. Exports:
       - `checkpoint3_model_comparison.csv`
       - `checkpoint3_feature_importance.csv` (out-of-fold on best Dry-to-Rebound model)
       - `checkpoint3_results.png` (ROC curves, out-of-fold importance, & threshold tradeoff)

Requirements (all pre-installed in Anaconda):
    pip install numpy pandas scipy scikit-learn matplotlib
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_CSV = os.path.join(SCRIPT_DIR, "olives_visit_level_clean.csv")
PICKLE_FILE = os.path.join(SCRIPT_DIR, "dataset.pk1")

MODEL_RESULTS_CSV = os.path.join(SCRIPT_DIR, "checkpoint3_model_comparison.csv")
IMPORTANCE_CSV = os.path.join(SCRIPT_DIR, "checkpoint3_feature_importance.csv")
SUMMARY_PLOT_PNG = os.path.join(SCRIPT_DIR, "checkpoint3_results.png")

INJECTION_COST_USD = 2000.0  # Benchmark anti-VEGF injection cost (Puri, 2026)


def load_and_engineer_features(csv_path: str) -> pd.DataFrame:
    """Load the clean visit-level panel and build causal longitudinal features for TREX-DME."""
    df = pd.read_csv(csv_path)
    dme = df[df["Trial"] == "TREX DME"].copy()
    dme = dme.sort_values(["Eye_ID", "Visit_Order"]).reset_index(drop=True)

    # Strictly causal (past-only) longitudinal trajectory & volatility features
    dme["CST_lag1"] = dme.groupby("Eye_ID")["CST"].shift(1)
    dme["BCVA_lag1"] = dme.groupby("Eye_ID")["BCVA"].shift(1)
    dme["delta_BCVA"] = dme.groupby("Eye_ID")["BCVA"].diff()
    dme["rolling_cst_std"] = (
        dme.groupby("Eye_ID")["CST"].transform(lambda s: s.expanding().std()).fillna(0.0)
    )
    dme["rolling_cst_mean"] = (
        dme.groupby("Eye_ID")["CST"].transform(lambda s: s.expanding().mean())
    )
    dme["cst_vs_baseline"] = dme["CST"] - dme.groupby("Eye_ID")["CST"].transform("first")
    dme["is_trex_or_gila"] = dme["Regimen_Arm"].isin(["TREX", "GILA"]).astype(int)

    # Keep transitions where next-visit targets and 1-step lag deltas exist
    mod = dme.dropna(
        subset=["rebound_t_plus_1", "CST_t_plus_1", "BCVA_t_plus_1", "delta_CST", "delta_BCVA"]
    ).copy()
    mod["rebound_t_plus_1"] = mod["rebound_t_plus_1"].astype(int)

    print(
        f"[Data] TREX-DME modeling panel: {len(mod)} visit transitions across "
        f"{mod['Eye_ID'].nunique()} eyes ({mod['Patient_ID'].nunique()} unique patients)."
    )
    print(
        f"       Overall t+1 Rebound rate: {mod['rebound_t_plus_1'].mean()*100:.1f}% "
        f"({mod['rebound_t_plus_1'].sum()}/{len(mod)})"
    )
    dry_sub = mod[mod["trex_rebound_now"] == 0]
    print(
        f"       Dry-to-Rebound subset:    {dry_sub['rebound_t_plus_1'].mean()*100:.1f}% "
        f"({dry_sub['rebound_t_plus_1'].sum()}/{len(dry_sub)} dry visits rebound at t+1)"
    )
    return mod


def extract_visit_level_oct_features(pickle_path: str) -> pd.DataFrame | None:
    """
    Optional Multimodal Bridge for `dataset.pk1` (created by `dataset_script.py`).
    CRITICAL: Aggregates 49 B-scan slices into 1 feature vector per (Eye_ID, Visit_Label)
    so slices from the same visit are NEVER split across train and test folds.
    """
    if not os.path.isfile(pickle_path):
        print("[OCT] 'dataset.pk1' not found in repo folder — running tabular ablation (Models 1 & 2).")
        print("      (Once 'dataset.pk1' is present, Model 3 Multimodal Fusion will run automatically.)")
        return None

    print(f"[OCT] Found '{pickle_path}'. Extracting visit-level B-scan summary features...")
    img_df = pd.read_pickle(pickle_path)

    # Identify grouping keys available in dataset.pk1
    key_cols = [c for c in ["Eye_ID", "Visit_Label", "Visit_Order"] if c in img_df.columns]
    if "Image" not in img_df.columns or len(key_cols) < 2:
        print("[OCT] Could not find expected ('Eye_ID', 'Visit_Label', 'Image') columns in dataset.pk1; skipping.")
        return None

    def slice_stats(arr: np.ndarray) -> list[float]:
        """Fast, interpretable radiomic/intensity descriptors per OCT B-scan (focusing on central retina)."""
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 3:
            a = a.mean(axis=-1)
        h, w = a.shape
        # Central foveal crop (middle 50% horizontally & vertically)
        center = a[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4]
        # Dark fluid-pocket proxy (low-intensity pixels inside central macula)
        p10, p50, p90 = np.percentile(center, [10, 50, 90])
        dark_frac = float(np.mean(center < p10 + 0.15 * (p90 - p10 + 1e-6)))
        bright_frac = float(np.mean(center > p90))
        grad_y = float(np.mean(np.abs(np.diff(center, axis=0))))
        grad_x = float(np.mean(np.abs(np.diff(center, axis=1))))
        return [
            float(center.mean()),
            float(center.std()),
            float(p10),
            float(p50),
            float(p90),
            dark_frac,
            bright_frac,
            grad_y,
            grad_x,
        ]

    stat_names = [
        "oct_mean",
        "oct_std",
        "oct_p10",
        "oct_p50",
        "oct_p90",
        "oct_dark_fluid_proxy",
        "oct_bright_hrf_proxy",
        "oct_grad_vertical",
        "oct_grad_horizontal",
    ]
    extracted = np.vstack([slice_stats(im) for im in img_df["Image"]])
    feat_df = pd.DataFrame(extracted, columns=stat_names, index=img_df.index)
    combined = pd.concat([img_df[key_cols[:2]], feat_df], axis=1)

    # Collapse 49 B-scans -> visit-level mean & std across slices
    visit_oct = combined.groupby(key_cols[:2], as_index=False).agg(
        {col: ["mean", "std"] for col in stat_names}
    )
    visit_oct.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0] for c in visit_oct.columns
    ]
    print(f"[OCT] Extracted {len(visit_oct.columns) - 2} visit-level OCT features across {len(visit_oct)} visits.")
    return visit_oct


def evaluate_classifier_cv(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    model,
    n_splits: int = 5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Run Patient-Grouped K-Fold CV and return out-of-fold metrics + probabilities."""
    X = df[features].copy()
    y = df[target_col].values
    groups = df["Patient_ID"].values

    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(df), dtype=float)

    for tr_idx, te_idx in gkf.split(X, y, groups):
        pipe = clone(model)
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        oof_probs[te_idx] = pipe.predict_proba(X.iloc[te_idx])[:, 1]

    preds = (oof_probs >= 0.5).astype(int)
    metrics = {
        "ROC_AUC": round(float(roc_auc_score(y, oof_probs)), 4),
        "PR_AUC": round(float(average_precision_score(y, oof_probs)), 4),
        "Accuracy": round(float(accuracy_score(y, preds)), 4),
        "Brier_Score": round(float(brier_score_loss(y, oof_probs)), 4),
    }
    return metrics, y, oof_probs


def compute_oof_permutation_importance(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    model,
    n_splits: int = 5,
    n_repeats: int = 30,
) -> pd.DataFrame:
    """
    Compute pooled out-of-fold permutation importance (zero in-sample leakage):
    fits 5 Patient-Grouped CV models once, then for each feature and repeat,
    shuffles that column across the dataset, predicts each patient strictly with
    their held-out fold model, and measures the drop from pooled OOF ROC-AUC.
    """
    X = df[features].reset_index(drop=True).copy()
    y = df[target_col].values
    groups = df["Patient_ID"].values

    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    oof_base = np.zeros(len(df), dtype=float)

    for tr_idx, te_idx in gkf.split(X, y, groups):
        pipe = clone(model)
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        oof_base[te_idx] = pipe.predict_proba(X.iloc[te_idx])[:, 1]
        folds.append((te_idx, pipe))

    base_auc = roc_auc_score(y, oof_base)
    rng = np.random.default_rng(42)
    drops = np.zeros((len(features), n_repeats), dtype=float)

    for f_idx, col in enumerate(features):
        orig_vals = X[col].values.copy()
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[col] = rng.permutation(orig_vals)
            oof_perm = np.zeros(len(df), dtype=float)
            for te_idx, pipe in folds:
                oof_perm[te_idx] = pipe.predict_proba(X_perm.iloc[te_idx])[:, 1]
            drops[f_idx, r] = base_auc - roc_auc_score(y, oof_perm)

    return (
        pd.DataFrame(
            {
                "Feature": features,
                "Importance_Mean_AUC_Drop": drops.mean(axis=1),
                "Importance_SD": drops.std(axis=1),
            }
        )
        .sort_values("Importance_Mean_AUC_Drop", ascending=False)
        .reset_index(drop=True)
    )


def evaluate_regressor_cv(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    model,
    n_splits: int = 5,
) -> dict:
    """Run Patient-Grouped K-Fold CV for continuous next-visit forecasting."""
    X = df[features].copy()
    y = df[target_col].values
    groups = df["Patient_ID"].values

    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(df), dtype=float)

    for tr_idx, te_idx in gkf.split(X, y, groups):
        pipe = clone(model)
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        oof_preds[te_idx] = pipe.predict(X.iloc[te_idx])

    return {
        "MAE": round(float(mean_absolute_error(y, oof_preds)), 3),
        "RMSE": round(float(root_mean_squared_error(y, oof_preds)), 3),
        "R2": round(float(r2_score(y, oof_preds)), 4),
    }


def run_checkpoint3():
    mod = load_and_engineer_features(CLEAN_CSV)

    # Optionally merge visit-level OCT features from dataset.pk1
    oct_df = extract_visit_level_oct_features(PICKLE_FILE)
    oct_cols = []
    if oct_df is not None:
        join_keys = [c for c in ["Eye_ID", "Visit_Label"] if c in oct_df.columns and c in mod.columns]
        mod = mod.merge(oct_df, on=join_keys, how="left")
        oct_cols = [c for c in oct_df.columns if c not in join_keys]

    # Feature sets for the 3-Stage Ablation Ladder
    feats_static = ["CST", "BCVA"]
    feats_longitudinal = [
        "CST",
        "BCVA",
        "nadir_CST",
        "delta_CST",
        "pct_change_CST",
        "pct_above_nadir",
        "delta_BCVA",
        "rolling_cst_std",
        "cst_vs_baseline",
        "Visit_Order",
        "is_trex_or_gila",
    ]
    feature_sets = {
        "M1: Static Snapshot (CST, BCVA)": feats_static,
        "M2: Longitudinal Trajectory": feats_longitudinal,
    }
    if oct_cols:
        feature_sets["M3: Multimodal (Trajectory + OCT)"] = feats_longitudinal + oct_cols

    # Candidate classifiers (all inside Pipelines to prevent scaling/imputation leakage;
    # all use class_weight="balanced" for fair comparison on the imbalanced dry subset)
    classifiers = {
        "Logistic Regression (L2)": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300, max_depth=5, min_samples_leaf=8, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.05,
                        max_iter=120,
                        min_samples_leaf=15,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    summary_rows = []
    roc_curves_dry = {}

    # ------------------------------------------------------------------
    # TASK A: CLASSIFICATION (All Visits vs. Dry-to-Rebound Early Warning)
    # ------------------------------------------------------------------
    dry_df = mod[mod["trex_rebound_now"] == 0].reset_index(drop=True)

    for cohort_name, df_sub in [
        ("All TREX-DME Visits (n=821)", mod),
        ("Dry-to-Rebound Early Warning (n=428)", dry_df),
    ]:
        for feat_label, feat_list in feature_sets.items():
            for clf_name, clf_pipe in classifiers.items():
                m, y_true, y_prob = evaluate_classifier_cv(df_sub, feat_list, "rebound_t_plus_1", clf_pipe)
                summary_rows.append(
                    {
                        "Task": cohort_name,
                        "Feature_Set": feat_label,
                        "Model": clf_name,
                        **m,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "R2": np.nan,
                    }
                )
                # Store Logistic Regression ROC curves for the Dry-to-Rebound comparison plot
                if "Dry-to-Rebound" in cohort_name and clf_name == "Logistic Regression (L2)":
                    roc_curves_dry[feat_label] = (y_true, y_prob, m["ROC_AUC"])

    # ------------------------------------------------------------------
    # TASK B: CONTINUOUS NEXT-VISIT FORECASTING (CST_t+1 & BCVA_t+1)
    # ------------------------------------------------------------------
    regressors = {
        "Ridge Regression": Pipeline(
            [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("reg", Ridge(alpha=5.0))]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("reg", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=120, random_state=42)),
            ]
        ),
    }

    for target_col, naive_col, unit in [
        ("CST_t_plus_1", "CST", "um"),
        ("BCVA_t_plus_1", "BCVA", "letters"),
    ]:
        # Naive Persistence Baseline (predict next visit = current visit)
        naive_mae = mean_absolute_error(mod[target_col], mod[naive_col])
        naive_rmse = root_mean_squared_error(mod[target_col], mod[naive_col])
        naive_r2 = r2_score(mod[target_col], mod[naive_col])
        summary_rows.append(
            {
                "Task": f"Next-Visit {target_col} ({unit})",
                "Feature_Set": "Baseline: Naive Persistence",
                "Model": "Carry-Forward (y_t+1 = y_t)",
                "ROC_AUC": np.nan,
                "PR_AUC": np.nan,
                "Accuracy": np.nan,
                "Brier_Score": np.nan,
                "MAE": round(float(naive_mae), 3),
                "RMSE": round(float(naive_rmse), 3),
                "R2": round(float(naive_r2), 4),
            }
        )
        for feat_label, feat_list in feature_sets.items():
            if feat_label.startswith("M1"):
                continue
            for reg_name, reg_pipe in regressors.items():
                rm = evaluate_regressor_cv(mod, feat_list, target_col, reg_pipe)
                summary_rows.append(
                    {
                        "Task": f"Next-Visit {target_col} ({unit})",
                        "Feature_Set": feat_label,
                        "Model": reg_name,
                        "ROC_AUC": np.nan,
                        "PR_AUC": np.nan,
                        "Accuracy": np.nan,
                        "Brier_Score": np.nan,
                        **rm,
                    }
                )

    results_df = pd.DataFrame(summary_rows)
    results_df.to_csv(MODEL_RESULTS_CSV, index=False)

    print("\n" + "=" * 98)
    print("CHECKPOINT 3: PATIENT-GROUPED 5-FOLD CV MODEL COMPARISON")
    print("=" * 98)
    print(results_df.to_string(index=False))
    print("=" * 98)
    print(f"[Saved] Model comparison table -> '{MODEL_RESULTS_CSV}'")

    # ------------------------------------------------------------------
    # TASK C: OUT-OF-FOLD PERMUTATION IMPORTANCE (Best Dry-to-Rebound Model)
    # ------------------------------------------------------------------
    best_label = max(roc_curves_dry, key=lambda k: roc_curves_dry[k][2])
    best_feats = feature_sets[best_label]
    best_tag = best_label.split(":")[0]
    print(f"[Best Dry-to-Rebound Model] {best_label} (ROC-AUC={roc_curves_dry[best_label][2]:.4f})")

    imp_df = compute_oof_permutation_importance(
        dry_df,
        best_feats,
        "rebound_t_plus_1",
        classifiers["Logistic Regression (L2)"],
        n_splits=5,
        n_repeats=30,
    )
    imp_df.to_csv(IMPORTANCE_CSV, index=False)
    print(f"[Saved] Out-of-fold permutation feature importance ({best_tag}) -> '{IMPORTANCE_CSV}'")

    # ------------------------------------------------------------------
    # TASK D: 3-PANEL CLINICAL & ECONOMIC SUMMARY FIGURE
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=160)

    # Panel 1: ROC Curves on Dry-to-Rebound Early Warning
    ax = axes[0]
    for label, (yt, yp, auc_val) in roc_curves_dry.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        ax.plot(fpr, tpr, lw=2.2, label=f"{label} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_title("Dry-to-Rebound Early Warning (ROC)", fontsize=11, fontweight="bold")
    ax.set_xlabel("False Positive Rate (Unnecessary Early Shot)")
    ax.set_ylabel("True Positive Rate (Rebounds Caught Early)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25)

    # Panel 2: Top 8 Out-of-Fold Feature Importances (Best Model)
    ax = axes[1]
    top_imp = imp_df.head(8).iloc[::-1]
    ax.barh(
        top_imp["Feature"],
        top_imp["Importance_Mean_AUC_Drop"],
        xerr=top_imp["Importance_SD"],
        color="#2b6cb0",
        error_kw={"ecolor": "#4a5568", "elinewidth": 1.2, "capsize": 3},
    )
    ax.axvline(0, color="#718096", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.set_title(f"Top Predictors ({best_tag} Out-of-Fold)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Out-of-Fold Permutation Importance (ROC-AUC Drop)")
    ax.grid(axis="x", alpha=0.25)

    # Panel 3: Proactive TREX Threshold Tradeoff (Best Model)
    ax = axes[2]
    yt_best, yp_best, _ = roc_curves_dry[best_label]
    thresholds = np.linspace(0.15, 0.85, 50)
    rebounds_caught_pct = []
    shots_avoided_pct = []
    for th in thresholds:
        pred_pos = yp_best >= th
        tp = np.sum((pred_pos == 1) & (yt_best == 1))
        total_pos = np.sum(yt_best == 1)
        rebounds_caught_pct.append(100.0 * tp / max(total_pos, 1))
        shots_avoided_pct.append(100.0 * np.mean(pred_pos == 0))

    ax.plot(thresholds, rebounds_caught_pct, lw=2.2, color="#c53030", label="% of Rebounds Caught Early")
    ax.plot(thresholds, shots_avoided_pct, lw=2.2, color="#2f855a", label="% of Dry Visits Safely Extended")
    ax.set_title(f"Proactive TREX Threshold Tradeoff ({best_tag})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Rebound Probability Threshold")
    ax.set_ylabel("Percentage of Visits (%)")
    ax.legend(loc="lower center", fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(SUMMARY_PLOT_PNG, bbox_inches="tight")
    print(f"[Saved] 3-panel summary figure -> '{SUMMARY_PLOT_PNG}'")


if __name__ == "__main__":
    run_checkpoint3()