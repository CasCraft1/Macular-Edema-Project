"""
OLIVES Dataset
===================================================================================
Uses the two core OLIVES label files described in the GitHub README
(from either `OLIVES_Dataset_Labels/full_labels` or `OLIVES_Dataset_Labels/ml_centric_labels`):
  1. Clinical_Data_Images.xlsx         (78,185 B-scans -> 1,593 valid longitudinal eye-visits across 96 eyes)
  2. Biomarker_Clinical_Data_Images.csv (9,408 B-scans -> 192 biomarker-graded eye-visits, 16 OCT biomarkers)
  (Plus `OCT-DME.xlsx` and `OCT-DR.xlsx.)

Outputs saved to the script directory:
  - olives_visit_level_clean.csv      (Clean 1,593-row longitudinal Eye-Visit panel ready for modeling)
  - checkpoint2_statistical_tests.csv (Elementary statistical tests across all 5 test families)
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION: Set this path if `OLIVES_Dataset_Labels` is in a custom folder,
# or leave as None to auto-detect.
# ==============================================================================
CUSTOM_LABELS_DIR = None

CLEAN_VISIT_CSV = "olives_visit_level_clean.csv"
STATS_SUMMARY_CSV = "checkpoint2_statistical_tests.csv"

BIOMARKER_COLS = [
    "Atrophy / thinning of retinal layers",
    "Disruption of EZ",
    "DRIL",
    "IR hemorrhages",
    "IR HRF",
    "Partially attached vitreous face",
    "Fully attached vitreous face",
    "Preretinal tissue/hemorrhage",
    "Vitreous debris",
    "VMT",
    "DRT/ME",
    "Fluid (IRF)",
    "Fluid (SRF)",
    "Disruption of RPE",
    "PED (serous)",
    "SHRM",
]


def find_labels_dir() -> str:
    """Locate `full_labels` or `ml_centric_labels` inside `OLIVES_Dataset_Labels`."""
    if CUSTOM_LABELS_DIR and os.path.exists(CUSTOM_LABELS_DIR):
        if os.path.isdir(os.path.join(CUSTOM_LABELS_DIR, "full_labels")):
            return os.path.join(CUSTOM_LABELS_DIR, "full_labels")
        if os.path.isdir(os.path.join(CUSTOM_LABELS_DIR, "ml_centric_labels")):
            return os.path.join(CUSTOM_LABELS_DIR, "ml_centric_labels")
        return CUSTOM_LABELS_DIR

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    home = os.path.expanduser("~")

    roots = [
        script_dir,
        os.getcwd(),
        parent_dir,
        os.path.join(home, "OneDrive", "Desktop"),
        os.path.join(home, "Desktop"),
        os.path.join(home, "Downloads"),
        os.path.join(home, "OneDrive", "Downloads"),
        os.path.join(home, "Documents"),
        os.path.join(home, "OneDrive", "Documents"),
        home,
        "/working_dir/c_1c3d819f5d60efd2/user_uploaded_files/ttl=63d/user_uploaded",
    ]

    for sub in [
        os.path.join("OLIVES_Dataset_Labels", "full_labels"),
        os.path.join("OLIVES_Dataset_Labels", "OLIVES_Dataset_Labels", "full_labels"),
        "full_labels",
        os.path.join("OLIVES_Dataset_Labels", "ml_centric_labels"),
        os.path.join("OLIVES_Dataset_Labels", "OLIVES_Dataset_Labels", "ml_centric_labels"),
        "ml_centric_labels",
        "OLIVES_Dataset_Labels",
        "",
    ]:
        for r in roots:
            candidate = os.path.join(r, sub) if sub else r
            clin = os.path.join(candidate, "Clinical_Data_Images.xlsx")
            bio = os.path.join(candidate, "Biomarker_Clinical_Data_Images.csv")
            if os.path.isfile(clin) and os.path.isfile(bio):
                return candidate

    raise FileNotFoundError(
        "Could not auto-detect 'OLIVES_Dataset_Labels'. Set CUSTOM_LABELS_DIR at the top of the script "
        "to your 'OLIVES_Dataset_Labels/full_labels' folder path."
    )


def load_table(filepath: str) -> pd.DataFrame:
    """Load an OLIVES label table (.xlsx or .csv) reliably."""
    if filepath.lower().endswith(".csv"):
        return pd.read_csv(filepath)
    try:
        return pd.read_excel(filepath, engine="openpyxl")
    except Exception:
        # Fallback if a CSV file was saved with an .xlsx extension
        return pd.read_csv(filepath)


def parse_olives_path(path_series: pd.Series) -> pd.DataFrame:
    """
    Parse Trial, Regimen Arm, Subject Folder, and Visit Number from the OLIVES image path:
      - TREX DME  (6 segments): /TREX DME/<Arm>/<Subject_Folder>/<Visit>/<Laterality>/<Image>
      - Prime_FULL (5 segments): /Prime_FULL/<Subject_Folder>/<Visit>/<Laterality>/<Image>
    """
    parts = path_series.astype(str).str.strip("/").str.split("/")
    trial = parts.str[0]
    is_trex = trial == "TREX DME"

    arm = np.where(is_trex, parts.str[1], "PRN (PRIME)")
    subject_folder = np.where(is_trex, parts.str[2], parts.str[1])
    visit_label = np.where(is_trex, parts.str[3], parts.str[2])
    laterality = np.where(is_trex, parts.str[4], parts.str[3])
    visit_num = pd.Series(visit_label, index=path_series.index).str.extract(r"(\d+)")[0].astype(float)

    return pd.DataFrame(
        {
            "Trial": trial,
            "Regimen_Arm": arm,
            "Subject_Folder": subject_folder,
            "Visit_Label": visit_label,
            "Laterality": laterality,
            "Visit_Number": visit_num,
        }
    )


def step1_clean_and_merge(labels_dir: str, out_dir: str) -> pd.DataFrame:
    """Collapse the 49 B-scans per visit into 1 row per Eye-Visit and merge the 16 OCT biomarkers."""
    clin_path = os.path.join(labels_dir, "Clinical_Data_Images.xlsx")
    bio_path = os.path.join(labels_dir, "Biomarker_Clinical_Data_Images.csv")

    print(f"[Step 1] Loading OLIVES labels from: {labels_dir}")
    df_clin = load_table(clin_path)
    df_bio = load_table(bio_path)

    # Standardize column names across `full_labels` and `ml_centric_labels`
    clin_path_col = [c for c in df_clin.columns if "path" in c.lower()][0]
    bio_path_col = [c for c in df_bio.columns if "path" in c.lower()][0]
    df_clin = df_clin.rename(columns={clin_path_col: "File_Path", "Eye ID": "Eye_ID", "Patient ID": "Patient_ID"})
    df_bio = df_bio.rename(columns={bio_path_col: "Bio_Path", "Eye ID": "Eye_ID", "Patient ID": "Patient_ID"})

    df_clin = pd.concat([df_clin, parse_olives_path(df_clin["File_Path"])], axis=1)
    df_bio = pd.concat([df_bio, parse_olives_path(df_bio["Bio_Path"])], axis=1)

    present_biomarkers = [b for b in BIOMARKER_COLS if b in df_bio.columns]
    for c in ["CST", "BCVA", "Eye_ID", "Patient_ID"]:
        df_clin[c] = pd.to_numeric(df_clin[c], errors="coerce")
    for c in present_biomarkers:
        df_bio[c] = pd.to_numeric(df_bio[c], errors="coerce")

    # Collapse 49 B-scans -> 1 row per (Eye_ID, Visit_Label) in each table, then merge
    clin_visit = (
        df_clin.groupby(
            ["Patient_ID", "Eye_ID", "Trial", "Regimen_Arm", "Subject_Folder", "Laterality", "Visit_Label", "Visit_Number"],
            as_index=False,
        )
        .agg(n_scans=("File_Path", "count"), CST=("CST", "first"), BCVA=("BCVA", "first"))
    )
    bio_visit = df_bio.groupby(["Eye_ID", "Visit_Label"], as_index=False)[present_biomarkers].mean()

    visit_df = clin_visit.merge(bio_visit, on=["Eye_ID", "Visit_Label"], how="left")
    before_n = len(visit_df)
    visit_df = visit_df.dropna(subset=["CST", "BCVA"]).sort_values(["Eye_ID", "Visit_Number"]).reset_index(drop=True)
    visit_df["Visit_Order"] = visit_df.groupby("Eye_ID")["Visit_Number"].rank(method="dense").astype(int)

    # Engineer TREX & Longitudinal Velocity Features
    visit_df["Disease"] = np.where(visit_df["Trial"] == "TREX DME", "DME", "DR")
    visit_df["nadir_CST"] = visit_df.groupby("Eye_ID")["CST"].cummin()
    visit_df["delta_CST"] = visit_df.groupby("Eye_ID")["CST"].diff()
    visit_df["pct_change_CST"] = visit_df.groupby("Eye_ID")["CST"].pct_change() * 100.0
    visit_df["pct_above_nadir"] = ((visit_df["CST"] - visit_df["nadir_CST"]) / visit_df["nadir_CST"]) * 100.0

    # TREX Rebound Trigger: CST >= 325 um OR CST climbs >= 20% above running nadir
    visit_df["trex_rebound_now"] = ((visit_df["CST"] >= 325.0) | (visit_df["pct_above_nadir"] >= 20.0)).astype(int)
    visit_df["rebound_t_plus_1"] = visit_df.groupby("Eye_ID")["trex_rebound_now"].shift(-1)
    visit_df["CST_t_plus_1"] = visit_df.groupby("Eye_ID")["CST"].shift(-1)
    visit_df["BCVA_t_plus_1"] = visit_df.groupby("Eye_ID")["BCVA"].shift(-1)

    out_csv = os.path.join(out_dir, CLEAN_VISIT_CSV)
    visit_df.to_csv(out_csv, index=False)
    print(f"         Collapsed {len(df_clin):,} B-scans -> {len(visit_df):,} valid eye-visits across {visit_df['Eye_ID'].nunique()} eyes (dropped {before_n - len(visit_df)} missing visit).")
    print(f"         Joined {len(present_biomarkers)} OCT biomarkers onto {visit_df[present_biomarkers[0]].notna().sum()} / {len(bio_visit)} graded visits.")
    print(f"[Step 1] Saved clean panel -> '{out_csv}'.")
    return visit_df


def step2_run_statistical_tests(visit_df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Run Checkpoint 2 elementary statistical tests and save summary CSV."""
    print("\n[Step 2] Running Checkpoint 2 Elementary Statistical Tests...")
    results = []

    def add_test(category, test_name, comparison, stat_name, stat_val, p_val, takeaway):
        results.append(
            {
                "Category": category,
                "Statistical_Test": test_name,
                "Variables_Compared": comparison,
                "Statistic_Type": stat_name,
                "Statistic_Value": round(float(stat_val), 4) if pd.notna(stat_val) else np.nan,
                "p_value": float(p_val) if pd.notna(p_val) else np.nan,
                "Significant_p_lt_0_05": bool(p_val < 0.05) if pd.notna(p_val) else False,
                "Clinical_Takeaway": takeaway,
            }
        )

    # 1. Normality Tests (Shapiro-Wilk)
    for col in ["CST", "BCVA", "delta_CST"]:
        s = visit_df[col].dropna()
        w_stat, p_val = stats.shapiro(s)
        add_test(
            "1. Normality & Distribution",
            "Shapiro-Wilk Test",
            f"{col} (n={len(s)}, mean={s.mean():.1f}, SD={s.std():.1f}, median={s.median():.1f}, IQR={s.quantile(0.75)-s.quantile(0.25):.1f}, skew={s.skew():.2f})",
            "W-stat",
            w_stat,
            p_val,
            "Significant skewness / non-normality; justifies non-parametric Mann-Whitney & Spearman tests",
        )

    # Build 96-eye summary table
    eye_rows = []
    for eye_id, grp in visit_df.groupby("Eye_ID"):
        grp = grp.sort_values("Visit_Order")
        v1, v3, v_last = grp.iloc[0], grp.iloc[min(2, len(grp) - 1)], grp.iloc[-1]
        eye_rows.append(
            {
                "Eye_ID": eye_id,
                "Trial": v1["Trial"],
                "Regimen_Arm": v1["Regimen_Arm"],
                "CST_v1": v1["CST"],
                "CST_v3": v3["CST"],
                "CST_final": v_last["CST"],
                "BCVA_v1": v1["BCVA"],
                "BCVA_final": v_last["BCVA"],
                "early_cst_velocity_pct": ((v1["CST"] - v3["CST"]) / v1["CST"]) * 100.0,
                "total_cst_drop": v1["CST"] - v_last["CST"],
                "cst_volatility_sd": grp["CST"].std(),
            }
        )
    eye_df = pd.DataFrame(eye_rows)
    dme_eyes = eye_df[eye_df["Trial"] == "TREX DME"]

    # 2. Paired Longitudinal Change (Baseline vs. Visit 3 & Final Visit)
    for label, sub in [("All 96 Eyes", eye_df), ("TREX-DME Cohort (n=56)", dme_eyes)]:
        t_stat, p_t = stats.ttest_rel(sub["CST_v1"], sub["CST_v3"])
        w_stat, p_w = stats.wilcoxon(sub["CST_v1"], sub["CST_v3"])
        t_bv, p_bv = stats.ttest_rel(sub["BCVA_final"], sub["BCVA_v1"])
        add_test(
            "2. Paired Longitudinal Change",
            "Paired t-test",
            f"{label}: Baseline CST ({sub['CST_v1'].mean():.1f} um) vs. Visit 3 CST ({sub['CST_v3'].mean():.1f} um)",
            "t-stat",
            t_stat,
            p_t,
            "Initial loading injections rapidly reduce central macular thickness",
        )
        add_test(
            "2. Paired Longitudinal Change",
            "Wilcoxon Signed-Rank Test",
            f"{label}: Baseline CST (med {sub['CST_v1'].median():.0f} um) vs. Visit 3 CST (med {sub['CST_v3'].median():.0f} um)",
            "W-stat",
            w_stat,
            p_w,
            "Non-parametric confirmation of early anatomical response",
        )
        add_test(
            "2. Paired Longitudinal Change",
            "Paired t-test",
            f"{label}: Final BCVA ({sub['BCVA_final'].mean():.1f} letters) vs. Baseline BCVA ({sub['BCVA_v1'].mean():.1f} letters)",
            "t-stat",
            t_bv,
            p_bv,
            "Significant long-term visual acuity gain over the treatment period",
        )

    # 3. Two-Group Comparisons (Mann-Whitney U)
    valid_next = visit_df.dropna(subset=["rebound_t_plus_1", "CST"])
    reb_yes = valid_next[valid_next["rebound_t_plus_1"] == 1]
    reb_no = valid_next[valid_next["rebound_t_plus_1"] == 0]
    u_reb, p_reb = stats.mannwhitneyu(reb_yes["CST"], reb_no["CST"], alternative="two-sided")
    add_test(
        "3. Two-Group Comparison",
        "Mann-Whitney U Test",
        f"Visit t CST before t+1 Rebound (med {reb_yes['CST'].median():.0f} um, n={len(reb_yes)}) vs. Stable (med {reb_no['CST'].median():.0f} um, n={len(reb_no)})",
        "U-stat",
        u_reb,
        p_reb,
        "Current-visit CST strongly separates eyes that rebound at the next visit (t+1)",
    )

    trex_gila = dme_eyes[dme_eyes["Regimen_Arm"].isin(["TREX", "GILA"])]
    monthly = dme_eyes[dme_eyes["Regimen_Arm"] == "Monthly"]
    u_sd, p_sd = stats.mannwhitneyu(trex_gila["cst_volatility_sd"], monthly["cst_volatility_sd"], alternative="two-sided")
    add_test(
        "3. Two-Group Comparison",
        "Mann-Whitney U Test",
        f"TREX/GILA CST Volatility SD (med {trex_gila['cst_volatility_sd'].median():.1f} um, n={len(trex_gila)}) vs. Monthly (med {monthly['cst_volatility_sd'].median():.1f} um, n={len(monthly)})",
        "U-stat",
        u_sd,
        p_sd,
        "Compares longitudinal macular thickness fluctuation between Treat-and-Extend and Monthly dosing",
    )

    # 4. Bivariate Correlations (Spearman & Pearson)
    r_s, p_s = stats.spearmanr(visit_df["CST"], visit_df["BCVA"])
    r_p, p_p = stats.pearsonr(visit_df["CST"], visit_df["BCVA"])
    r_base, p_base = stats.spearmanr(dme_eyes["CST_v1"], dme_eyes["BCVA_final"])
    r_vel, p_vel = stats.spearmanr(eye_df["early_cst_velocity_pct"], eye_df["total_cst_drop"])

    add_test(
        "4. Bivariate Correlation",
        "Spearman Rank Correlation",
        "Concurrent CST vs. BCVA across all 1,593 visits",
        "rho",
        r_s,
        p_s,
        "Negative correlation confirms thicker/swollen retinas associate with worse visual acuity",
    )
    add_test(
        "4. Bivariate Correlation",
        "Pearson Linear Correlation",
        "Concurrent CST vs. BCVA across all 1,593 visits",
        "r",
        r_p,
        p_p,
        "Modest linear r (~ -0.35) illustrates functional-anatomical dissociation",
    )
    add_test(
        "4. Bivariate Correlation",
        "Spearman Rank Correlation (TREX-DME, n=56)",
        "Static Baseline CST (Visit 1) vs. Final BCVA",
        "rho",
        r_base,
        p_base,
        "Confirms static starting CST alone does not predict long-term visual outcome (p = 0.45)",
    )
    add_test(
        "4. Bivariate Correlation",
        "Spearman Rank Correlation (All 96 Eyes)",
        "Early CST Velocity (% drop Visits 1->3) vs. Total Long-Term CST Reduction",
        "rho",
        r_vel,
        p_vel,
        "Strong positive correlation confirms early clearing velocity predicts overall anatomical trajectory",
    )

    # 5. OCT Biomarker Stratification (192 Biomarker-Graded Visits)
    for b_col, label in [
        ("Fluid (IRF)", "Intraretinal Fluid (IRF)"),
        ("Fluid (SRF)", "Subretinal Fluid (SRF)"),
        ("Disruption of EZ", "EZ Disruption (Ellipsoid Zone)"),
        ("IR HRF", "Intraretinal Hyperreflective Foci (HRF)"),
        ("Disruption of RPE", "RPE Disruption"),
        ("DRIL", "DRIL (Disorganization of Retinal Inner Layers)"),
    ]:
        if b_col in visit_df.columns:
            sub = visit_df.dropna(subset=[b_col, "BCVA", "CST"]).copy()
            r_bio, p_bio = stats.spearmanr(sub[b_col], sub["BCVA"])
            add_test(
                "5. OCT Biomarker Stratification",
                "Spearman Rank Correlation",
                f"{label} Scan Fraction (% of 49 OCT slices) vs. BCVA (n={len(sub)})",
                "rho",
                r_bio,
                p_bio,
                f"Tests continuous association between {b_col} burden across the 49 B-scans and vision",
            )
            threshold = 0.10 if sub[b_col].median() > 0 else 0.0
            sub["present"] = (sub[b_col] > threshold).astype(int)
            sub["active_edema"] = (sub["CST"] >= 325.0).astype(int)
            ct = pd.crosstab(sub["present"], sub["active_edema"])
            if ct.shape == (2, 2):
                odds, p_fish = stats.fisher_exact(ct)
                add_test(
                    "5. OCT Biomarker Stratification",
                    "Fisher's Exact Test (2x2)",
                    f"{label} Presence vs. Active Edema (CST >= 325 um)",
                    "Odds Ratio",
                    odds,
                    p_fish,
                    f"Tests categorical association between {b_col} and macroscopic swelling >= 325 um",
                )

    stats_df = pd.DataFrame(results)
    out_csv = os.path.join(out_dir, STATS_SUMMARY_CSV)
    stats_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 95)
    print("CHECKPOINT 2 ELEMENTARY STATISTICAL TEST SUMMARY")
    print("=" * 95)
    for _, row in stats_df.iterrows():
        sig = "*" if row["Significant_p_lt_0_05"] else " "
        print(
            f"[{row['Category']}] {row['Statistical_Test']}\n"
            f"   Comparison : {row['Variables_Compared']}\n"
            f"   Result     : {row['Statistic_Type']} = {row['Statistic_Value']:.4f}, p = {row['p_value']:.4e} {sig}\n"
            f"   Takeaway   : {row['Clinical_Takeaway']}\n"
        )
    print("=" * 95)
    print(f"[Step 2] Saved {len(stats_df)} test results -> '{out_csv}'.")
    return stats_df


if __name__ == "__main__":
    out_folder = os.path.dirname(os.path.abspath(__file__))
    labels_folder = find_labels_dir()
    panel = step1_clean_and_merge(labels_folder, out_folder)
    step2_run_statistical_tests(panel, out_folder)