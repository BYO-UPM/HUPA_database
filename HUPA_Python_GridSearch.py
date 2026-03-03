# -*- coding: utf-8 -*-
"""
HUPA_Python_GridSearch.py

Purpose:
    Runs a GridSearchCV (5-fold CV) to optimize binary classification models
    (Healthy vs. Pathological) using acoustic features.

    This version keeps the same workflow as the original script (ROC/AUC per
    feature group and model), and adds the analyses requested by the reviewer:
      1) Test AUC stratified by Sex (if column "Sex" exists in the features CSV).
      2) Confusion matrix on the hold-out test set using a threshold selected
         on the TRAIN set (Youden's J, computed from out-of-fold probabilities).
      3) Error audit by pathology subtype (which pathology codes are often
         predicted as Healthy), with optional mapping from pathology code to
         pathology name using HUPA_db.xlsx (sheet "Pathology classification").

Inputs:
    The script expects two CSV feature files in ./data/ (relative to this script).
    By default, it tries the "_with_meta" files (recommended for reproducibility),
    and falls back to the old filenames if they do not exist:

        - HUPA_voice_features_PRN_CPP_50kHz_with_meta.csv  (preferred)
          fallback: HUPA_voice_features_PRN_CPP_50kHz.csv

        - HUPA_voice_features_PRN_CPP_25kHz_with_meta.csv  (preferred)
          fallback: HUPA_voice_features_PRN_CPP_25kHz.csv

    Required column:
        - Label (0=Healthy, 1=Pathological)

    Optional columns (added by the extraction script v2):
        - Sex (string)
        - Pathology code (int)

    Optional Excel for pathology code -> name mapping:
        - HUPA_db.xlsx (in ./data/ or next to this script)
          Sheet: "Pathology classification"
          Columns: "Code" (e.g., "1.2.1.2.2") and "Pathology" (name)

Outputs:
    - ROC figures (PNG + PDF) under ./figures/
    - Summary CSV under ./data/
    - Confusion matrices and subtype audit CSVs under ./figures/ and ./data/

Requirements:
    pandas, numpy, matplotlib, scikit-learn, openpyxl
"""

import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
# Force a non-interactive backend to avoid Tkinter/thread issues (Windows + joblib)
os.environ.setdefault('MPLBACKEND', 'Agg')
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

# Parallelism control (to avoid GUI/thread issues on Windows).
# You can override by setting environment variable HUPA_N_JOBS.
_DEFAULT_N_JOBS = "1" if os.name == "nt" else "-1"
N_JOBS = int(os.environ.get("HUPA_N_JOBS", _DEFAULT_N_JOBS))


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def pick_first_existing_column(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC is undefined if y_true has only one class."""
    y_unique = np.unique(y_true)
    if len(y_unique) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def load_pathology_map(excel_path: str) -> Dict[int, Dict[str, str]]:
    """
    Load pathology classification mapping from HUPA_db.xlsx (sheet 'Pathology classification').

    Returns:
        dict mapping compact_code_int -> {
            'dotted': original dotted code string, e.g., '1.2.1.2.2',
            'leaf':   leaf node name from the row,
            'full':   hierarchical path name built from dotted prefixes, joined with ' > '
        }

    The compact integer code is computed by removing dots and non-digits:
        '1.2.1.2.2' -> '12122' -> int(12122)
    """
    df_map = pd.read_excel(excel_path, sheet_name="Pathology classification")
    if "Code" not in df_map.columns or "Pathology" not in df_map.columns:
        raise ValueError("Expected columns 'Code' and 'Pathology' in 'Pathology classification' sheet.")

    dotted_to_name: Dict[str, str] = {}
    rows = []
    for _, row in df_map.iterrows():
        code_raw = str(row["Code"]).strip()
        name_raw = str(row["Pathology"]).strip()

        if code_raw.lower() in ("nan", "", "none"):
            continue

        dotted = "".join(ch for ch in code_raw if (ch.isdigit() or ch == ".")).strip(".")
        if dotted == "":
            continue

        dotted_to_name[dotted] = name_raw
        rows.append((dotted, name_raw))

    def build_full_name(dotted_code: str) -> str:
        parts = []
        toks = dotted_code.split(".")
        for i in range(1, len(toks) + 1):
            prefix = ".".join(toks[:i])
            nm = dotted_to_name.get(prefix, "").strip()
            if nm and (len(parts) == 0 or parts[-1] != nm):
                parts.append(nm)
        return " > ".join(parts) if parts else "NR"

    out: Dict[int, Dict[str, str]] = {}
    for dotted, leaf_name in rows:
        code_digits = "".join(ch for ch in dotted if ch.isdigit())
        if code_digits == "":
            continue
        try:
            code_int = int(code_digits)
        except ValueError:
            continue

        if code_int not in out:
            out[code_int] = {
                "dotted": dotted,
                "leaf": leaf_name,
                "full": build_full_name(dotted),
            }

    return out

def youden_threshold_from_scores(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick threshold that maximizes Youden's J = TPR - FPR."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thr[best_idx])

def youden_threshold_from_oof(best_model, X_train: pd.DataFrame, y_train: pd.Series, cv_scheme) -> float:
    """
    Compute out-of-fold probabilities on TRAIN and pick threshold that
    maximizes Youden's J = TPR - FPR.
    """
    oof_prob = cross_val_predict(
        best_model,
        X_train,
        y_train,
        cv=cv_scheme,
        method="predict_proba",
        n_jobs=N_JOBS
    )[:, 1]

    fpr, tpr, thr = roc_curve(y_train, oof_prob)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thr[best_idx])


def compute_confusion_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Tuple[np.ndarray, Dict[str, float]]:
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    bacc = 0.5 * (sens + spec)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)

    metrics = {
        "TN": float(tn),
        "FP": float(fp),
        "FN": float(fn),
        "TP": float(tp),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "BalancedAcc": float(bacc),
        "Accuracy": float(acc),
        "Threshold_Youden_OOF": float(thr),
    }
    return cm, metrics


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy", "Pathological"])
    ax.set_yticklabels(["Healthy", "Pathological"])

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
def main():
    # =========================================================================
    # 1. SETUP PATHS & DATASETS
    # =========================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    fig_dir = os.path.join(script_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Prefer corrected CSVs with metadata, fallback to older filenames
    csv_candidates = [
        ("HUPA_voice_features_PRN_CPP_50kHz_with_meta.csv", "HUPA_voice_features_PRN_CPP_50kHz.csv"),
        ("HUPA_voice_features_PRN_CPP_25kHz_with_meta.csv", "HUPA_voice_features_PRN_CPP_25kHz.csv"),
    ]
    fs_labels = ["50 kHz", "25 kHz"]
    fs_suffixes = ["50kHz", "25kHz"]

    # CV scheme for all datasets
    cv_scheme = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Optional pathology map
    pathology_map = {}
    excel_candidates = [
        os.path.join(data_dir, "HUPA_db.xlsx"),
        os.path.join(script_dir, "HUPA_db.xlsx"),
    ]
    for xp in excel_candidates:
        if os.path.exists(xp):
            try:
                pathology_map = load_pathology_map(xp)
                print(f"Loaded pathology map from: {xp} (n={len(pathology_map)})")
            except Exception as e:
                print(f"WARNING: Could not load pathology map from {xp}: {e}")
            break

    # =========================================================================
    # 2. DEFINE MODEL PIPELINES & HYPERPARAMETER GRIDS
    # =========================================================================
    pipelines_and_grids = {
        "logreg": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(solver="saga", max_iter=5000, random_state=42)),
            ]),
            {
                "clf__penalty": ["l2", "l1"],
                "clf__C": [0.01, 0.1, 1, 10, 30],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "svc_rbf": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, max_iter=50000, random_state=42)),
            ]),
            {
                "clf__C": [0.1, 1, 10, 30, 100],
                "clf__gamma": ["scale", 0.001, 0.01, 0.1],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "rf": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(random_state=42, n_jobs=(-1 if N_JOBS == -1 else max(1, N_JOBS)))),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 2],
                "clf__max_features": ["sqrt", "log2"],
            },
        ),
        "mlp": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(max_iter=2000, early_stopping=True, random_state=42)),
            ]),
            {
                "clf__hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "clf__alpha": [1e-3, 1e-2],
                "clf__learning_rate_init": [1e-3, 1e-2],
                "clf__activation": ["relu", "tanh"],
            },
        ),
    }

    pretty_names = {
        "logreg": "Logistic Regression",
        "svc_rbf": "SVM (RBF)",
        "rf": "Random Forest",
        "mlp": "MLP",
    }
    model_order = ["logreg", "svc_rbf", "rf", "mlp"]

    # =========================================================================
    # 3. LOOP OVER DATASETS
    # =========================================================================
    for (preferred_csv, fallback_csv), fs_label, fs_suffix in zip(csv_candidates, fs_labels, fs_suffixes):
        preferred_path = os.path.join(data_dir, preferred_csv)
        fallback_path = os.path.join(data_dir, fallback_csv)

        if os.path.exists(preferred_path):
            input_csv_path = preferred_path
        else:
            input_csv_path = fallback_path

        print("\n" + "=" * 70)
        print(f"Dataset: {os.path.basename(input_csv_path)}  (Sampling rate: {fs_label})")
        print("=" * 70)

        if not os.path.exists(input_csv_path):
            print(f"WARNING: Input file not found at: {input_csv_path}")
            print("Skipping this dataset.\n")
            continue

        df = pd.read_csv(input_csv_path)

        if "Label" not in df.columns:
            raise ValueError(f"Column 'Label' not found in CSV: {input_csv_path}")

        # Optional columns
        sex_col = pick_first_existing_column(df, ["Sex", "sex", "Gender", "gender"])
        path_code_col = pick_first_existing_column(df, ["Pathology code", "Pathology_code", "pathology code", "pathology_code"])

        if sex_col is not None:
            df[sex_col] = df[sex_col].astype(str)
        if path_code_col is not None:
            # Keep numeric if possible; but do not crash if some values are missing
            df[path_code_col] = pd.to_numeric(df[path_code_col], errors="coerce").fillna(-1).astype(int)

        y = df["Label"].astype(int)

        # ---------------------------------------------------------------------
        # 3.1 Define feature groups (same as original)
        # ---------------------------------------------------------------------
        def existing(cols):
            return [c for c in cols if c in df.columns]

        noise_features = existing([
            "HNR_mean", "HNR_std",
            "CHNR_mean", "CHNR_std",
            "GNE_mean", "GNE_std",
            "NNE_mean", "NNE_std",
        ])

        perturbation_features = existing([
            "CPP",
            "rShdB",
            "rShim",
            "rShimmer",
            "rAPQ",
            "rSAPQ",
            "rJitta",
            "rJitt",
            "rRrRAP",
            "rPPQ",
            "rSPPQ",
        ])

        tremor_features = existing([
            "rFTRI",
            "rATRI",
            "rFftr",
            "rFatr",
        ])

        complexity_features = existing([
            "rApEn_mean", "rApEn_std",
            "rSampEn_mean", "rSampEn_std",
            "rFuzzyEn_mean", "rFuzzyEn_std",
            "rGSampEn_mean", "rGSampEn_std",
            "rmSampEn_mean", "rmSampEn_std",
            "CorrDim_mean", "CorrDim_std",
            "LLE_mean", "LLE_std",
            "Hurst_mean", "Hurst_std",
            "mDFA_mean", "mDFA_std",
            "RPDE_mean", "RPDE_std",
            "PE_mean", "PE_std",
            "MarkEnt_mean", "MarkEnt_std",
        ])

        
        all_features = sorted(set(noise_features + perturbation_features + tremor_features + complexity_features))

        feature_groups = {
            "Noise": noise_features,
            "Perturbation": perturbation_features,
            "Tremor": tremor_features,
            "Complexity": complexity_features,
            "All": all_features,
        }

        print("\nFeature groups:")
        for gname, cols in feature_groups.items():
            print(f"  {gname:<12}: {len(cols)} features")

        # ---------------------------------------------------------------------
        # 3.2 Train/Test split (same split for all groups)
        # If Sex exists, we keep stratification only by Label (as original),
        # because Sex can be missing or unbalanced. Reviewer analysis is done
        # AFTER the split, on the test set.
        # ---------------------------------------------------------------------
        indices = np.arange(len(df))
        idx_train, idx_test = train_test_split(
            indices,
            test_size=0.20,
            stratify=y,
            random_state=42
        )
        print(f"\nSplit: {len(idx_train)} train / {len(idx_test)} test")

        # Containers for results and ROC data
        results_by_group = {}
        roc_data_by_group = {}

        # =========================================================================
        # 4. GRID SEARCH & EVALUATION PER GROUP
        # =========================================================================
        for group_name, cols in feature_groups.items():
            if len(cols) == 0:
                print(f"\n[SKIP] Group '{group_name}' has no valid columns in this dataset.")
                continue

            print("\n" + "-" * 55)
            print(f"GROUP: {group_name} ({len(cols)} features) [{fs_label}]")
            print("-" * 55)

            X_group = df.loc[:, cols]
            cols_all_nan = X_group.columns[X_group.isna().all()]
            X_group = X_group.drop(columns=cols_all_nan)
            X_group_full = X_group.copy()

            X_train = X_group.iloc[idx_train]
            X_test = X_group.iloc[idx_test]
            y_train = y.iloc[idx_train]
            y_test = y.iloc[idx_test]


            if sex_col is not None:
                sex_test = df.iloc[idx_test][sex_col].astype(str).values
                sex_all = df[sex_col].astype(str).values
            else:
                sex_test = None
                sex_all = None
            
            if path_code_col is not None:
                path_code_test = df.iloc[idx_test][path_code_col].astype(int).values
                path_code_all = df[path_code_col].astype(int).values
            else:
                path_code_test = None
                path_code_all = None

            group_summary = []
            group_roc_curves = {}

            # Store per-model evaluations so we can pick the best model per group
            eval_cache = {}

            for model_key in model_order:
                pipe, grid = pipelines_and_grids[model_key]
                print(f" -> Tuning {pretty_names[model_key]}...")

                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=grid,
                    scoring="roc_auc",
                    n_jobs=N_JOBS,
                    cv=cv_scheme,
                    refit=True,
                    verbose=0,
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_

                # Predict probabilities on TEST
                y_prob = best_model.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_prob)
                fpr, tpr, _ = roc_curve(y_test, y_prob)

                group_summary.append({
                    "Group": group_name,
                    "Model": pretty_names[model_key],
                    "CV_AUC_Mean": float(gs.best_score_),
                    "Test_AUC": float(test_auc),
                    "Best_Params": str(gs.best_params_),
                })
                group_roc_curves[model_key] = (fpr, tpr, test_auc)

                eval_cache[model_key] = {
                    "gs_best_score": float(gs.best_score_),
                    "best_params": str(gs.best_params_),
                    "best_model": best_model,
                    "y_prob": y_prob,
                    "y_test": y_test.to_numpy(),
                    "sex_test": sex_test,
                    "path_code_test": path_code_test,
                }

                print(f"    Best CV AUC: {gs.best_score_:.3f} | Test AUC: {test_auc:.3f}")

            # Pick best model by Test AUC (as you usually report on the hold-out)
            results_df = pd.DataFrame(group_summary).sort_values(by="Test_AUC", ascending=False)
            results_by_group[group_name] = results_df
            roc_data_by_group[group_name] = group_roc_curves

            best_model_key = None
            if not results_df.empty:
                # Recover key from pretty_names
                best_model_name = str(results_df.iloc[0]["Model"])
                for k, v in pretty_names.items():
                    if v == best_model_name:
                        best_model_key = k
                        break

            # -----------------------------------------------------------------
            # Reviewer analyses on the BEST model per group
            # -----------------------------------------------------------------
            if best_model_key is not None and best_model_key in eval_cache:
                pack = eval_cache[best_model_key]
                best_model = pack["best_model"]
                y_prob_best = pack["y_prob"]
                y_test_np = pack["y_test"]

                # Threshold selection on TRAIN using OOF probs (Youden)
                thr = youden_threshold_from_oof(best_model, X_train, y_train, cv_scheme)
                cm, cm_metrics = compute_confusion_metrics(y_test_np, y_prob_best, thr)

                cm_dir = os.path.join(fig_dir, "confusion_matrices")
                os.makedirs(cm_dir, exist_ok=True)
                cm_path = os.path.join(cm_dir, f"CM_{fs_suffix}_{group_name}_{best_model_key}.png")
                plot_confusion_matrix(cm, f"{group_name} – {fs_label} – {pretty_names[best_model_key]}", cm_path)

                
                # AUC by Sex (test)
                auc_by_sex = {}
                if pack["sex_test"] is not None:
                    for sx in np.unique(pack["sex_test"]):
                        msk = (pack["sex_test"] == sx)
                        auc_by_sex[str(sx)] = safe_auc(y_test_np[msk], y_prob_best[msk])

                # -----------------------------------------------------------------
                # OOF analyses across the FULL dataset (representative by subtype)
                # -----------------------------------------------------------------
                subtype_audit_path_oof = None
                auc_by_sex_oof = {}

                if (sex_all is not None) or (path_code_all is not None):
                    try:
                        audit_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        oof_prob_all = cross_val_predict(
                            best_model,
                            X_group_full,
                            y,
                            cv=audit_cv,
                            method="predict_proba",
                            n_jobs=N_JOBS,
                        )[:, 1]

                        # OOF AUC by Sex (full dataset)
                        if sex_all is not None:
                            for sx in np.unique(sex_all):
                                msk = (sex_all == sx)
                                auc_by_sex_oof[str(sx)] = safe_auc(y.values[msk], oof_prob_all[msk])

                        # Subtype audit (OOF, full dataset; only pathological samples)
                        if path_code_all is not None:
                            pat_mask_all = (y.values == 1) & (path_code_all >= 0)
                            thr_oof_all = youden_threshold_from_scores(y.values, oof_prob_all)
                            y_pred_oof_all = (oof_prob_all >= thr_oof_all).astype(int)
                            fn_mask_all = pat_mask_all & (y_pred_oof_all == 0)

                            rows = []
                            for code in np.unique(path_code_all[pat_mask_all]):
                                msk = pat_mask_all & (path_code_all == code)
                                n_pat = int(np.sum(msk))
                                n_fn = int(np.sum(fn_mask_all & (path_code_all == code)))
                                fn_rate = n_fn / (n_pat + 1e-12)

                                info = pathology_map.get(int(code), None) if pathology_map else None
                                dotted = info.get("dotted", "") if isinstance(info, dict) else ""
                                full_name = info.get("full", "NR") if isinstance(info, dict) else "NR"

                                rows.append({
                                    "Pathology code": int(code),
                                    "Code (dotted)": dotted,
                                    "Full pathology name": full_name,
                                    "N_pathological_total": n_pat,
                                    "N_false_negative": n_fn,
                                    "FN_rate": float(fn_rate),
                                })

                            audit_df = pd.DataFrame(rows).sort_values(
                                by=["N_false_negative", "FN_rate", "N_pathological_total", "Pathology code"],
                                ascending=[False, False, False, True]
                            )

                            audit_dir = os.path.join(data_dir, "subtype_error_audit_oof")
                            os.makedirs(audit_dir, exist_ok=True)
                            subtype_audit_path_oof = os.path.join(
                                audit_dir,
                                f"SubtypeAudit_OOF_{fs_suffix}_{group_name}_{best_model_key}.csv"
                            )
                            audit_df.to_csv(subtype_audit_path_oof, index=False)
                    except Exception as e:
                        print(f"    [WARNING] OOF audit failed: {e}")

                # Subtype error audit (only pathological in test)
                subtype_audit_path = None
                if pack["path_code_test"] is not None:
                    pat_mask = (y_test_np == 1)
                    y_pred_best = (y_prob_best >= thr).astype(int)
                    fn_mask = pat_mask & (y_pred_best == 0)

                    rows = []
                    for code in np.unique(pack["path_code_test"][pat_mask]):
                        m = pat_mask & (pack["path_code_test"] == code)
                        n_pat = int(np.sum(m))
                        n_fn = int(np.sum(fn_mask & (pack["path_code_test"] == code)))
                        fn_rate = n_fn / (n_pat + 1e-12)

                        info = pathology_map.get(int(code), None) if pathology_map else None
                        full_name = info.get("full", "NR") if isinstance(info, dict) else "NR"
                        rows.append({
                            "Pathology code": int(code),
                            "Full pathology name": full_name,
                            "N_pathological_test": n_pat,
                            "N_false_negative": n_fn,
                            "FN_rate": float(fn_rate),
                        })

                    audit_df = pd.DataFrame(rows).sort_values(["FN_rate", "N_pathological_test"], ascending=[False, False])
                    audit_dir = os.path.join(data_dir, "subtype_error_audit")
                    os.makedirs(audit_dir, exist_ok=True)
                    subtype_audit_path = os.path.join(audit_dir, f"SubtypeAudit_{fs_suffix}_{group_name}_{best_model_key}.csv")
                    audit_df.to_csv(subtype_audit_path, index=False)

                # Append these reviewer metrics to the best row in results_df (for convenience)
                # We will create a separate small CSV with reviewer analyses per group.
                reviewer_rows = []
                reviewer_rows.append({
                    "SamplingRate": fs_label,
                    "Group": group_name,
                    "BestModel": pretty_names[best_model_key],
                    "Test_AUC": float(results_df.iloc[0]["Test_AUC"]),
                    "CV_AUC_Mean": float(results_df.iloc[0]["CV_AUC_Mean"]),
                    "YoudenThr_OOF": float(cm_metrics["Threshold_Youden_OOF"]),
                    "Test_Sensitivity": float(cm_metrics["Sensitivity"]),
                    "Test_Specificity": float(cm_metrics["Specificity"]),
                    "Test_BalancedAcc": float(cm_metrics["BalancedAcc"]),
                    "ConfusionMatrixPath": cm_path,
                    "SubtypeAuditPath_Test": subtype_audit_path if subtype_audit_path else "",
                    "SubtypeAuditPath_OOF": subtype_audit_path_oof if subtype_audit_path_oof else "",
                    "AUC_by_Sex_Test": str(auc_by_sex) if auc_by_sex else "",
                    "AUC_by_Sex_OOF": str(auc_by_sex_oof) if auc_by_sex_oof else "",
                })

                reviewer_df = pd.DataFrame(reviewer_rows)
                reviewer_out = os.path.join(data_dir, f"HUPA_Python_ReviewerAnalysis_{fs_suffix}_{group_name}.csv")
                reviewer_df.to_csv(reviewer_out, index=False)
                print(f"    Reviewer analysis saved to: {reviewer_out}")

        
        # =========================================================================
        # 5. VISUALIZATION (ROC CURVES) FOR THIS DATASET
        #    4 subplots (one per MODEL). Each subplot overlays ROC curves for
        #    each FEATURE GROUP (Noise/Perturbation/Tremor/Complexity/All).
        # =========================================================================
        groups_to_plot = ["Noise", "Perturbation", "Tremor", "Complexity", "All"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
        axes = axes.ravel()
        for i, model_key in enumerate(model_order):
            ax = axes[i]

            # Plot one ROC curve per feature group for this model
            for gname in groups_to_plot:
                if gname in roc_data_by_group and model_key in roc_data_by_group[gname]:
                    fpr, tpr, auc = roc_data_by_group[gname][model_key]
                    ax.plot(fpr, tpr, lw=2, label=f"{gname} (AUC={auc:.2f})")

            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{pretty_names[model_key]} – {fs_label}")
            ax.legend(loc="lower right", frameon=False, fontsize=10)
            ax.grid(True, alpha=0.3)
            # Make each ROC subplot square (box aspect 1:1)
            try:
                ax.set_box_aspect(1)
            except Exception:
                pass
            ax.set_aspect('equal', adjustable='box')
        fig.suptitle(f"HUPA ROC Curves – Sampling rate: {fs_label}", fontsize=16)
        # tight_layout removed (using constrained_layout for square subplots)
        fig.subplots_adjust(top=0.92)

        tool_suffix = "Python"
        file_base = f"ROC_HUPA_{fs_suffix}_{tool_suffix}"
        fig_path_png = os.path.join(fig_dir, file_base + ".png")
        fig_path_pdf = os.path.join(fig_dir, file_base + ".pdf")

        fig.savefig(fig_path_png, dpi=300)
        fig.savefig(fig_path_pdf, dpi=300)
        print(f"\nROC figure saved to:\n  {fig_path_png}\n  {fig_path_pdf}")

        plt.close(fig)

# =========================================================================
        # 6. SAVE RESULTS TO CSV (THIS DATASET ONLY)
        # =========================================================================
        if results_by_group:
            final_summary = pd.concat(results_by_group.values(), ignore_index=True)
            cols_order = ["Group", "Model", "Test_AUC", "CV_AUC_Mean", "Best_Params"]
            final_summary = final_summary[cols_order]

            output_csv_path = os.path.join(data_dir, f"HUPA_Python_Results_Summary_{fs_suffix}.csv")
            final_summary.to_csv(output_csv_path, index=False)
            print(f"\n[DONE] Summary results saved to: {output_csv_path}")

            print(f"\n=== TOP MODEL PER GROUP ({fs_label}) ===")
            for gname, rdf in results_by_group.items():
                best_row = rdf.sort_values(by="Test_AUC", ascending=False).iloc[0]
                print(
                    f"{gname}: {best_row['Model']} "
                    f"(Test AUC={best_row['Test_AUC']:.3f}, CV AUC={best_row['CV_AUC_Mean']:.3f})"
                )
        else:
            print(f"\n[WARNING] No valid results to summarize for dataset: {fs_label}")


if __name__ == "__main__":
    main()
