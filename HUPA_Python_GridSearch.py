# -*- coding: utf-8 -*-
"""
HUPA_Python_GridSearch.py

Purpose:
    Performs a Grid Search with Cross-Validation (CV) to optimize binary classification models
    (Normal vs. Pathological) using acoustic features.
    
    This version runs the full pipeline on TWO datasets:
        - ./data/HUPA_voice_features_PRN_CPP_50kHz.csv
        - ./data/HUPA_voice_features_PRN_CPP_44_1kHz.csv

    For each dataset:
    1. Loads features from CSV.
    2. Splits data into Train (80%) and Hold-out Test (20%).
    3. Defines feature groups (Noise, Perturbation, Tremor, Complexity).
    4. For each group, runs a GridSearchCV on 5 classifiers:
       - Logistic Regression
       - SVM (RBF)
       - Random Forest
       - k-NN
       - MLP (Neural Network)
    5. Evaluates the best models on the Test set and plots ROC curves.

Usage:
    Ensure the input CSVs are located at:
        ./data/HUPA_voice_features_PRN_CPP_50kHz.csv
        ./data/HUPA_voice_features_PRN_CPP_44_1kHz.csv

    Run via terminal:
        python HUPA_Python_GridSearch.py

Requirements:
    pandas, numpy, matplotlib, scikit-learn
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Suppress convergence warnings to keep the console clean during GridSearch
warnings.filterwarnings("ignore")


def main():
    # =========================================================================
    # 1. SETUP PATHS & DATASETS
    # =========================================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    # Two feature files: 50 kHz and 44.1 kHz
    csv_files = [
        "HUPA_voice_features_PRN_CPP_50kHz.csv",
        "HUPA_voice_features_PRN_CPP_44_1kHz.csv",
    ]
    fs_labels = ["50 kHz", "44.1 kHz"]
    fs_suffixes = ["50kHz", "44_1kHz"]  # for output filenames

    # CV scheme for all datasets
    cv_scheme = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # =========================================================================
    # 2. DEFINE MODEL PIPELINES & HYPERPARAMETER GRIDS (common to all datasets)
    # =========================================================================
    pipelines_and_grids = {
        # --- Logistic Regression ---
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

        # --- SVM (RBF Kernel) ---
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

        # --- Random Forest ---
        # Imputation is still done; scaling is not necessary for trees.
        "rf": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
            ]),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_leaf": [1, 2],
                "clf__max_features": ["sqrt", "log2"],
            },
        ),

        # --- k-Nearest Neighbors ---
        "knn": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            {
                "clf__n_neighbors": [3, 5, 7, 11, 15],
                "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "manhattan"],
            },
        ),

        # --- MLP (Neural Network) ---
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
        "knn": "k-NN",
        "rf": "Random Forest",
        "mlp": "Neural Network",
    }

    model_order = ["logreg", "svc_rbf", "rf", "knn", "mlp"]

    # =========================================================================
    # 3. LOOP OVER DATASETS (50 kHz / 44.1 kHz)
    # =========================================================================
    for csv_file, fs_label, fs_suffix in zip(csv_files, fs_labels, fs_suffixes):
        input_csv_path = os.path.join(data_dir, csv_file)

        print("\n" + "=" * 70)
        print(f"Running Grid Search for dataset: {csv_file}  (Sampling rate: {fs_label})")
        print("=" * 70)

        if not os.path.exists(input_csv_path):
            print(f"WARNING: Input file not found at: {input_csv_path}")
            print("Skipping this dataset.\n")
            continue

        print(f"Loading data from: {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        # Validate target column
        if "Label" not in df.columns:
            raise ValueError(
                f"Column 'Label' not found in CSV: {input_csv_path}. "
                "It must exist and contain binary labels (0/1)."
            )

        # Ensure labels are integers
        y = df["Label"].astype(int)

        # ---------------------------------------------------------------------
        # 3.1 Define feature groups for THIS dataset (columns may differ slightly)
        # ---------------------------------------------------------------------
        def existing(cols):
            """Return only the columns that exist in the current DataFrame."""
            return [c for c in cols if c in df.columns]

        # --- Group 1: Noise Parameters ---
        noise_features = existing([
            "HNR_mean", "HNR_std",   # Harmonics-to-Noise Ratio
            "CHNR_mean", "CHNR_std",  # Cepstral HNR
            "GNE_mean", "GNE_std",   # Glottal to Noise Excitation Ratio
            "NNE_mean", "NNE_std",   # Normalized Noise Energy
        ])

        # --- Group 2: Perturbation Measures ---
        # rShim vs rShimmer: we include both names; existing() will pick whichever is present.
        perturbation_features = existing([
            "CPP",        # Cepstral Peak Prominence (Covarep)
            "rShdB",      # Shimmer (dB)
            "rShim",      # Relative Shimmer (old name)
            "rShimmer",   # Relative Shimmer (alternative name)
            "rAPQ",       # Amplitude Perturbation Quotient
            "rSAPQ",      # Smoothed APQ
            "rJitta",     # Absolute Jitter
            "rJitt",      # Relative Jitter (%)
            "rRrRAP",     # Relative Average Perturbation
            "rPPQ",       # Pitch Period Perturbation Quotient
            "rSPPQ",      # Smoothed PPQ
        ])

        # --- Group 3: Tremor Parameters ---
        tremor_features = existing([
            "rFTRI",      # Frequency Tremor Intensity Index
            "rATRI",      # Amplitude Tremor Intensity Index
            "rFftr",      # Frequency Tremor Frequency
            "rFatr",      # Amplitude Tremor Frequency
        ])

        # --- Group 4: Complexity / Nonlinear Features ---
        complexity_features = existing([
            # Entropy measures
            "rApEn_mean", "rApEn_std",
            "rSampEn_mean", "rSampEn_std",
            "rFuzzyEn_mean", "rFuzzyEn_std",
            "rGSampEn_mean", "rGSampEn_std",
            "rmSampEn_mean", "rmSampEn_std",

            # Fractal / Chaos dimensions
            "CorrDim_mean", "CorrDim_std",
            "LLE_mean", "LLE_std",       # Largest Lyapunov Exponent
            "Hurst_mean", "Hurst_std",   # Hurst Exponent
            "mDFA_mean", "mDFA_std",     # Detrended Fluctuation Analysis

            # Recurrence & Others
            "RPDE_mean", "RPDE_std",     # Recurrence Period Density Entropy
            "PE_mean", "PE_std",         # Permutation Entropy
            "MarkEnt_mean", "MarkEnt_std",  # Markov Entropy
        ])

        feature_groups = {
            "Noise": noise_features,
            "Perturbation": perturbation_features,
            "Tremor": tremor_features,
            "Complexity": complexity_features,
        }

        print("\nFeature groups for this dataset:")
        for gname, cols in feature_groups.items():
            print(f"  {gname:<12}: {len(cols)} features")

        # ---------------------------------------------------------------------
        # 3.2 Train/Test split (same split used by all feature groups)
        # ---------------------------------------------------------------------
        indices = np.arange(len(df))
        idx_train, idx_test = train_test_split(
            indices,
            test_size=0.20,
            stratify=y,
            random_state=42
        )

        print(f"\nData Split ({fs_label}): {len(idx_train)} Training samples, {len(idx_test)} Test samples.")

        # Containers for results (for this dataset only)
        results_by_group = {}
        roc_data_by_group = {}

        # =========================================================================
        # 4. MAIN LOOP: GRID SEARCH & EVALUATION PER FEATURE GROUP
        # =========================================================================
        for group_name, cols in feature_groups.items():
            if len(cols) == 0:
                print(f"\n[SKIP] Group '{group_name}' has no valid columns in this dataset.")
                continue

            print("\n" + "-" * 50)
            print(f"PROCESSING GROUP: {group_name} ({len(cols)} features) [{fs_label}]")
            print("-" * 50)

            # Subset data for this specific group
            X_group = df.loc[:, cols]

            X_train = X_group.iloc[idx_train]
            X_test = X_group.iloc[idx_test]
            y_train = y.iloc[idx_train]
            y_test = y.iloc[idx_test]

            group_summary = []
            group_roc_curves = {}

            for model_key in model_order:
                if model_key not in pipelines_and_grids:
                    continue

                pipe, grid = pipelines_and_grids[model_key]

                print(f" -> Tuning {pretty_names[model_key]}...")

                # Run Grid Search with Cross Validation
                gs = GridSearchCV(
                    estimator=pipe,
                    param_grid=grid,
                    scoring="roc_auc",
                    n_jobs=-1,  # Use all CPU cores
                    cv=cv_scheme,
                    refit=True,
                    verbose=0,
                )

                gs.fit(X_train, y_train)

                # --- Evaluation on HOLD-OUT TEST SET ---
                best_model = gs.best_estimator_

                # Try predict_proba first, fallback to decision_function if needed
                if hasattr(best_model, "predict_proba"):
                    y_prob = best_model.predict_proba(X_test)[:, 1]
                else:
                    scores = best_model.decision_function(X_test)
                    # Min-max scale to [0, 1] to get pseudo-probabilities
                    y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

                test_auc = roc_auc_score(y_test, y_prob)
                fpr, tpr, _ = roc_curve(y_test, y_prob)

                group_summary.append({
                    "Group": group_name,
                    "Model": pretty_names[model_key],
                    "CV_AUC_Mean": gs.best_score_,
                    "Test_AUC": test_auc,
                    "Best_Params": str(gs.best_params_),
                })

                group_roc_curves[model_key] = (fpr, tpr, test_auc)

                print(f"    Best CV AUC: {gs.best_score_:.3f} | Test AUC: {test_auc:.3f}")

            if group_summary:
                results_df = pd.DataFrame(group_summary)
                results_df = results_df.sort_values(by="Test_AUC", ascending=False)
                results_by_group[group_name] = results_df
                roc_data_by_group[group_name] = group_roc_curves

        # =========================================================================
        # 5. VISUALIZATION (ROC CURVES) FOR THIS DATASET
        # =========================================================================
        groups_to_plot = ["Noise", "Perturbation", "Tremor", "Complexity"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, gname in enumerate(groups_to_plot):
            ax = axes[i]
        
            if gname not in roc_data_by_group:
                ax.set_visible(False)
                continue
        
            roc_curves = roc_data_by_group[gname]
        
            for model_key in model_order:
                if model_key in roc_curves:
                    fpr, tpr, auc = roc_curves[model_key]
                    label_text = f"{pretty_names[model_key]} (AUC={auc:.2f})"
                    ax.plot(fpr, tpr, lw=2, label=label_text)
        
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{gname} Features ({fs_label})")
            ax.legend(loc="lower right", frameon=False, fontsize=9)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"HUPA ROC Curves – Sampling rate: {fs_label}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # ==== SAVE FIG ====
        fig_dir = os.path.join(data_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        
        tool_suffix = "Python"  
        file_base = f"ROC_HUPA_{fs_suffix}_{tool_suffix}"
        # Example:
        #   ROC_HUPA_50kHz_Python
        #   ROC_HUPA_44_1kHz_Python
        
        fig_path_png = os.path.join(fig_dir, file_base + ".png")
        fig_path_pdf = os.path.join(fig_dir, file_base + ".pdf")
        
        fig.savefig(fig_path_png, dpi=300, bbox_inches="tight")
        fig.savefig(fig_path_pdf, dpi=300, bbox_inches="tight")
        
        print(f"\nROC figure for {fs_label} saved to:")
        print(f"  {fig_path_png}")
        print(f"  {fig_path_pdf}")
        
        plt.show()

        # =========================================================================
        # 6. SAVE RESULTS TO CSV (THIS DATASET ONLY)
        # =========================================================================
        if results_by_group:
            final_summary = pd.concat(results_by_group.values(), ignore_index=True)

            cols_order = ["Group", "Model", "Test_AUC", "CV_AUC_Mean", "Best_Params"]
            final_summary = final_summary[cols_order]

            output_csv_path = os.path.join(
                data_dir,
                f"HUPA_Python_Results_Summary_{fs_suffix}.csv",
            )

            final_summary.to_csv(output_csv_path, index=False)
            print(f"\n[DONE] Summary results for {fs_label} saved to: {output_csv_path}")

            print("\n=== TOP MODEL PER GROUP ({} dataset) ===".format(fs_label))
            for gname, rdf in results_by_group.items():
                best_row = rdf.iloc[0]
                print(
                    f"{gname}: {best_row['Model']} "
                    f"(Test AUC={best_row['Test_AUC']:.3f}, "
                    f"CV AUC={best_row['CV_AUC_Mean']:.3f})"
                )
        else:
            print(f"\n[WARNING] No valid results to summarize for dataset: {fs_label}")


if __name__ == "__main__":
    main()