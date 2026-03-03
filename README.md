# HUPA Voice Disorders Dataset

This repository contains MATLAB and Python scripts for feature extraction and classification of sustained vowel voice signals (pathological vs. healthy) using the HUPA database. The focus is on Perturbation, Regularity, Noise (PRN), and Complexity features.

---

## HUPA Database

To run the scripts, you need the **HUPA database**.

> https://zenodo.org/uploads/17704572

After downloading and organising the data, the expected structure inside this repository is:

```text
HUPA-Voice-Analysis/
├── toolboxes/
│   ├── AVCA-ByO-master/
│   ├── covarep-master/
│   ├── fastdfa/
│   ├── hctsa-main/
│   ├── hurst estimators/
│   ├── ME-master/
│   └── rpde/
├── data/
│   ├── HUPA_db/
│   │   ├── healthy/
│   │   │   ├── 50 kHz/      ← mono .wav files at / resampled to 50 kHz
│   │   │   └── 25 kHz/      ← mono .wav files resampled to 25 kHz
│   │   ├── pathological/
│   │   │   ├── 50 kHz/      ← mono .wav files at / resampled to 50 kHz
│   │   │   └── 25 kHz/      ← mono .wav files resampled to 25 kHz
│   │   ├── HUPA_db.xlsx
│   │   └── README.md
│   ├── HUPA_voice_features_PRN_CPP_25kHz.csv
│   ├── HUPA_voice_features_PRN_CPP_50kHz.csv
│   ├── HUPA_Python_Results_Summary_25kHz.csv
│   ├── HUPA_Python_Results_Summary_50kHz.csv
│   ├── reviewer_analysis/
│   │   ├── ReviewerAnalysis_25kHz_<Group>.csv
│   │   └── ReviewerAnalysis_50kHz_<Group>.csv
│   ├── subtype_error_audit/          ← subtype audit on the held-out test split
│   │   └── SubtypeAudit_<fs>_<Group>_<ModelKey>.csv
│   └── subtype_error_audit_oof/      ← subtype audit using OOF predictions across the full dataset
│       └── SubtypeAudit_OOF_<fs>_<Group>_<ModelKey>.csv
├── figures/
│   ├── ROC_HUPA_25kHz_Python.png
│   ├── ROC_HUPA_25kHz_Python.pdf
│   ├── ROC_HUPA_50kHz_Python.png
│   ├── ROC_HUPA_50kHz_Python.pdf
│   ├── ROC_HUPA_25kHz_MATLAB.png
│   ├── ROC_HUPA_25kHz_MATLAB.pdf
│   ├── ROC_HUPA_50kHz_MATLAB.png
│   ├── ROC_HUPA_50kHz_MATLAB.pdf
│   └── confusion_matrices/
│       └── CM_<fs>_<Group>_<ModelKey>.png
├── HUPA_Features_Extraction.m
├── HUPA_PRN_GridSearch_ROC.m
├── HUPA_Python_GridSearch.py
├── requirements.txt
└── README.md
```

* The **healthy/** folder contains recordings from healthy speakers.
* The **pathological/** folder contains recordings from patients with different laryngeal pathologies.
* Each condition is available at **50 kHz** and **25 kHz** (all files are mono).
* Inside `data/HUPA_db/` there is a spreadsheet `HUPA_db.xlsx` describing all speakers and recordings (age, sex, GRBAS scores, pathology codes, etc.), together with a local `README.md` in the same folder that documents the database structure and metadata fields.

**New (metadata used by the scripts).**  
The file `HUPA_db.xlsx` is also used to:
1) add metadata columns to the exported feature CSVs (e.g., `Sex` and `Pathology code`), and  
2) map `Pathology code` values to human-readable pathology names using the worksheet **`Pathology classification`**.

---

## MATLAB Workflow

### 1. Feature Extraction (`HUPA_Features_Extraction.m`)

This script:

1. Loads `.wav` files from:

   * `data/HUPA_db/healthy/50 kHz/`
   * `data/HUPA_db/pathological/50 kHz/`
   * `data/HUPA_db/healthy/25 kHz/`
   * `data/HUPA_db/pathological/25 kHz/`

2. Extracts:

   * AVCA PRN features (Perturbation, Regularity, Noise)
   * Nonlinear/complexity features (depending on AVCA configuration)
   * CPP (Cepstral Peak Prominence) using Covarep

3. Saves **two CSV files**, one per sampling frequency, in the `data/` folder:

   * `HUPA_voice_features_PRN_CPP_50kHz.csv`
   * `HUPA_voice_features_PRN_CPP_25kHz.csv`

Each CSV includes:

* One row per audio file
* Columns:

  * All AVCA PRN (and complexity) features
  * `CPP`
  * `FileName`
  * `Label` (0 = healthy, 1 = pathological)

**New (metadata columns for reproducible modelling).**  
The exported feature CSVs also include:
* `Sex` (string)
* `Pathology code` (integer; 0 for healthy, >0 for pathological subtypes)

These columns are taken from `data/HUPA_db/HUPA_db.xlsx` and are intended to support stratified analyses and subtype-level audits.

---

### 2. Classification & ROC Analysis (`HUPA_PRN_GridSearch_ROC.m`)

For each CSV:

1. Loads `HUPA_voice_features_PRN_CPP_50kHz.csv` or `HUPA_voice_features_PRN_CPP_25kHz.csv`.
2. Defines feature sets:

   * Noise
   * Perturbation (including CPP and jitter/shimmer)
   * Tremor
   * Complexity / nonlinear measures
   * **All features** (union of all feature blocks)

3. Cleans the data:

   * Removes all-NaN / constant columns
   * Imputes remaining NaNs (median)

4. Splits the data:

   * 80% Train (for hyperparameter optimisation via 5-fold CV)
   * 20% independent Test set

5. Trains and tunes:

   * Logistic Regression (`fitclinear`)
   * SVM (RBF) (`fitcsvm` + `fitPosterior`)
   * Random Forest (`TreeBagger`)
   * MLP (`fitcnet`, if available)

6. Evaluates models on the Test set and computes AUC.

7. **Plots ROC curves organised by classifier (2×2 subplots).**  
   Each subplot corresponds to one classifier, and each ROC curve within a subplot corresponds to one feature set (Noise, Perturbation, Tremor, Complexity, All).

**Additional outputs (reviewer-oriented analysis).**
Using thresholds selected from out-of-fold (OOF) predictions via Youden’s J, the script also:
* saves test-set confusion matrices to `figures/confusion_matrices/`,
* reports sex-stratified AUC (test and OOF when `Sex` is available),
* writes subtype-level false negative audits:
  * on the held-out test set (`data/subtype_error_audit/`),
  * and using OOF predictions across the full dataset (`data/subtype_error_audit_oof/`).

The script saves **one ROC figure per sampling rate**:

* `figures/ROC_HUPA_50kHz_MATLAB.png` and `.pdf`
* `figures/ROC_HUPA_25kHz_MATLAB.png` and `.pdf`

---

## Python Workflow (`HUPA_Python_GridSearch.py`)

A Python implementation using `scikit-learn` reproduces the MATLAB analysis for both sampling frequencies.

### Inputs

The script expects the two CSVs generated by MATLAB:

* `data/HUPA_voice_features_PRN_CPP_50kHz.csv`
* `data/HUPA_voice_features_PRN_CPP_25kHz.csv`

Each CSV may include `Sex` and `Pathology code`. If present, the script will run the reviewer-oriented analyses described below.

### Steps

For each sampling frequency (50 kHz, 25 kHz):

1. Loads the corresponding CSV.

2. Defines feature sets:

   * Noise, Perturbation, Tremor, Complexity, and **All features**.

3. Uses a common train–test split:

   * 80% Train, 20% Test, stratified by label.

4. For each feature set, runs a `GridSearchCV` with 5-fold CV and AUC as the scoring metric, over:

   * Logistic Regression
   * SVM (RBF)
   * Random Forest
   * MLP

   Each model is wrapped in a `Pipeline` with:

   * `SimpleImputer(strategy="median")`
   * `StandardScaler` (except Random Forest, which only uses imputation)

5. Evaluates the best model (per algorithm) on the hold-out Test set.

6. **Plots ROC curves organised by classifier (2×2 subplots).**  
   Each subplot corresponds to one classifier, and ROC curves within a subplot correspond to feature sets (Noise, Perturbation, Tremor, Complexity, All). Figures are saved to `figures/`:

   * `figures/ROC_HUPA_50kHz_Python.png` and `.pdf`
   * `figures/ROC_HUPA_25kHz_Python.png` and `.pdf`

7. Saves a summary CSV with all models and feature sets:

   * `data/HUPA_Python_Results_Summary_50kHz.csv`
   * `data/HUPA_Python_Results_Summary_25kHz.csv`

Each summary file contains, for every combination of feature set and model:

* `Group`
* `Model`
* `Test_AUC`
* `CV_AUC_Mean`
* `Best_Params`

**Additional outputs (reviewer-oriented analysis).**  
For each feature set, the script also creates:

* `data/reviewer_analysis/ReviewerAnalysis_<fs>_<Group>.csv` with:
  * test AUC and CV AUC,
  * sex-stratified AUC on test (`AUC_by_Sex_Test`) when `Sex` is available,
  * sex-stratified AUC using OOF predictions (`AUC_by_Sex_OOF`) when `Sex` is available,
  * Youden threshold selected from OOF predictions,
  * test confusion-matrix metrics (Sensitivity, Specificity, BalancedAcc),
  * paths to the confusion-matrix figure and subtype audit files.

* subtype-level audits:
  * `data/subtype_error_audit/SubtypeAudit_<fs>_<Group>_<ModelKey>.csv` (test split),
  * `data/subtype_error_audit_oof/SubtypeAudit_OOF_<fs>_<Group>_<ModelKey>.csv` (OOF full dataset).

---

## Requirements

### MATLAB

* MATLAB (R2020b or newer recommended)
* Statistics and Machine Learning Toolbox
* Deep Learning Toolbox (optional, for `fitcnet`)

### External Toolboxes

Place these libraries inside `toolboxes/`:

* **[AVCA-ByO](https://github.com/BYO-UPM/AVCA-ByO)**: Essential for P, R, N features.
* **[Covarep](https://github.com/covarep/covarep)**: Used for CPP feature extraction.
* **[Hurst Estimators](https://www.mathworks.com/matlabcentral/fileexchange/19148-hurst-parameter-estimate)**: Implementation to compute the Hurst exponent.
* **[RPDE](http://www.maxlittle.net/software/rpde.zip)**: Code to compute Recurrence Period Density Entropy (Little et al., 2007).
* **[FastDFA](http://www.maxlittle.net/software/fastdfa.zip)**: Implementation to compute Detrended Fluctuation Analysis (Little et al., 2006).
* **[HCTSA](https://github.com/benfulcher/hctsa)**: Highly Comparative Time-Series Analysis (used for D2 and LLE).
* **[ME (Markovian Entropies)](https://github.com/jdariasl/ME)**: Functions for the computation of entropies from Markov Models.

> **Compatibility Note for Newer MATLAB Versions**
>
> Many of these toolboxes were developed years ago. If you are using a recent version of MATLAB (e.g., R2020b+), please be aware of the following:
> * **Legacy Code:** You may need to manually update small parts of the external toolboxes to fix deprecated functions.
> * **Path Conflicts:** The script `HUPA_Features_Extraction.m` already handles a known conflict with Covarep (it removes `backcompatibility_2015` to avoid breaking the built-in `audioread`).
> * **Debugging:** If you encounter "function not found" or "input argument" errors inside these toolboxes, check that their internal paths are correctly added and that they support your MATLAB version.

### Python

Install dependencies via:

```bash
pip install -r requirements.txt
```

**Optional (Windows stability).**  
If parallel jobs cause issues on Windows, set the number of jobs to 1 before running:

```bash
set HUPA_N_JOBS=1
python HUPA_Python_GridSearch.py
```

---

## Citation

[Add here the reference to the HUPA database and the related publication, once finalised.]
