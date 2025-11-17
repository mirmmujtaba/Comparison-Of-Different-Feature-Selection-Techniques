# Comparison of Feature Selection Techniques with GeFeS (Genetic Algorithm)

This project compares **multiple feature selection techniques** for both **classification** and **regression** under a **fair, controlled setup**:

- A **Genetic Algorithm–based wrapper method (GeFeS)** first decides how many features are “optimal”.
- That same number of features **k** is then enforced for:
  - `SelectKBest` (ANOVA / f_regression),
  - `RFE` (Recursive Feature Elimination with Random Forest),
  - `SelectKBest` with Mutual Information (`mutual_info_classif` / `mutual_info_regression`).

This allows us to answer:

> “Given the same feature budget **k**, which feature selection technique gives the best predictive performance?”

All experiments are implemented in a single, self-contained notebook.

---

## 🔍 Methods Implemented

### 1. GeFeS – Genetic Algorithm Feature Selection

`GeFeS` is a **wrapper-based feature selection** class that uses a **Genetic Algorithm** to search over subsets of features:

- Works for **classification** and **regression**.
- Uses **cross-validation** with a chosen scoring metric (`accuracy`, `f1`, `roc_auc`, `r2`, `neg_mean_squared_error`, etc.).
- Supports:
  - Binary chromosome representation (select / drop each feature),
  - Tournament selection,
  - Single-point crossover,
  - Bit-flip mutation,
  - Elitism,
  - Fitness history tracking.

GeFeS returns:

- `best_features_` (indices),
- `best_score_` (best CV score),
- `get_feature_names()` (names of selected features),
- `get_support()` (mask / indices),
- `get_feature_importance_dict()` (name → index mapping).

### 2. Baseline Feature Selection Methods

All baselines are constrained to select the **same number of features k** determined by GeFeS:

- **SelectKBest (ANOVA / f_regression)**  
  - Classification: `f_classif`  
  - Regression: `f_regression`
- **RFE (Recursive Feature Elimination)**  
  - Estimator: `RandomForestClassifier` / `RandomForestRegressor`
- **Mutual Information**  
  - Classification: `mutual_info_classif` via `SelectKBest`  
  - Regression: `mutual_info_regression` via `SelectKBest`

---

## 📦 Datasets

### Classification – Covertype (Binary Subset)

- `sklearn.datasets.fetch_covtype`
- Original: 581,012 samples, 54 features, 7 classes.
- This project uses:
  - Only **classes 1 and 2**, remapped to a **binary** label:
    - Class 0: Spruce/Fir
    - Class 1: Lodgepole Pine
  - A **stratified subset of 1,000 samples** for speed.
- Features include:
  - Elevation, slope, distances to hydrology/roadways/fire points,
  - Hillshade measurements,
  - Wilderness areas,
  - Soil types.

### Regression – California Housing

- `sklearn.datasets.fetch_california_housing`
- 20,640 samples, 8 features.
- This project uses:
  - A subset of **1,000 samples** for speed.
- Features include:
  - Median income, house age, average rooms and bedrooms, population, occupancy, latitude, longitude.
- Target:
  - Median house value (in 100k USD).

---

## 🧪 Experimental Setup

### Classification Framework – `FeatureSelectionComparison`

- Estimator: `RandomForestClassifier(n_estimators=100, random_state=42)`
- CV: `RepeatedStratifiedKFold` with:
  - `cv_folds = 5`
  - `n_repeats = 10`
- Primary metric:
  - **Accuracy** (dataset is reasonably balanced).
- Additional metrics:
  - Precision, recall, F1, ROC–AUC.
- Steps:
  1. Run **GeFeS** → determine optimal k.
  2. Run SelectKBest, RFE, Mutual Information with that **same k**.
  3. Collect distributions of scores across repeated CV.
  4. Perform:
     - Pairwise **paired t-tests** (with Bonferroni correction),
     - **Friedman test** across all methods,
     - **Cohen’s d** effect sizes.
  5. Produce:
     - Summary tables,
     - Boxplots, bar charts,
     - Feature-overlap heatmap.

### Regression Framework – `RegressionFeatureSelectionComparison`

- Estimator: `RandomForestRegressor(n_estimators=100, random_state=42)`
- CV: `RepeatedKFold` with:
  - `cv_folds = 5`
  - `n_repeats = 5`
- Primary metric:
  - **R²**.
- Additional metrics:
  - `neg_mean_squared_error`, `neg_mean_absolute_error`, `neg_root_mean_squared_error`.
- Same workflow: GeFeS decides **k**, other methods are constrained to that k, then significance testing and plotting.

---

## 📊 Key Results (Summary)

### Classification (Covertype – Binary Task, k = 27 features)

Primary metric: **Accuracy** (mean ± std)

| Method       | Accuracy | Std    |
|-------------|----------|--------|
| **GeFeS**   | **0.7796** | 0.0236 |
| MutualInfo  | 0.7755   | 0.0253 |
| RFE         | 0.7705   | 0.0234 |
| SelectKBest | 0.7561   | 0.0249 |

- **GeFeS** achieved the best mean accuracy.
- GeFeS vs SelectKBest: **large** effect (Cohen’s d ≈ 0.97), statistically significant.
- GeFeS vs RFE: **small** effect, significant but modest.
- GeFeS vs MutualInfo: difference is **not statistically significant**; they are effectively tied.
- 5 features were selected by **all methods** (e.g., Elevation, Horizontal_Distance_To_Roadways, Soil_Type_10/12/39).

### Regression (California Housing, k = 6 features)

Primary metric: **R²** (mean ± std)

| Method       | R²      | Std    |
|-------------|---------|--------|
| **GeFeS**   | **0.7144** | 0.0379 |
| RFE         | 0.7073  | 0.0433 |
| MutualInfo  | 0.6833  | 0.0494 |
| SelectKBest | 0.6657  | 0.0490 |

- GeFeS chose 6 intuitive features: `MedInc`, `HouseAge`, `AveBedrms`, `AveOccup`, `Latitude`, `Longitude`.
- GeFeS vs SelectKBest: **large** effect (Cohen’s d ≈ 1.11), strongly significant.
- GeFeS vs MutualInfo: **medium** effect, significant.
- GeFeS vs RFE: **negligible** effect, **not significant** after Bonferroni → essentially tied.

**Overall:**  
GeFeS is **best or tied-best**, with clearly better performance than simple univariate methods (SelectKBest), and generally slightly better than Mutual Information. RFE with Random Forest is a very strong, simpler alternative that often comes very close to GeFeS.
