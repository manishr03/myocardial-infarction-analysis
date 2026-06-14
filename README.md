# Myocardial Infarction Analysis

UC Davis STA 141C (Statistical Computing) — Spring Quarter 2024 Final Project

## Overview

This project analyzes the [UCI Myocardial Infarction Complications dataset](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications) (dataset ID 579), which contains clinical data from 1,700 heart attack patients with 112 features. The full pipeline covers data cleaning, missing value imputation, correlation analysis, dimensionality reduction with FAMD, and classification of patient outcomes (target variable: `LET_IS`) using XGBoost.

## Dataset

- **Source**: UCI ML Repository (`ucimlrepo`, dataset ID 579)
- **Size**: 1,700 patients × 112 features
- **Target**: `LET_IS` — lethal outcome / in-hospital complication (binarized: 0 = survived, 1 = complication/death)
- **Feature types**: Binary (74), Categorical (22), Continuous (9)

## Project Structure

```
myocardial-infarction-analysis/
├── final_project.ipynb                 # Main analysis notebook
├── utils.py                            # Utility functions for correlation analysis
├── myocardial-infarction-analysis.pdf  # Written report
└── presentation copy.pdf               # Slide deck
```

## Methodology

### 1. Data Cleaning
- Visualized missing values with a heatmap
- Dropped columns with more than 25% missing values (112 → 105 columns)
- Dropped rows missing more than 25% of their values (1,700 → 1,624 rows)

### 2. Data Type Identification
Features are classified into three types to apply appropriate imputation and correlation measures:
- **Binary**: 2 unique values — e.g., `SEX`, ECG rhythm flags, complication indicators
- **Categorical**: 3–10 unique values — e.g., `INF_ANAM`, `FK_STENOK`, infarction location
- **Continuous**: >10 unique values — e.g., `AGE`, blood pressure (`S_AD_ORIT`, `D_AD_ORIT`), lab values

### 3. Imputation
Two imputation strategies were compared:

**KNN Imputation** (`fancyimpute.KNN`, k=4) — applied to the full dataset at once. Fast and straightforward.

**Iterative Imputer + Random Forest** (`sklearn.impute.IterativeImputer`) — separate pipelines for each data type:
- Binary columns: `RandomForestClassifier`, predictions rounded to 0/1
- Categorical columns: `RandomForestClassifier`, predictions mapped back to original category values
- Continuous columns: `RandomForestRegressor`

Parallelized with `joblib`. About 75% of imputed values matched between the two methods.

### 4. Data Visualization
Bar plots generated for all binary and categorical features to inspect class distributions.

### 5. Correlation Analysis
Type-appropriate correlation matrices were computed and visualized as heatmaps:
- **Binary–Binary**: Phi coefficient matrix (`coolwarm` heatmap)
- **Categorical–Categorical**: Cramér's V matrix (`Reds` heatmap)
- **Continuous–Continuous**: Pearson correlation matrix (`coolwarm` heatmap)

Top 5 most correlated predictors were extracted for each variable.

### 6. Dimensionality Reduction (FAMD)
Factor Analysis of Mixed Data (`prince.FAMD`) was used to reduce the mixed-type feature space:
- Continuous features were standardized with `StandardScaler` before fitting
- 3D scatter plot of the first 3 components, colored by binarized target
- Cumulative explained variance plotted against number of components (1–50)
- Final model reduced to **10 components** for downstream classification

### 7. Classification (XGBoost)
An `XGBClassifier` was fit on the 10 FAMD components:
- **Hyperparameter tuning**: `GridSearchCV` with 10-fold CV over `n_estimators`, `max_depth`, and `learning_rate`
- **Evaluation**: 10-fold `StratifiedKFold` CV with the optimal estimator; aggregated confusion matrix and classification report
- **Feature importance**: FAMD component importances plotted from the optimal model

## Utility Functions (`utils.py`)

| Function | Description |
|---|---|
| `data_type_classifier(df)` | Classifies each column as Binary, Categorical, or Continuous |
| `compute_phi_matrix(df)` | Phi coefficient matrix for binary variables |
| `compute_cramers_v_matrix(df)` | Cramér's V correlation matrix for categorical variables |
| `top_correlated_predictors_df(corr_matrix)` | Returns top N correlated predictors per variable |
| `top_correlated_categorical_all(df)` | Top correlated categorical pairs using Cramér's V |
| `top_correlated_continuous_all(corr_matrix)` | Top correlated continuous pairs by absolute correlation |

## Setup

**Requirements**: Python 3.11+

```bash
pip install numpy pandas scipy seaborn matplotlib scikit-learn \
            fancyimpute joblib prince xgboost ucimlrepo
```

**Run the notebook**:

```bash
jupyter notebook final_project.ipynb
```

## Team

| Member | Columns (0-indexed) |
|---|---|
| Manish | 0–27 |
| Soto | 28–55 |
| Owen | 56–83 |
| Anna | 84–111 |
