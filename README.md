# Myocardial Infarction Analysis

UC Davis STA 141C (Statistical Computing) — Spring Quarter 2024 Final Project

## Overview

This project analyzes the [UCI Myocardial Infarction Complications dataset](https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications) (dataset ID 579), which contains clinical data from 1,700 heart attack patients with 112 features. The goal is to preprocess the data, handle missing values, and perform correlation analysis as a foundation for imputation and classification of patient outcomes (target variable: `LET_IS`).

## Dataset

- **Source**: UCI ML Repository (`ucimlrepo`, dataset ID 579)
- **Size**: 1,700 patients × 112 features
- **Target**: `LET_IS` — lethal outcome / in-hospital complication category
- **Feature types**: Binary (74), Categorical (22), Continuous (9)

## Project Structure

```
myocardial-infarction-analysis/
├── final_project.ipynb              # Main analysis notebook
├── utils.py                         # Utility functions for correlation analysis
├── myocardial-infarction-analysis.pdf  # Written report
└── presentation copy.pdf            # Slide deck
```

## Methodology

### 1. Data Cleaning
- Dropped columns with more than 25% missing values (112 → 105 columns)
- Dropped rows missing more than 25% of their values (1,700 → 1,624 rows)

### 2. Data Type Classification
Features are classified into three types to apply appropriate correlation measures:
- **Binary**: 2 unique values — e.g., `SEX`, ECG rhythm flags, complication indicators
- **Categorical**: 3–10 unique values — e.g., `INF_ANAM`, `FK_STENOK`, infarction location
- **Continuous**: >10 unique values — e.g., `AGE`, blood pressure (`S_AD_ORIT`, `D_AD_ORIT`), lab values

### 3. Correlation Analysis (for Imputation)
Type-appropriate correlation metrics were computed to identify the strongest predictors for each feature (used to guide missing value imputation):
- **Binary–Binary**: Phi coefficient matrix
- **Categorical–Categorical**: Cramér's V matrix
- **Continuous–Continuous**: Spearman correlation matrix

Top 5 most correlated predictors were extracted for each variable.

### 4. Bootstrap Resampling
Each subset (binary, categorical, continuous) was oversampled at 1.25× with replacement to support downstream modeling.

## Utility Functions (`utils.py`)

| Function | Description |
|---|---|
| `data_type_classifier(df)` | Classifies each column as Binary, Categorical, or Continuous |
| `compute_phi_matrix(df)` | Pearson/Phi coefficient matrix for binary variables |
| `compute_cramers_v_matrix(df)` | Cramér's V correlation matrix for categorical variables |
| `top_correlated_predictors_df(corr_matrix)` | Returns top N correlated predictors per variable |
| `top_correlated_categorical_all(df)` | Top correlated categorical pairs using Cramér's V |
| `top_correlated_continuous_all(corr_matrix)` | Top correlated continuous pairs by absolute correlation |

## Setup

**Requirements**: Python 3.11+

```bash
pip install numpy pandas polars scipy ucimlrepo
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

