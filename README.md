# MI Detection System — XGBoost with Temporal Biomarker Feature Engineering

## Overview
Comparative machine learning study for acute myocardial infarction (MI) detection
using laboratory biomarkers (troponin and CK-MB) with temporal feature engineering.

Published in: PeerJ (under review)
Authors: Hussam Aziz Ayed Alshammari, Abdullah Eyada Mutlaq Alshammari
Institution: University of Hail, College of Computer Science and Engineering

## Dataset
- 2,327 hospital visits from 1,506 unique patients
- Target: Acute MI based on ICD-10 codes I21, I22
- Positive rate: 57.5% at visit level
- De-identified: patient IDs replaced with cryptographic hashes

## Features (12 total)
| Feature | Type | Description |
|---|---|---|
| age | Numerical | Patient age (z-score normalized) |
| visit_type | Categorical | One-hot encoded |
| visit_index | Numerical | Sequential visit counter per patient |
| days_since_last_visit | Numerical | Days between visits |
| troponin | Numerical | Current level |
| troponin_prev | Numerical | Previous measurement |
| troponin_delta | Numerical | Change from previous |
| troponin_max_prior | Numerical | Maximum prior value |
| ckmb | Numerical | Current level |
| ckmb_prev | Numerical | Previous measurement |
| ckmb_delta | Numerical | Change from previous |
| ckmb_max_prior | Numerical | Maximum prior value |

## Models Compared
- Logistic Regression (L2 regularization)
- Random Forest (600 trees)
- Histogram Gradient Boosting (HGB)
- XGBoost (900 estimators) ← Best performer

## Results Summary
| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| XGBoost | 0.986 | 0.982 | 0.937 |
| HGB | 0.985 | 0.981 | 0.941 |
| RF | 0.981 | 0.980 | 0.945 |
| LogReg | 0.965 | 0.961 | 0.867 |

## Requirements
```
pip install -r requirements.txt
```
Key dependencies: Python 3.12+, xgboost, scikit-learn>=1.6.1, pandas, numpy, matplotlib

## Usage
```
jupyter notebook "MI_NestedCV_4Models_Target098_ProgressOutputs.ipynb"
```

## Methodology
1. Data preprocessing: median imputation + z-score normalization
2. Temporal feature engineering (delta, max-prior, visit index)
3. Nested 5-fold CV (outer) / 3-fold (inner) with patient-level stratification
4. Threshold optimization targeting 98% recall
5. Evaluation: PR-AUC, ROC-AUC, Precision, Recall, F1

## Ethical Statement
Approved by the IRB of the University of Hail, Saudi Arabia.
Dataset de-identified prior to analysis.

## Citation
Alshammari, H.A.A. and Alshammari, A.E.M. (2026). AI-based prediction and
monitoring system for acute myocardial infarction detection using laboratory
biomarkers. PeerJ.

## License
MIT License
