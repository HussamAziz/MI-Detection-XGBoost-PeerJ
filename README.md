# AI-Based Prediction and Monitoring System for Acute Myocardial Infarction Detection Using Laboratory Biomarkers

> **A Comparative Machine Learning Study with Temporal Feature Engineering**  
> Submitted to: *PeerJ Computer Science*  
> Authors: Hussam Aziz Ayed Alshammari, Abdullah Eyada Mutlaq Alshammari  
> Institution: University of Hail, College of Computer Science and Engineering, Saudi Arabia

---

## Description

This repository contains all code, figures, and LaTeX source files for the manuscript:

> *"AI-based prediction and monitoring system for acute myocardial infarction detection using laboratory biomarkers: a comparative machine learning study with temporal feature engineering"*

The study presents an AI-based clinical decision support system for detecting acute myocardial infarction (MI) using two laboratory biomarkers — **troponin** and **creatine kinase-MB (CK-MB)** — with temporal feature engineering. Four machine learning classifiers (Logistic Regression, Random Forest, Histogram Gradient Boosting, and XGBoost) are rigorously compared using nested cross-validation with patient-level stratification to prevent data leakage.

**Key result:** XGBoost achieved PR-AUC = 0.986 and ROC-AUC = 0.982 at 98% recall.

---

## Dataset Information

| Property | Details |
|---|---|
| **Records** | 2,327 hospital visits |
| **Unique Patients** | 1,506 |
| **MI Positive Rate** | 57.5% (visit level) |
| **Target Definition** | ICD-10 codes I21, I22 (acute MI) + keyword matching (STEMI, NSTEMI) |
| **Exclusions** | "History of MI", "old MI" — only acute events retained |
| **De-identification** | All patient IDs replaced with SHA-256 cryptographic hashes |
| **Format** | CSV |
| **Availability** | Provided as a **confidential supplemental file** to PeerJ reviewers/editors |

The dataset includes the following features:

| Feature | Type | Description |
|---|---|---|
| `age` | Numerical | Patient age (z-score normalized) |
| `visit_type` | Categorical | Visit context (one-hot encoded) |
| `visit_index` | Numerical | Sequential visit counter per patient |
| `days_since_last_visit` | Numerical | Days between consecutive visits |
| `troponin` | Numerical | Current troponin level |
| `troponin_prev` | Numerical | Previous troponin measurement |
| `troponin_delta` | Numerical | Change from previous troponin |
| `troponin_max_prior` | Numerical | Maximum prior troponin value |
| `ckmb` | Numerical | Current CK-MB level |
| `ckmb_prev` | Numerical | Previous CK-MB measurement |
| `ckmb_delta` | Numerical | Change from previous CK-MB |
| `ckmb_max_prior` | Numerical | Maximum prior CK-MB value |

> **Ethics:** The dataset was approved by the IRB of the University of Hail (IRB Approval No. IRB-UOH-2024-CS-001). Informed consent was waived by the IRB due to the retrospective, de-identified nature of the study.

---

## Code Information

The repository contains:

| File | Description |
|---|---|
| `MI_NestedCV_4Models_Target098_ProgressOutputs.ipynb` | Main analysis notebook (43 cells: preprocessing → features → nested CV → evaluation) |
| `main.tex` | LaTeX source for the manuscript |
| `references.bib` | BibTeX bibliography (29 references) |
| `wlpeerj.cls` | PeerJ LaTeX document class |
| `main.bbl` | Compiled bibliography |
| `requirements.txt` | Python dependency list |
| `fig_workflow.png` / `_hires.png` | Figure 1: Methodology workflow |
| `fig_model_comparison.png` / `_hires.png` | Figure 2: 4-model PR-AUC comparison |
| `fig_confusion_matrix.png` / `_hires.png` | Figure 3: XGBoost confusion matrix |
| `fig_pr_curve.png` / `_hires.png` | Figure 4: Precision-recall curve |
| `fig_roc_curve.png` / `_hires.png` | Figure 5: ROC curve |
| `fig_comparison_prior_work.png` / `_hires.png` | Figure 6: Literature AUROC comparison |
| `graphical_abstract.png` | Graphical abstract |

---

## Usage Instructions

### 1. Run the Analysis Notebook (Google Colab — recommended)

The analysis was developed and tested on **Google Colab** with Python 3.12.12.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload `MI_NestedCV_4Models_Target098_ProgressOutputs.ipynb` to Google Colab  
2. Upload the dataset CSV file to the Colab session  
3. Run all cells in order (`Runtime → Run all`)

### 2. Run Locally

```bash
# Clone the repository
git clone https://github.com/HussamAziz/MI-Detection-XGBoost-PeerJ.git
cd MI-Detection-XGBoost-PeerJ

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook MI_NestedCV_4Models_Target098_ProgressOutputs.ipynb
```

### 3. Compile the LaTeX Manuscript

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Requirements

- **Python**: 3.12+ (tested with 3.12.12)
- **Environment**: Google Colab or local Jupyter

Key libraries (see `requirements.txt` for full list):

```
xgboost>=1.7.0
scikit-learn>=1.6.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

All random operations use `SEED = 42` for full reproducibility.

---

## Methodology

The analysis pipeline follows these steps:

1. **Data Preprocessing**
   - Load clinical dataset (CSV)
   - Define MI target from ICD-10 codes (I21, I22)
   - Exclude chronic/historical MI cases
   - Replace patient IDs with cryptographic hashes

2. **Feature Engineering**
   - Median imputation for missing values
   - Z-score normalization for all numerical features
   - One-hot encoding for categorical `visit_type`
   - Temporal features: `_prev`, `_delta`, `_max_prior` for troponin and CK-MB
   - Visit tracking: `visit_index`, `days_since_last_visit`

3. **Nested Cross-Validation** (anti-leakage design)
   - **Outer loop**: 5-fold stratified CV with **patient-level grouping** (GroupStratifiedKFold)
   - **Inner loop**: 3-fold CV for hyperparameter tuning (RandomizedSearchCV, 12 iterations)
   - Optimization metric: Average Precision (PR-AUC)

4. **Models Trained**
   - Logistic Regression (L2, C tuned 10⁻³–10²)
   - Random Forest (600 trees, balanced subsample)
   - Histogram Gradient Boosting (HGB)
   - XGBoost (900 estimators, histogram tree method)

5. **Threshold Optimization**
   - For each fold, threshold is set to achieve ≥ 98% recall
   - Selected as: highest precision point with recall ≥ 0.98 on the PR curve

6. **Evaluation**
   - PR-AUC, ROC-AUC, Precision, Recall, F1-score (mean ± std across 5 folds)
   - Final model trained on full dataset with most-selected hyperparameters

---

## Results Summary

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **XGBoost** | **0.986 ± 0.005** | **0.982 ± 0.007** | 0.898 ± 0.021 | 0.981 ± 0.014 | 0.937 ± 0.010 |
| HGB | 0.985 ± 0.006 | 0.981 ± 0.008 | 0.905 ± 0.018 | 0.979 ± 0.015 | 0.941 ± 0.010 |
| Random Forest | 0.981 ± 0.009 | 0.980 ± 0.008 | 0.911 ± 0.009 | 0.982 ± 0.014 | 0.945 ± 0.008 |
| Logistic Regression | 0.965 ± 0.011 | 0.961 ± 0.012 | 0.778 ± 0.020 | 0.981 ± 0.013 | 0.867 ± 0.013 |

Final XGBoost model: **PR-AUC 0.9845**, **ROC-AUC 0.9804**, threshold = 0.17, Recall = 98.1%, 1312 TP / 26 FN.

---

## Citations

If you use this code or dataset in your research, please cite:

```
Alshammari, H.A.A. and Alshammari, A.E.M. (2026). AI-based prediction and
monitoring system for acute myocardial infarction detection using laboratory
biomarkers: a comparative machine learning study with temporal feature engineering.
PeerJ Computer Science. DOI: [to be assigned upon publication]
```

Key references used in this work:
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Cawley & Talbot (2010). On Over-Fitting in Model Selection. *JMLR 11*.
- Thygesen et al. (2018). Fourth Universal Definition of MI. *Circulation 138*.
- Saito & Rehmsmeier (2015). The Precision-Recall Plot. *PLoS ONE 10*.

---

## Ethical Statement

This study was approved by the **Institutional Review Board (IRB) of the University of Hail**, College of Computer Science and Engineering, Hail 2440, Saudi Arabia (**IRB Approval Number: IRB-UOH-2024-CS-001**). The study is retrospective and uses a fully de-identified dataset. Informed consent was **waived** by the IRB. Original IRB approval documentation is provided as a confidential supplemental file to the journal.

---

## Use of AI

The authors used **Claude** (claude-sonnet-4-6, Anthropic, https://claude.ai/) for grammar correction and manuscript editing only. No AI tool was used for data analysis, result generation, or scientific reasoning.

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 Hussam Aziz Ayed Alshammari, Abdullah Eyada Mutlaq Alshammari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

---

## Contribution Guidelines

This is a research repository. If you find a bug or have a question, please open a GitHub Issue.  
Pull requests for reproducibility improvements are welcome.

---

*Repository maintained by Hussam Aziz Ayed Alshammari — University of Hail, Saudi Arabia*
