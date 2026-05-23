# 🏦 Australian Credit Approval Prediction

> **Can machine learning reliably decide who gets a credit card?**  
> This project builds a full classification pipeline — EDA → preprocessing → cross-validated model tuning → business-oriented evaluation — on the classic UCI Australian Credit Approval dataset.

---

## 📊 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | **87.68%** | 0.9115 |
| SVM (RBF kernel) | 86.23% | **0.9166** |

Both models were tuned with `GridSearchCV` over 5-fold Stratified CV, trained on 80% of the data, and evaluated on a completely held-out 20% test set.

---

## 🗂️ Project Structure

```
Australian_Credit_Approval/
│
├── Australian_Credit_Approval.ipynb   ← Main analysis notebook
├── README.md                          ← You are here
└── requirements.txt                   ← Python dependencies
```

---

## 🔍 What's Inside the Notebook

| Section | What it covers |
|---|---|
| **EDA** | Class distribution, boxplots, correlation heatmap, feature-target correlations |
| **Preprocessing** | Stratified train/test split, StandardScaler, SelectKBest (top 10 features by ANOVA F-score) |
| **Modelling** | Logistic Regression & SVM with GridSearchCV + StratifiedKFold |
| **Evaluation** | Accuracy, ROC-AUC, full classification report (precision/recall/F1), confusion matrices, ROC curve comparison |
| **Conclusion** | Business interpretation — when to use each model, cost of false positives vs false negatives |

---

## 📦 Dataset

**UCI Australian Credit Approval** (Quinlan, 1987)  
- 690 instances, 14 anonymised features, binary target (approved / rejected)  
- Loaded directly from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval) — no manual download needed

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/rohanbhosalerb/Australian_Credit_Approval.git
cd Australian_Credit_Approval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook Australian_Credit_Approval.ipynb
```

No local data file needed — the dataset is fetched automatically from the UCI repository.

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.x-purple?logo=pandas)
![seaborn](https://img.shields.io/badge/seaborn-0.13-teal)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

---

## 💡 Key Takeaways

- **Logistic Regression** (higher accuracy, more interpretable) suits regulated environments where model decisions need to be audited
- **SVM with RBF kernel** (higher AUC) offers better discrimination across approval thresholds — useful when risk appetite needs tuning
- Feature selection reduced the feature space from 14 → 10 with no loss in performance
- Next steps: add XGBoost/Random Forest, SHAP explainability, threshold optimisation via a business cost function

---

*Part of my M.Sc. Data Analytics coursework at RWTH Aachen University*
