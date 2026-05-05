# Perovskite Band Gap Classification

> Predicting metallic vs. semiconducting behavior in perovskite oxides using quantum-chemical features and Support Vector Machines.

---

## Problem Statement

Perovskites (ABX₃ structures) are a critical class of materials in energy, electronics, and photovoltaics. A key property is whether a compound is **metallic** (band gap = 0 eV) or **semiconducting** (band gap > 0 eV). Experimental measurement is expensive and time-consuming. This project builds a machine learning classifier trained on quantum-chemical descriptors to predict this property directly from atomic and electronic features — enabling rapid computational screening of candidate materials.

---

## Dataset

- **Source**: Excavate Perovskite Dataset
- **Size**: 5,152 perovskite compounds × 38 features
- **Target**: Binary — `0 = Metal` (band gap = 0), `1 = Semiconductor` (band gap > 0)
- **Class balance**: ~51% Metal, ~49% Semiconductor (near-balanced)

**Feature groups:**
- A-site and B-site ionic properties: HOMO/LUMO energies, ionization energy, electronegativity (Pauling), ionic radii, electron affinity, oxidation state
- Structural descriptors: Goldschmidt tolerance factor (`t`), octahedral factor (`mu`), tau descriptor, new tolerance (`new_tol`)

---

## Approach

1. **EDA**: correlation analysis, class distribution, PCA variance study
2. **Feature Engineering**: 6 derived features (electronegativity differences, HOMO-LUMO gap proxies, ionic radii asymmetry)
3. **Preprocessing**: StandardScaler normalization
4. **Dimensionality Reduction**: PCA retaining 95% variance (38 → 20 components)
5. **Modeling**: SVM (RBF kernel) as primary; compared against Random Forest and Logistic Regression
6. **Evaluation**: Accuracy, F1, ROC-AUC; 5-fold stratified cross-validation
7. **Deployment**:FastAPI REST endpoint for inference

---

## Results

| Model               | Accuracy | F1 Score | ROC-AUC |
|---------------------|----------|----------|---------|
| **SVM (RBF)**       | **0.782** | **0.771** | **0.854** |
| Random Forest       | 0.768    | 0.752    | 0.859   |
| Logistic Regression | 0.629    | 0.626    | 0.692   |

**SVM CV ROC-AUC (5-fold):** `0.861 ± 0.009`

SVM was selected as the final model due to its strong generalization, consistent cross-validation performance, and interpretability for a classification boundary in high-dimensional PCA space.

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — SVM, PCA, preprocessing, evaluation
- **pandas / numpy** — data manipulation
- **FastAPI + Uvicorn** — model serving
- **joblib** — model serialization

---

## Future Improvements

- Hyperparameter tuning with `GridSearchCV.`
- Multi-class regression: predict the actual band gap value (eV)
- SHAP explainability for feature importance post-PCA
- Docker containerization for deployment
- Extend to non-perovskite crystal structures
