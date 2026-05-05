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

1. **EDA** — correlation analysis, class distribution, PCA variance study
2. **Feature Engineering** — 6 derived features (electronegativity differences, HOMO-LUMO gap proxies, ionic radii asymmetry)
3. **Preprocessing** — StandardScaler normalization
4. **Dimensionality Reduction** — PCA retaining 95% variance (38 → 20 components)
5. **Modeling** — SVM (RBF kernel) as primary; compared against Random Forest and Logistic Regression
6. **Evaluation** — Accuracy, F1, ROC-AUC; 5-fold stratified cross-validation
7. **Deployment** — FastAPI REST endpoint for inference

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

## Project Structure

```
perovskite_ml/
├── data/
│   └── dataset_excavate.csv       # Raw dataset
├── models/                        # Saved model artifacts (generated after training)
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── feature_cols.pkl
├── src/
│   ├── preprocessing.py           # Data loading, feature engineering, scaling
│   ├── eda.py                     # Correlation, PCA analysis utilities
│   └── model.py                   # Training, evaluation, cross-validation
├── api/
│   ├── app.py                     # FastAPI application
│   └── example_request.py         # Sample API client
├── main.py                        # End-to-end training pipeline
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/perovskite-band-gap-classifier.git
cd perovskite-band-gap-classifier
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the model

```bash
python main.py
```

This will run the full pipeline — EDA, preprocessing, PCA, model training, evaluation, and save artifacts to `models/`.

### 4. Start the API server

```bash
uvicorn api.app:app --reload
```

API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### 5. Send a prediction request

```bash
python api/example_request.py
```

Or use curl:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"A_OS": 1, "A_prime_OS": 1, "B_OS": 5, "B_prime_OS": 5.0,
       "A_HOMO_minus": -5.6, "A_HOMO_plus": -5.6, "A_IE_minus": 7.58,
       "A_IE_plus": 7.58, "A_LUMO_minus": -0.5, "A_LUMO_plus": -0.5,
       "A_X_minus": 1.93, "A_X_plus": 1.93, "A_Z_radii_minus": 1.29,
       "A_Z_radii_plus": 1.29, "A_e_affin_minus": 1.30, "A_e_affin_plus": 1.30,
       "B_HOMO_minus": -7.2, "B_HOMO_plus": -7.2, "B_IE_minus": 8.0,
       "B_IE_plus": 8.0, "B_LUMO_minus": -1.1, "B_LUMO_plus": -1.1,
       "B_X_minus": 2.02, "B_X_plus": 2.02, "B_Z_radii_minus": 1.03,
       "B_Z_radii_plus": 1.03, "B_e_affin_minus": 0.95, "B_e_affin_plus": 0.95,
       "mu": 0.54, "tau": 0.0, "new_tol": 0.0, "t": 0.946}'
```

**Expected response:**
```json
{
  "prediction": 1,
  "label": "Semiconductor",
  "probability_semiconductor": 0.7832,
  "probability_metal": 0.2168
}
```

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — SVM, PCA, preprocessing, evaluation
- **pandas / numpy** — data manipulation
- **FastAPI + Uvicorn** — model serving
- **joblib** — model serialization

---

## Future Improvements

- Hyperparameter tuning with `GridSearchCV` or `Optuna`
- Multi-class regression: predict the actual band gap value (eV)
- SHAP explainability for feature importance post-PCA
- Docker containerization for deployment
- Extend to non-perovskite crystal structures
