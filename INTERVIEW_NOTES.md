# Technical Interview Notes — Perovskite Band Gap Classifier

---

## 1. Why SVM?

SVM was chosen as the primary model for three reasons:

**High-dimensional, small-to-medium data.** With 5,152 samples and 38 features (reduced to 20 via PCA), SVMs are well-suited. They work by finding the optimal hyperplane in the transformed feature space — they don't struggle in moderate dimensions the way distance-based methods can.

**RBF kernel for non-linear separation.** The relationship between electronic/atomic features and band gap behavior is non-linear. The RBF (Gaussian) kernel implicitly maps inputs to an infinite-dimensional space, allowing the model to capture complex boundaries without manually engineering non-linear features.

**Strong generalization.** SVM maximizes the margin between classes, which acts as implicit regularization and reduces overfitting — especially important here since many features are correlated. The final model achieved **ROC-AUC = 0.854** on the test set and **0.861 ± 0.009** on 5-fold CV, indicating consistent generalization.

**Comparison against alternatives:**
- Random Forest achieved similar AUC (0.859) but lower F1 (0.752 vs 0.771). It also requires more hyperparameter decisions (depth, min samples) for stable results.
- Logistic Regression significantly underperformed (AUC 0.692), confirming the problem is not linearly separable.

---

## 2. Role of PCA

PCA was applied **after scaling** and before model training for two reasons:

**Dimensionality reduction.** Many features are physically correlated — for example, HOMO/LUMO energies for + and − oxidation states of the same element. These correlations inflate the feature space without adding discriminative power. PCA decorrelates the features and compresses 38 dimensions to **20 principal components** while retaining 95% of variance.

**SVM efficiency.** SVM training complexity scales with n_samples × n_features. Reducing from 38 to 20 components speeds up training meaningfully, and the decorrelated PCA space is better conditioned for the RBF kernel.

**Key insight:** PC1 alone captures ~20.7% of variance, and the first 7 components explain ~73%. This shows significant redundancy in the raw feature set — a strong signal that PCA is appropriate here.

---

## 3. Key Preprocessing Steps

| Step | Why |
|------|-----|
| Encoding fix (`latin1`) | CSV contains non-UTF-8 bytes from special characters in column names |
| Column renaming | `?`, `??`, `???_?` are unreadable; renamed to `mu`, `tau`, `new_tol` |
| Binary target creation | `PBE band gap > 0` → semiconductor, `= 0` → metal |
| Feature engineering | 6 derived features: electronegativity differences (X_diff_A/B), HOMO-LUMO gaps (A/B_HL_gap), ionic radii asymmetry (radii_asym_A/B) — these encode physically meaningful contrasts |
| StandardScaler | SVM and PCA are both distance/variance sensitive; without scaling, features with large ranges (e.g., IE in eV) dominate |
| PCA (95% variance) | Reduces 38 → 20 components; removes correlated noise |

---

## 4. Evaluation Metrics and Reasoning

**Why not just accuracy?**
The dataset is nearly balanced (~51/49 split), so accuracy is a reasonable metric here. But it alone doesn't reveal per-class errors — a model biased toward one class can still score well.

| Metric | Value (SVM) | What it tells us |
|--------|-------------|-----------------|
| Accuracy | 0.782 | Overall correct classification rate |
| F1 Score | 0.771 | Harmonic mean of precision and recall; good for imbalanced-adjacent tasks |
| ROC-AUC | 0.854 | Probability ranking quality; threshold-independent; most informative for screening tasks |
| CV AUC (5-fold) | 0.861 ± 0.009 | Confirms model isn't overfitting to one train/test split |

**ROC-AUC is the primary metric** because in materials screening, you want to rank candidates by confidence (not just classify). A high AUC means the model reliably assigns higher semiconductor probability to actual semiconductors across all thresholds.

---

## 5. Trade-offs and Possible Improvements

**Trade-offs made:**
- *PCA loses interpretability.* We can no longer directly say "feature X drove this prediction." SHAP values on the original space or kernel SHAP on the PCA-transformed model would help recover this.
- *SVM doesn't scale to very large datasets.* For >500K samples, a tree-based method (XGBoost, LightGBM) or a neural network would be more practical.
- *Binary classification discards information.* Many compounds have band gaps of 0.1–0.5 eV vs. 3–4 eV — collapsing these to a single class loses nuance.

**Improvements worth exploring:**
- **Regression instead of classification** — predict the actual band gap value (eV); use MAE and R² as metrics
- **Hyperparameter tuning** — `GridSearchCV` over SVM's `C` and `gamma`; the current values (C=10) were chosen from domain intuition but could be optimized
- **Ensemble methods** — Stacking SVM + Random Forest often outperforms either alone
- **SHAP explainability** — critical if this were used in real materials discovery
- **Neural networks (e.g., SchNet, MEGNet)** — graph neural networks that encode crystal structure directly outperform tabular ML on large perovskite databases
- **Active learning** — use the model's uncertainty to select which compounds to simulate next, reducing DFT compute cost
