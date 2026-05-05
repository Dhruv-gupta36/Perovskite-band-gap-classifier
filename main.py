"""
Perovskite Band Gap Classification — Training Pipeline
Dataset: Excavate Perovskite Dataset (5152 compounds, 38 features)
Task: Predict whether a perovskite is metallic (band gap = 0) or semiconducting (band gap > 0)
"""

import os
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import load_data, preprocess, get_feature_columns, engineer_features
from src.eda import class_distribution, correlation_summary, run_pca_analysis
from src.model import split_data, apply_pca, build_models, evaluate_model, cross_validate_model

DATA_PATH = "data/dataset_excavate.csv"
MODEL_DIR = "models"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Load & preprocess ────────────────────────────────────────────────
    print("\n[1] Loading data...")
    df = load_data(DATA_PATH)
    print(f"  Dataset shape: {df.shape}")
    class_distribution(df)

    # ── 2. EDA ──────────────────────────────────────────────────────────────
    print("\n[2] Running EDA...")
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    correlation_summary(df, feature_cols, target='PBE band gap', top_n=10)

    # ── 3. Feature scaling ──────────────────────────────────────────────────
    print("\n[3] Preprocessing & scaling...")
    X_scaled, y, scaler, feature_cols = preprocess(df, fit_scaler=True)
    print(f"  Feature matrix: {X_scaled.shape}")

    # ── 4. PCA analysis ─────────────────────────────────────────────────────
    print("\n[4] PCA analysis...")
    run_pca_analysis(X_scaled, n_components=15)

    # ── 5. Train/test split ─────────────────────────────────────────────────
    print("\n[5] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ── 6. Apply PCA (95% variance) ─────────────────────────────────────────
    print("\n[6] Applying PCA for dimensionality reduction...")
    X_train_pca, X_test_pca, pca = apply_pca(X_train, X_test, n_components=0.95)

    # ── 7. Train & compare models ────────────────────────────────────────────
    print("\n[7] Training and evaluating models...")
    models = build_models()
    results = {}

    for name, model in models.items():
        result = evaluate_model(model, X_train_pca, X_test_pca, y_train, y_test, model_name=name)
        results[name] = result

    # ── 8. Cross-validate best model (SVM) ──────────────────────────────────
    print("\n[8] Cross-validating SVM (5-fold stratified)...")
    svm = models['SVM (RBF)']
    svm.fit(X_train_pca, y_train)  # already fitted; re-fit for CV on full train
    cross_validate_model(
        type(svm)(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
        X_train_pca, y_train, cv=5
    )

    # ── 9. Save best model & artifacts ──────────────────────────────────────
    print("\n[9] Saving model artifacts...")
    best_model = results['SVM (RBF)']['model']
    joblib.dump(best_model, f"{MODEL_DIR}/svm_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    joblib.dump(pca, f"{MODEL_DIR}/pca.pkl")
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")

    print("  Saved: models/svm_model.pkl")
    print("  Saved: models/scaler.pkl")
    print("  Saved: models/pca.pkl")
    print("  Saved: models/feature_cols.pkl")

    # ── 10. Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*45)
    print("  FINAL RESULTS SUMMARY")
    print("="*45)
    for name, res in results.items():
        print(f"  {name:<22} | AUC: {res['roc_auc']:.4f} | F1: {res['f1']:.4f} | Acc: {res['accuracy']:.4f}")
    print("="*45)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
