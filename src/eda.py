import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def class_distribution(df, target='is_semiconductor'):
    counts = df[target].value_counts()
    labels = {0: 'Metal', 1: 'Semiconductor'}
    print("Class Distribution:")
    for k, v in counts.items():
        print(f"  {labels[k]}: {v} ({v/len(df)*100:.1f}%)")


def correlation_summary(df, feature_cols, target='PBE band gap', top_n=10):
    corr = df[feature_cols + [target]].corr()[target].drop(target).abs().sort_values(ascending=False)
    print(f"\nTop {top_n} Features Correlated with '{target}':")
    for feat, val in corr.head(top_n).items():
        print(f"  {feat}: {val:.4f}")
    return corr


def run_pca_analysis(X_scaled, n_components=10):
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.searchsorted(cumvar, 0.95) + 1

    print("\nPCA Explained Variance:")
    for i, (ev, cv) in enumerate(zip(pca.explained_variance_ratio_, cumvar)):
        print(f"  PC{i+1}: {ev:.4f}  (cumulative: {cv:.4f})")

    print(f"\n  → {n_95} components explain 95% of variance")
    return pca, n_95
