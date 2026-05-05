import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


NUMERIC_FEATURES = [
    'A_OS', "A'_OS", 'A_HOMO-', 'A_HOMO+', 'A_IE-', 'A_IE+',
    'A_LUMO-', 'A_LUMO+', 'A_X-', 'A_X+', 'A_Z_radii-', 'A_Z_radii+',
    'A_e_affin-', 'A_e_affin+', 'B_OS', "B'_OS", 'B_HOMO-', 'B_HOMO+',
    'B_IE-', 'B_IE+', 'B_LUMO-', 'B_LUMO+', 'B_X-', 'B_X+',
    'B_Z_radii-', 'B_Z_radii+', 'B_e_affin-', 'B_e_affin+',
    'mu', 'tau', 'new_tol', 't'
]

TARGET = 'is_semiconductor'


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding='latin1')
    df.columns = [c.replace('\xa0', '_') for c in df.columns]

    # Rename cryptic columns to interpretable names
    df.rename(columns={'?': 'mu', '??': 'tau', '???_?': 'new_tol'}, inplace=True)

    # Binary classification target: semiconductor (band gap > 0) vs metal (band gap = 0)
    df[TARGET] = (df['PBE band gap'] > 0).astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Electronegativity difference between A and B site ions
    df['X_diff_A'] = df['A_X+'] - df['A_X-']
    df['X_diff_B'] = df['B_X+'] - df['B_X-']

    # HOMO-LUMO gap proxies
    df['A_HL_gap'] = df['A_LUMO+'] - df['A_HOMO-']
    df['B_HL_gap'] = df['B_LUMO+'] - df['B_HOMO-']

    # Ionic radius asymmetry
    df['radii_asym_A'] = df['A_Z_radii+'] - df['A_Z_radii-']
    df['radii_asym_B'] = df['B_Z_radii+'] - df['B_Z_radii-']

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    engineered = ['X_diff_A', 'X_diff_B', 'A_HL_gap', 'B_HL_gap', 'radii_asym_A', 'radii_asym_B']
    return NUMERIC_FEATURES + engineered


def preprocess(df: pd.DataFrame, scaler=None, fit_scaler=True):
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values
    y = df[TARGET].values if TARGET in df.columns else None

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler, feature_cols
