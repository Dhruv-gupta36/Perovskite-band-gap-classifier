"""
FastAPI app for Perovskite Band Gap Classification
Endpoint: POST /predict
"""

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os

app = FastAPI(
    title="Perovskite Classifier API",
    description="Predicts whether a perovskite compound is metallic or semiconducting.",
    version="1.0.0"
)

MODEL_DIR = "models"
model, scaler, pca, feature_cols = None, None, None, None


def load_artifacts():
    global model, scaler, pca, feature_cols
    model = joblib.load(f"{MODEL_DIR}/svm_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    pca = joblib.load(f"{MODEL_DIR}/pca.pkl")
    feature_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")


@app.on_event("startup")
def startup_event():
    if os.path.exists(f"{MODEL_DIR}/svm_model.pkl"):
        load_artifacts()
    else:
        raise RuntimeError("Model artifacts not found. Run main.py first to train the model.")


class PerovskiteInput(BaseModel):
    A_OS: float = Field(..., description="Oxidation state of A-site ion")
    A_prime_OS: float = Field(..., description="Oxidation state of A'-site ion")
    A_HOMO_minus: float
    A_HOMO_plus: float
    A_IE_minus: float
    A_IE_plus: float
    A_LUMO_minus: float
    A_LUMO_plus: float
    A_X_minus: float
    A_X_plus: float
    A_Z_radii_minus: float
    A_Z_radii_plus: float
    A_e_affin_minus: float
    A_e_affin_plus: float
    B_OS: float
    B_prime_OS: float
    B_HOMO_minus: float
    B_HOMO_plus: float
    B_IE_minus: float
    B_IE_plus: float
    B_LUMO_minus: float
    B_LUMO_plus: float
    B_X_minus: float
    B_X_plus: float
    B_Z_radii_minus: float
    B_Z_radii_plus: float
    B_e_affin_minus: float
    B_e_affin_plus: float
    mu: float = Field(..., description="Octahedral factor")
    tau: float = Field(..., description="Tau descriptor")
    new_tol: float = Field(..., description="New tolerance factor descriptor")
    t: float = Field(..., description="Goldschmidt tolerance factor")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_semiconductor: float
    probability_metal: float


def build_feature_vector(data: PerovskiteInput) -> np.ndarray:
    raw = np.array([
        data.A_OS, data.A_prime_OS,
        data.A_HOMO_minus, data.A_HOMO_plus,
        data.A_IE_minus, data.A_IE_plus,
        data.A_LUMO_minus, data.A_LUMO_plus,
        data.A_X_minus, data.A_X_plus,
        data.A_Z_radii_minus, data.A_Z_radii_plus,
        data.A_e_affin_minus, data.A_e_affin_plus,
        data.B_OS, data.B_prime_OS,
        data.B_HOMO_minus, data.B_HOMO_plus,
        data.B_IE_minus, data.B_IE_plus,
        data.B_LUMO_minus, data.B_LUMO_plus,
        data.B_X_minus, data.B_X_plus,
        data.B_Z_radii_minus, data.B_Z_radii_plus,
        data.B_e_affin_minus, data.B_e_affin_plus,
        data.mu, data.tau, data.new_tol, data.t,
        # Engineered features
        data.A_X_plus - data.A_X_minus,        # X_diff_A
        data.B_X_plus - data.B_X_minus,        # X_diff_B
        data.A_LUMO_plus - data.A_HOMO_minus,  # A_HL_gap
        data.B_LUMO_plus - data.B_HOMO_minus,  # B_HL_gap
        data.A_Z_radii_plus - data.A_Z_radii_minus,  # radii_asym_A
        data.B_Z_radii_plus - data.B_Z_radii_minus,  # radii_asym_B
    ])
    return raw.reshape(1, -1)


@app.get("/")
def root():
    return {"message": "Perovskite Classifier API is running. Use POST /predict to get predictions."}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PerovskiteInput):
    try:
        X = build_feature_vector(data)
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        pred = int(model.predict(X_pca)[0])
        prob = model.predict_proba(X_pca)[0]

        return PredictionResponse(
            prediction=pred,
            label="Semiconductor" if pred == 1 else "Metal",
            probability_semiconductor=round(float(prob[1]), 4),
            probability_metal=round(float(prob[0]), 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
