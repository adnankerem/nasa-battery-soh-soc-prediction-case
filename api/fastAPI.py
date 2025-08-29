import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import torch
import os

# --- Models config ---
MODELS = {
    # (target, model_type): (model_file, scaler_file)
    ("SoH_%", "xgboost"): ("/models/xgboost_model_SoH_%_discharge.pkl", None),
    ("SoH_%", "lightgbm"): ("/models/lightgbm_model_SoH_%_discharge.pkl", None),
    ("SoH_%", "mlp"): ("/models/mlp_regressor_SoH_%_discharge.pt", "/models/scaler_SoH_%_discharge.pkl"),
    ("SoC_Progress_%", "xgboost"): ("/models/xgboost_model_SoC_Progress_%_discharge.pkl", None),
    ("SoC_Progress_%", "lightgbm"): ("/models/lightgbm_model_SoC_Progress_%_discharge.pkl", None),
    ("SoC_Progress_%", "mlp"): ("/models/mlp_regressor_SoC_Progress_%_discharge.pt", "/models/scaler_SoC_Progress_%_discharge.pkl"),
}
FEATURE_LIST = joblib.load("feature_list.pkl")  # Save the feature list you use in training (recommended!)

# --- PyTorch Model (must match training!) ---
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- FastAPI and request schema ---
app = FastAPI()

class PredictRequest(BaseModel):
    features: dict
    target: str  # "SoH_%" or "SoC_Progress_%"
    model_type: str  # "xgboost", "lightgbm", "mlp"

@app.on_event("startup")
def load_all_models():
    global loaded_models
    loaded_models = {}
    for (target, model_type), (model_file, scaler_file) in MODELS.items():
        if not os.path.isfile(model_file):
            print(f"Model file missing: {model_file}")
            continue
        if model_type == "mlp":
            scaler = joblib.load(scaler_file)
            model = MLP(input_dim=len(FEATURE_LIST))
            model.load_state_dict(torch.load(model_file, map_location="cpu"))
            model.eval()
            loaded_models[(target, model_type)] = (model, scaler)
        else:
            model = joblib.load(model_file)
            loaded_models[(target, model_type)] = (model, None)
    print(f"Loaded models: {list(loaded_models.keys())}")

@app.post("/predict")
def predict(req: PredictRequest):
    # --- Validate ---
    key = (req.target, req.model_type)
    if key not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model not found: {req.target} + {req.model_type}")
    model, scaler = loaded_models[key]
    # --- Prepare feature vector in correct order ---
    features = req.features
    x_vec = []
    for fname in FEATURE_LIST:
        val = features.get(fname, np.nan)
        x_vec.append(val)
    X = np.array(x_vec).reshape(1, -1)

    # --- Impute or error for NaN values (optional: smarter imputation) ---
    if np.any(np.isnan(X)):
        raise HTTPException(status_code=400, detail="Missing feature(s) for prediction.")

    # --- Predict ---
    if req.model_type == "mlp":
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_pred = model(X_tensor).cpu().numpy().flatten()
    else:
        y_pred = model.predict(X)
    return {"prediction": float(y_pred[0])}

# --- To run the API server: ---
# uvicorn scriptname:app --reload --host 0.0.0.0 --port 8080
