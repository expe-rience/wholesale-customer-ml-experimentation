from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("artifacts/best_model.joblib")
app = FastAPI(title="Wholesale Customer Classifier", version="0.1.0")
model = None

class CustomerFeatures(BaseModel):
    Region: int
    Fresh: float = Field(ge=0)
    Milk: float = Field(ge=0)
    Grocery: float = Field(ge=0)
    Frozen: float = Field(ge=0)
    Detergents_Paper: float = Field(ge=0)
    Delicassen: float = Field(ge=0)

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(payload: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Run benchmark first.")
    frame = pd.DataFrame([payload.model_dump()])
    prediction = model.predict(frame)[0]
    result = {"predicted_class": str(prediction)}
    if hasattr(model, "predict_proba"):
        result["class_probabilities"] = {
            str(label): float(prob)
            for label, prob in zip(model.classes_, model.predict_proba(frame)[0])
        }
    return result
