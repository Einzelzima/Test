"""FastAPI inference service.

Usage (after running train.py):
    uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000

Try:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"gross_floor_area_buildings_sq_ft": 100000}'
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
app = FastAPI(title="Chicago Energy Benchmarking — GHG Predictor")
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class BuildingInput(BaseModel):
    gross_floor_area_buildings_sq_ft: float = Field(
        ..., gt=0, description="Total floor area in square feet"
    )


class PredictOut(BaseModel):
    predicted_ghg_tons_co2e: float
    unit: str = "metric tons CO2e"

# Default path
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(b: BuildingInput):
    bundle = _load()
    X = pd.DataFrame([b.model_dump()])[bundle["features"]]
    pred = float(bundle["model"].predict(X)[0])
    return PredictOut(predicted_ghg_tons_co2e=round(pred, 2))
