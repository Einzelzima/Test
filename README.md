# Enterprise AI – Course Project Skeleton

A **working** ML pipeline. Trains a Linear Regression on Chicago building
energy data and serves predictions via a REST API.

## Data source
**Chicago Energy Benchmarking** — buildings >50,000 sq ft in Chicago report
annual energy use to the city. Free, no authentication.
- Portal: https://data.cityofchicago.org/d/xq83-jr8c
- API: `https://data.cityofchicago.org/resource/xq83-jr8c.json`

The skeleton ships with `data/data_full.csv` (sample data). In Tutorial 2
you'll replace it with a real API fetch.

## Quick start (inside the dev container)

```bash
# Train: train a simple linear regression using 1 numerical feature
python src/train.py

# After training the model, run this command to host the trained model locally
uvicorn api:app --app-dir src --host 0.0.0.0 --port 8000

# You can use this command to call the hosted model to make predictions
# For example, we call to the local host endpoint /predict, and provide 
# feature "gross_floor_area_buildings_sq_ft" = 150000,
# and expect the corresponding MSE and R2 score.
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"gross_floor_area_buildings_sq_ft": 150000}'

# You can see the interactive docs. You might want to extend the documentation later.
# Open http://localhost:8000/docs
```

## What's inside

| File | Does | Potential extensions |
|------|------|-----------------|
| `data.py` | Loads bundled CSV | Replace with API fetch, auto-scheduling for fetching |
| `preprocess.py` | 1 feature (floor area) for regression task | More features, add classification task |
| `split.py` | Training and testing models | Implement a validation phase |
| `train.py` | Linear Regression + MSE/R_2 scores | More complex models, more metrics, experimental tracking|
| `api.py` | Hosting a simple model for predictions with FastAPI | Extend inputs/outputs, deploy your app to a real remote server |

## Why is R_2 score low?
You will notice that by first running `train.py`, R_2 scores are around 0.5. Your task is to check out
what this means: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html. This will
give you some hints on how to extend the project.
