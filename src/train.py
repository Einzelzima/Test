"""Train a Linear Regression to predict building energy use intensity (EUI).

Usage:
    python src/train.py
"""
import json
from pathlib import Path
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from data import load_data
from preprocess import preprocess, FEATURE_COLS, LABEL_COL
from split import split_data


def train_model(train, features, label):
    model = LinearRegression()
    model.fit(train[features], train[label])
    return model


def evaluate(model, df, features, label):
    pred = model.predict(df[features])
    return {
        "mse":  round(float(mean_squared_error(df[label], pred)), 2),
        "mae":  round(float(mean_absolute_error(df[label], pred)), 2),
        "r2":   round(float(r2_score(df[label], pred)), 4),
    }


def main():
    # 1. Load the data
    print("[1/4] Loading data…")
    raw = load_data()
    print(f"{len(raw):,} rows loaded")

    # 2. Preprocess the raw data into a suitable form for learning + feature engineering
    print("[2/4] Preprocessing…")
    clean = preprocess(raw)
    print(f"{len(clean):,} rows after cleaning, features: {FEATURE_COLS}")

    # 3. Split data into different sets for standard machine learning
    print("[3/4] Splitting…")
    train, test = split_data(clean)
    print(f"train: {len(train):,}  test: {len(test):,}")

    # 4. Train & evaluate a simple linear regression model
    print("[4/4] Training Linear Regression…")
    model = train_model(train, FEATURE_COLS, LABEL_COL)

    # 5. Define simple metrics: MSE and R2 scores
    metrics = {
        "train": evaluate(model, train, FEATURE_COLS, LABEL_COL),
        "test":  evaluate(model, test, FEATURE_COLS, LABEL_COL),
    }
    print(json.dumps(metrics, indent=2))

    # Potential extension: adding WanDB or MLFlow experimental tracking

    # Save the trained model into your local folder. This will be used
    # later for hosting the moodel locally. 
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLS}, "models/model.joblib")
    print("[done] saved → models/model.joblib")


if __name__ == "__main__":
    main()
