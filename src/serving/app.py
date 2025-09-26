# FastAPI app placeholder
# src/serving/app.py
from fastapi import FastAPI
import joblib, yaml, os, pandas as pd
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="Spotify Churn Predictor")

class UserFeatures(BaseModel):
    user_id: str
    subscription_type: str
    country: str = None
    avg_daily_minutes: float
    number_of_playlists: int
    top_genre: str = None
    skips_per_day: float
    support_tickets: int
    days_since_last_login: int

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

cfg = load_config()
MODEL_PATH = os.path.join(cfg["data_paths"]["model_dir"], "churn_model.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first with the pipeline.")
model = joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(u: UserFeatures):
    d = u.dict()
    df = pd.DataFrame([d])

    # feature engineering consistent with training
    df["engagement_score"] = (df["avg_daily_minutes"] / (1 + df["skips_per_day"])) + 2*df["number_of_playlists"]
    df["inactivity_streak"] = df["days_since_last_login"]
    df["support_frustration"] = df["support_tickets"] / (1 + df["days_since_last_login"])
    df["plan_is_premium"] = (df["subscription_type"] == "Premium").astype(int)

    # select model features — this should match training features
    feature_cols = [c for c in df.columns if c not in ("user_id", "subscription_type", "country", "top_genre")]
    X = df[feature_cols]
    prob = float(model.predict_proba(X)[0,1])
    return {"user_id": u.user_id, "churn_probability": prob}
