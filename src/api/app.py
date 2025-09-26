from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from src.features.feature_engineering import add_features

MODEL_PATH = "src/data/model/churn_model.joblib"
FEATURES_PATH = "src/data/model/feature_columns.joblib"

# Load trained model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("⚠️ Model file not found! Train the model first.")

# Load feature column order
if os.path.exists(FEATURES_PATH):
    feature_columns = joblib.load(FEATURES_PATH)
else:
    feature_columns = None
    print("⚠️ feature_columns.joblib not found! Train the model to save them.")

app = FastAPI(title="Spotify Churn Insights API")

class UserInput(BaseModel):
    avg_daily_minutes: float
    number_of_playlists: int
    top_genre: str
    skips_per_day: int
    days_since_last_login: int
    subscription_type: str
    support_tickets: int
    country: str

@app.get("/")
def read_root():
    return {"message": "Spotify Churn Insights API is running 🚀"}

@app.post("/predict")
def predict(user: UserInput):
    if model is None or feature_columns is None:
        return {"error": "❌ Model or feature_columns not loaded. Train first."}

    # Convert to DataFrame
    df = pd.DataFrame([user.dict()])

    # Apply feature engineering (same as training)
    df = add_features(df)

    # ✅ One-hot encode categorical vars
    df = pd.get_dummies(df)

    # ✅ Reindex with training features (extra cols filled with 0)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Predict churn probability
    if "id" in df.columns:
     df = df.drop(columns=["id"])

    churn_probability = model.predict_proba(df)[0, 1]

    return {
        "churn_probability": float(churn_probability),
        "prediction": "Churn" if churn_probability > 0.5 else "Stay"
    }
