# Unit tests for feature engineering
# tests/test_feature_engineering.py
from src.features.feature_engineering import add_features
import pandas as pd

def test_add_features():
    df = pd.DataFrame([{
        "avg_daily_minutes": 100,
        "skips_per_day": 5,
        "number_of_playlists": 2,
        "support_tickets": 1,
        "days_since_last_login": 3,
        "subscription_type": "Premium"
    }])
    out = add_features(df)
    assert "engagement_score" in out.columns
    assert "support_frustration" in out.columns
    # quick numeric sanity checks
    assert out.loc[0, "plan_is_premium"] == 1
    assert out.loc[0, "inactivity_streak"] == 3
