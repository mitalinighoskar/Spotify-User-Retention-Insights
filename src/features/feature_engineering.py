import pandas as pd
import numpy as np

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["id"] = range(1, len(df) + 1) 
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["engagement_score"] = (
        df["avg_daily_minutes"] / (1 + df["skips_per_day"])
    ) + 2 * df["number_of_playlists"]

    df["inactivity_streak"] = df["days_since_last_login"]

    df["support_frustration"] = df["support_tickets"] / (1 + df["days_since_last_login"])

    df["plan_is_premium"] = (df["subscription_type"] == "Premium").astype(int)

    df.fillna(0, inplace=True)

    return df


def save_features(df: pd.DataFrame, out_path: str):
    df.to_parquet(out_path, index=False)
