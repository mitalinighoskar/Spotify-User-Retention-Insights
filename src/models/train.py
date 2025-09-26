import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from joblib import dump
from src.features.feature_engineering import load_raw, add_features, save_features
import os
import shap
import joblib


def train_model():
    raw_path = "src/data/raw/spotify_churn_dataset.csv"
    features_out_path = "src/data/processed/features.parquet"
    model_out_path = "src/models/churn_model.pkl"
    shap_out_path = "src/models/shap_explainer.pkl"
    predictions_out_path = "src/data/predictions/test_with_preds.parquet"

    df = load_raw(raw_path)

    df = add_features(df)

    save_features(df, features_out_path)

    target_col = "churned"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, "src/data/model/feature_columns.joblib")

    id_col = X["id"]  
    X = X.select_dtypes(include=[np.number])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, id_col, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train.drop(columns=["id"]), y_train)

    y_pred_proba = model.predict_proba(X_test.drop(columns=["id"]))[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Validation AUC: {auc:.3f}")
    print(classification_report(y_test, model.predict(X_test.drop(columns=["id"]))))

    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    dump(model, model_out_path)
    joblib.dump(model, "src/data/model/churn_model.joblib")
    joblib.dump(X_train.columns.tolist(), "src/data/model/feature_columns.joblib")
    print(f"✅ Saved model to {model_out_path}")

    print("Fitting SHAP explainer (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    dump(explainer, shap_out_path)
    print(f"✅ Saved SHAP explainer to {shap_out_path}")

    preds_df = X_test.copy()
    preds_df["id"] = id_test
    preds_df[target_col] = y_test
    preds_df["prediction"] = model.predict(X_test.drop(columns=["id"]))
    preds_df["pred_prob"] = y_pred_proba
    os.makedirs(os.path.dirname(predictions_out_path), exist_ok=True)
    preds_df.to_parquet(predictions_out_path, index=False)
    print(f"✅ Saved predictions to {predictions_out_path}")


if __name__ == "__main__":
    train_model()
