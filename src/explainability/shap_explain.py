import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

def explain_saved_test():
    print("Explaining test set (SHAP)...")
    
    model = joblib.load("src/data/model/churn_model.joblib")
    
    df_test = pd.read_parquet("src/data/predictions/test_with_preds.parquet")
    
    target_col = "churned"
    feature_cols = [c for c in df_test.columns if c not in [target_col, "prediction", "pred_prob", "id"]]
    X_test = df_test[feature_cols]
    
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = feature_cols
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[2] == 2:
        shap_values = shap_values[:, :, 1]
    

    min_len = min(len(feature_names), shap_values.shape[1])
    feature_names = feature_names[:min_len]
    shap_values = shap_values[:, :min_len]
    X_test = X_test[feature_names]
    
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    os.makedirs("src/data/predictions", exist_ok=True)
    plt.savefig("src/data/predictions/shap_summary.png")
    print("Saved SHAP summary plot to src/data/predictions/shap_summary.png")
    
    os.makedirs("src/data/powerbi", exist_ok=True)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    
    shap_df.insert(0, "id", df_test["id"].values)
    shap_df.insert(1, "prediction", df_test["prediction"].values)
    shap_df.insert(2, "pred_prob", df_test["pred_prob"].values)
    shap_df.insert(3, target_col, df_test[target_col].values)
    
    shap_df.to_parquet("src/data/powerbi/shap_values.parquet", index=False)
    df_test.to_parquet("src/data/powerbi/test_with_preds.parquet", index=False)
    
    print("Saved SHAP values to src/data/powerbi/shap_values.parquet")
    print("Saved predictions to src/data/powerbi/test_with_preds.parquet")


if __name__ == "__main__":
    explain_saved_test()
