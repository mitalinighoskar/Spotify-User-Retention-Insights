
from src.models.train import train_model
from src.explainability.shap_explain import explain_saved_test
import os
import pandas as pd

def export_for_powerbi():
    """Export processed data & predictions to Power BI folder."""
    powerbi_dir = "powerbi"
    os.makedirs(powerbi_dir, exist_ok=True)

    preds_path = "src/data/predictions/test_with_preds.parquet"
    if os.path.exists(preds_path):
        preds_df = pd.read_parquet(preds_path)
        preds_df.to_csv(os.path.join(powerbi_dir, "predictions.csv"), index=False)

    shap_values_path = "src/data/predictions/shap_values.parquet"
    if os.path.exists(shap_values_path):
        shap_df = pd.read_parquet(shap_values_path)
        shap_df.to_csv(os.path.join(powerbi_dir, "shap_values.csv"), index=False)

    print(f"✅ Exported Power BI files to {powerbi_dir}/")

def run_all():
    print("Training model...")
    train_model()
    print("Explaining test set (SHAP)...")
    explain_saved_test()
    export_for_powerbi()
    print("Pipeline finished.")

if __name__ == "__main__":
    run_all()
