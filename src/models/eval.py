
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
import os

def evaluate(preds_path: str):
    if not os.path.exists(preds_path):
        raise FileNotFoundError(preds_path)
    df = pd.read_parquet(preds_path)
    if "churn_prob" not in df.columns:
        raise ValueError("preds file must contain churn_prob column")
    if "churned" not in df.columns:
        print("No ground-truth label 'churned' in preds file. Can't compute metrics. Showing sample preds:")
        print(df.head())
        return

    y_true = df["churned"].values
    y_pred = df["churn_prob"].values
    auc = roc_auc_score(y_true, y_pred)
    print("AUC:", auc)
    print(classification_report(y_true, (y_pred > 0.5).astype(int)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="Path to parquet with test predictions")
    args = parser.parse_args()
    evaluate(args.preds)
