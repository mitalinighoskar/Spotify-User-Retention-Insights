# spotify-user-rentention-insights-project

Simple end-to-end example for a churn prediction pipeline:
- feature engineering
- training (RandomForest)
- saving test predictions
- SHAP-based explainability
- a lightweight FastAPI serving endpoint
- unit test for feature engineering

## Requirements

Install (recommended in a virtualenv):

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# or: source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt


![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-RandomForest-green.svg)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal.svg)
