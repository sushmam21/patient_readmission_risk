# Predictive Risk Modeling for Patient Readmission

## Results (baseline @ threshold = 0.50)
- ROC–AUC: **0.636**
- PR–AUC: **0.239**
- F1: **0.272**
- Accuracy: **0.725**

### Curves & Explainability
![Feature Importances](docs/img/feature_importance.png)
![ROC Curve](docs/img/roc_curve.png)
![PR Curve](docs/img/pr_curve.png)
![Confusion Matrix (0.50)](docs/img/confusion_matrix_t0_50.png)
![Confusion Matrix (tuned)](docs/img/confusion_matrix_tuned.png)

## Usage
```bash
python src/train.py
uvicorn serve:app --app-dir src --host 127.0.0.1 --port 8000 --reload
# open http://127.0.0.1:8000/docs and call POST /predict
