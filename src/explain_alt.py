# src/explain_alt.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import load
from config import RAW_CSV, PIPELINE_PATH, MODEL_PATH, ARTIFACTS_DIR
from pipeline import split_Xy

ARTIFACTS_DIR.mkdir(exist_ok=True)

def main():
    # Load artifacts
    pre = load(PIPELINE_PATH)
    model = load(MODEL_PATH)

    # Load & transform data
    df = pd.read_csv(RAW_CSV)
    X, _ = split_Xy(df)
    X_trans = pre.transform(X)

    # Get feature names
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f_{i}" for i in range(X_trans.shape[1])])

    # Built-in feature importance (XGBoost)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise RuntimeError("Model has no feature_importances_. Try permutation_importance.")

    # Plot top 20
    idx = np.argsort(importances)[-20:]
    plt.figure(figsize=(9, 6))
    plt.barh(range(len(idx)), importances[idx])
    plt.yticks(range(len(idx)), feature_names[idx])
    plt.xlabel("Feature importance")
    plt.title("Top 20 Feature Importances (XGBoost)")
    plt.tight_layout()
    out = ARTIFACTS_DIR / "feature_importance.png"
    plt.savefig(out, dpi=160)
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
