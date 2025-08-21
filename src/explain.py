import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load
from config import PIPELINE_PATH, MODEL_PATH, RAW_CSV, SHAP_SUMMARY_PNG
from pipeline import split_Xy

def main():
    pre = load(PIPELINE_PATH)
    model = load(MODEL_PATH)
    df = pd.read_csv(RAW_CSV)
    X, y = split_Xy(df)

    X_trans = pre.transform(X)
    # Use a small background sample for speed
    background = shap.sample(X_trans, 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(background)

    # Summary plot (bar)
    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PNG, dpi=180)
    print(f"Saved SHAP summary -> {SHAP_SUMMARY_PNG}")

if __name__ == "__main__":
    main()
