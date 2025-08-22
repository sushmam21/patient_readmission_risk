import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from utils import load
from config import RAW_CSV, PIPELINE_PATH, MODEL_PATH, ARTIFACTS_DIR
from pipeline import split_Xy

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def save_cm(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def main():
    pre = load(PIPELINE_PATH)
    model = load(MODEL_PATH)

    df = pd.read_csv(RAW_CSV)
    X, y = split_Xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_te_p = pre.transform(X_te)

    proba = model.predict_proba(X_te_p)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    prec, rec, _ = precision_recall_curve(y_te, proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_te, proba)

    # Load tuned threshold if present
    thresh_path = ARTIFACTS_DIR / "threshold.json"
    t = 0.5
    if thresh_path.exists():
        t = json.loads(thresh_path.read_text()).get("threshold", 0.5)

    preds_50 = (proba >= 0.5).astype(int)
    preds_t  = (proba >= t).astype(int)

    # Save plots
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'--',lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(ARTIFACTS_DIR/"roc_curve.png", dpi=180); plt.close()

    plt.figure()
    plt.plot(rec, prec, label=f"PR (AP={pr_auc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve"); plt.legend()
    plt.tight_layout(); plt.savefig(ARTIFACTS_DIR/"pr_curve.png", dpi=180); plt.close()

    save_cm(y_te, preds_50, ARTIFACTS_DIR/"confusion_matrix_t0_50.png", "Confusion Matrix @ 0.50")
    save_cm(y_te, preds_t,  ARTIFACTS_DIR/"confusion_matrix_tuned.png", f"Confusion Matrix @ tuned={t:.2f}")

    # Save a metrics JSON blob
    out = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "threshold_used": float(t),
        "report_at_0.50": classification_report(y_te, preds_50, output_dict=True),
        "report_at_tuned": classification_report(y_te, preds_t, output_dict=True),
    }
    (ARTIFACTS_DIR/"metrics.json").write_text(json.dumps(out, indent=2))
    print("Saved: roc_curve.png, pr_curve.png, confusion_matrix_t0_50.png, confusion_matrix_tuned.png, metrics.json")

if __name__ == "__main__":
    main()
