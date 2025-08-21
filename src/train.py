import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from pipeline import build_preprocessor, split_Xy
from utils import save
from config import RAW_CSV, PIPELINE_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE, N_ESTIMATORS, LEARNING_RATE, MAX_DEPTH

def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Either place your CSV there or run generate_synthetic.py first."
        )

    df = pd.read_csv(RAW_CSV)
    X, y = split_Xy(df)

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Preprocess
    pre = build_preprocessor()
    X_tr_p = pre.fit_transform(X_tr)
    X_te_p = pre.transform(X_te)

    # Class imbalance handling (simple oversampling)
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_tr_bal, y_tr_bal = ros.fit_resample(X_tr_p, y_tr)

    # Model
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        tree_method="hist",
        eval_metric="logloss"
    )
    model.fit(X_tr_bal, y_tr_bal)

    # Eval
    proba = model.predict_proba(X_te_p)[:,1]
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_te, proba)
    ap  = average_precision_score(y_te, proba)
    f1  = f1_score(y_te, preds)
    print(f"ROC-AUC: {auc:.3f} | AvgPrec: {ap:.3f} | F1: {f1:.3f}")
    print(classification_report(y_te, preds, digits=3))

    # Save artifacts
    save(pre, PIPELINE_PATH)
    save(model, MODEL_PATH)
    print(f"Saved pipeline -> {PIPELINE_PATH}")
    print(f"Saved model    -> {MODEL_PATH}")

if __name__ == "__main__":
    main()
