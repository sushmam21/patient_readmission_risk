import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

NUMERIC = [
    "age","length_of_stay","prior_admissions_1y",
    "num_medications","med_adherence_score","avg_systolic_bp","avg_glucose"
]
BINARY = ["chronic_diabetes","chronic_hf","chronic_ckd"]
CATEGORICAL = ["sex","discharge_to","insurance_type"]

FEATURES = NUMERIC + BINARY + CATEGORICAL
TARGET   = "readmit_30d"

def build_preprocessor():
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    bin_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC),
            ("bin", bin_pipe, BINARY),
            ("cat", cat_pipe, CATEGORICAL),
        ]
    )
    return pre

def split_Xy(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y
