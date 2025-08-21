import numpy as np
import pandas as pd
from pathlib import Path
from config import DATA_DIR

np.random.seed(42)
DATA_DIR.mkdir(exist_ok=True)

N = 15000
sex = np.random.choice(["M", "F"], N)
discharge_to = np.random.choice(["home", "snf", "rehab", "other"], N, p=[0.7, 0.15, 0.1, 0.05])
insurance = np.random.choice(["medicare","medicaid","commercial","selfpay"], N, p=[0.45,0.25,0.25,0.05])

df = pd.DataFrame({
    "age": np.random.randint(20, 95, N),
    "sex": sex,
    "length_of_stay": np.round(np.random.gamma(2.0, 2.0, N), 1),
    "prior_admissions_1y": np.random.poisson(0.6, N),
    "chronic_diabetes": np.random.binomial(1, 0.25, N),
    "chronic_hf": np.random.binomial(1, 0.15, N),
    "chronic_ckd": np.random.binomial(1, 0.12, N),
    "num_medications": np.random.randint(0, 20, N),
    "med_adherence_score": np.clip(np.random.normal(0.75, 0.15, N), 0, 1),
    "avg_systolic_bp": np.random.normal(128, 15, N).round(1),
    "avg_glucose": np.random.normal(105, 25, N).round(1),
    "discharge_to": discharge_to,
    "insurance_type": insurance,
})

# True signal: risk grows with comorbidities, prior admits, LoS, low adherence, SNF discharge, older age
logit = (
    -3.0
    + 0.02 * df["age"]
    + 0.15 * df["prior_admissions_1y"]
    + 0.07 * df["length_of_stay"]
    + 0.6 * df["chronic_diabetes"]
    + 0.7 * df["chronic_hf"]
    + 0.6 * df["chronic_ckd"]
    - 1.2 * df["med_adherence_score"]
    + 0.5 * (df["discharge_to"] == "snf").astype(int)
)

prob = 1 / (1 + np.exp(-logit))
df["readmit_30d"] = (np.random.rand(N) < prob).astype(int)

out = DATA_DIR / "sample_readmissions.csv"
df.to_csv(out, index=False)
print(f"Wrote {out} (positives={df['readmit_30d'].mean():.3f})")
