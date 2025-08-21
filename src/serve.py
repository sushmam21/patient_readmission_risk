from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import uvicorn
from utils import load
from config import PIPELINE_PATH, MODEL_PATH

app = FastAPI(title="Readmission Risk API", version="1.0")

# Load artifacts at startup
pre = load(PIPELINE_PATH)
model = load(MODEL_PATH)

class PredictRequest(BaseModel):
    age: int
    sex: str
    length_of_stay: float
    prior_admissions_1y: int
    chronic_diabetes: int
    chronic_hf: int
    chronic_ckd: int
    num_medications: int
    med_adherence_score: float
    avg_systolic_bp: float
    avg_glucose: float
    discharge_to: str
    insurance_type: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = pd.DataFrame([req.model_dump()])
    X_trans = pre.transform(X)
    p = float(model.predict_proba(X_trans)[:,1][0])
    label = int(p >= 0.5)
    return {"readmit_proba": round(p, 4), "readmit_label": label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
