from pydantic import BaseModel
from typing import Optional

class PatientRecord(BaseModel):
    age: int
    sex: str                      # "M" or "F"
    length_of_stay: float
    prior_admissions_1y: int
    chronic_diabetes: int         # 0/1
    chronic_hf: int               # 0/1
    chronic_ckd: int              # 0/1
    num_medications: int
    med_adherence_score: float    # 0..1
    avg_systolic_bp: float
    avg_glucose: float
    discharge_to: str             # "home","snf","rehab","other"
    insurance_type: str           # "medicare","medicaid","commercial","selfpay"
    # label only for training
    readmit_30d: Optional[int] = None
