from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Files
RAW_CSV = DATA_DIR / "sample_readmissions.csv"
MODEL_PATH = ARTIFACTS_DIR / "model_xgb.pkl"
PIPELINE_PATH = ARTIFACTS_DIR / "sklearn_pipeline.pkl"
SHAP_SUMMARY_PNG = ARTIFACTS_DIR / "shap_summary.png"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 400
LEARNING_RATE = 0.05
MAX_DEPTH = 5
