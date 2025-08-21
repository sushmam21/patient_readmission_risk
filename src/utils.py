import joblib
from pathlib import Path

def save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load(path: Path):
    return joblib.load(path)
