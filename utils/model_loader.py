import os
import glob
import joblib
import re

MODEL_DIR = "artifacts/models"

def get_latest_artifact_paths(model_dir=MODEL_DIR):
    model_files = glob.glob(os.path.join(model_dir, "model_*.pkl"))
    if not model_files:
        raise FileNotFoundError("Model tidak ditemukan di artifacts/models.")

    model_main_files = [
        f for f in model_files if "_scaler" not in f and "_preprocessing" not in f
    ]
    latest_model = max(model_main_files, key=os.path.getctime)
    base_filename = os.path.basename(latest_model)

    match = re.match(r"model_(\d+_\d+)_.*\.pkl", base_filename)
    if not match:
        raise ValueError("Format nama model tidak dikenali.")
    timestamp = match.group(1)

    return {
        "model": os.path.join(model_dir, base_filename),
        "scaler": os.path.join(model_dir, f"model_{timestamp}_scaler.pkl"),
        "preprocessing": os.path.join(model_dir, f"model_{timestamp}_preprocessing.pkl"),
        "version": base_filename.replace(".pkl", "")
    }

def load_latest_model():
    paths = get_latest_artifact_paths()
    
    model = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])
    preprocessing = joblib.load(paths["preprocessing"])

    return model, scaler, preprocessing, paths["version"]
