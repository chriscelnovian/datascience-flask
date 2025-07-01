import os
import joblib
from sklearn.preprocessing import MinMaxScaler

MODEL_DIR = os.path.join("artifacts", "models")

def clean_old_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            os.remove(os.path.join(MODEL_DIR, f))

def save_model_artifacts(
    model,
    normalize_data,
    raw_onehot_columns,
    encoded_onehot_columns,
    label_columns,
    normalize_columns,
    timestamp,
    model_name
):
    clean_old_models()

    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}_{model_name}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"model_{timestamp}_scaler.pkl")
    info_path = os.path.join(MODEL_DIR, f"model_{timestamp}_preprocessing.pkl")

    # Simpan model utama
    joblib.dump(model, model_path)

    # Fit dan simpan scaler
    scaler = MinMaxScaler().fit(normalize_data)
    joblib.dump(scaler, scaler_path)

    # Simpan informasi preprocessing
    encoder_info = {
        "onehot_columns": raw_onehot_columns,
        "onehot_encoded_columns": encoded_onehot_columns,
        "label_columns": label_columns,
        "normalize_columns": normalize_columns
    }
    joblib.dump(encoder_info, info_path)

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "info_path": info_path
    }
