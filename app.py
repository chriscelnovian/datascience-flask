from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from utils.model_loader import load_latest_model

app = Flask(__name__)

model, scaler, preprocessing, model_version = load_latest_model()

@app.route("/rest/open/student/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({"error": "Data harus berupa JSON."}), 400

        # Konfigurasi preprocessing
        label_cols = preprocessing["label_columns"]
        onehot_cols = preprocessing["onehot_columns"]
        encoded_cols = preprocessing["onehot_encoded_columns"]
        normalize_cols = preprocessing["normalize_columns"]

        # Konversi family_size
        fs = data.get("family_size")
        if isinstance(fs, int):
            data["family_size"] = "GT3" if fs > 3 else "LE3"
        elif fs not in ["GT3", "LE3"]:
            data["family_size"] = "LE3"

        # Buat DataFrame
        df_input = pd.DataFrame([data])

        # Rata-rata pendidikan orang tua
        df_input["parent_edu_avg"] = (
            df_input["mother_education_level"] + df_input["father_education_level"]
        ) / 2

        # Flag absensi tinggi
        df_input["high_absence_flag"] = (df_input["presence_rate"] > 0.3).astype(int)

        # Orang tua tunggal
        df_input["single_parent_flag"] = (df_input["parent_status"] == "T").astype(int)

        # Interaksi studi x internet
        df_input["study_internet_interaction"] = (
            df_input["study_freq"] * df_input["internet_support"]
        )

        # Total dukungan
        df_input["support_total"] = (
            df_input["school_support"]
            + df_input["family_support"]
            + df_input["tutoring_support"]
            + df_input["internet_support"]
        )

        # One-hot encoding
        df_onehot_raw = pd.get_dummies(df_input[onehot_cols])
        df_onehot = pd.DataFrame(0, index=df_input.index, columns=encoded_cols)
        for col in df_onehot_raw.columns:
            if col in encoded_cols:
                df_onehot[col] = df_onehot_raw[col]
        df_onehot = df_onehot[encoded_cols]

        # Label encoding
        df_label = df_input[label_cols]

        # Normalisasi numerik
        df_norm = pd.DataFrame(
            scaler.transform(df_input[normalize_cols]),
            columns=normalize_cols
        )

        # Gabungkan semua fitur
        X_final = pd.concat([df_onehot, df_label, df_input[["high_absence_flag", "single_parent_flag"]], df_norm], axis=1)
        X_final = X_final.reindex(columns=model.feature_names_in_, fill_value=0)

        # Validasi akhir
        if X_final.isnull().any().any():
            missing_info = X_final.isnull().sum()
            return jsonify({
                "error": "Terdapat nilai NaN dalam data setelah preprocessing.",
                "missing_info": missing_info[missing_info > 0].to_dict()
            }), 400

        # Prediksi
        pred = int(model.predict(X_final)[0])
        proba_lulus = float(model.predict_proba(X_final)[0][1])
        confidence = round(proba_lulus, 4)

        # Klasifikasi intervensi
        intervention = None
        if pred == 0:
            if proba_lulus >= 0.7:
                intervention = "Critical"
            elif proba_lulus >= 0.5:
                intervention = "High"
            else:
                intervention = "Medium"
        elif pred == 1 and proba_lulus < 0.6:
            intervention = "Medium"

        result = {
            "prediction": pred,
            "confidence": confidence,
            "model_version": model_version
        }

        if intervention:
            result["intervention_risk"] = intervention

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
