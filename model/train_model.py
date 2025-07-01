import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model.save_model import save_model_artifacts
from utils.logger import log_model_results

# 1. Load dataset
df = pd.read_csv("dataset/student-data.csv", sep=";")
df["graduate"] = (df["G3"] >= 10).astype(int)

# 2. Rename kolom
rename_map = {
    "famsize": "family_size",
    "Pstatus": "parent_status",
    "Medu": "mother_education_level",
    "Fedu": "father_education_level",
    "guardian": "guardian",
    "studytime": "study_freq",
    "failures": "failure_rate",
    "schoolsup": "school_support",
    "famsup": "family_support",
    "paid": "tutoring_support",
    "activities": "extracurricular_participation",
    "higher": "study_intent",
    "internet": "internet_support",
    "famrel": "family_relation_level",
    "goout": "hangout_freq",
    "absences": "presence_rate",
}
df = df[list(rename_map.keys()) + ["graduate"]].rename(columns=rename_map)
df["presence_rate"] = 1 - (df["presence_rate"] / df["presence_rate"].max())

# 3. Label encoding untuk kolom boolean
label_cols = [
    "school_support",
    "family_support",
    "tutoring_support",
    "extracurricular_participation",
    "study_intent",
    "internet_support"
]
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# 4. Feature Engineering
df["parent_edu_avg"] = (df["mother_education_level"] + df["father_education_level"]) / 2
df["high_absence_flag"] = (df["presence_rate"] > 0.3).astype(int)
df["single_parent_flag"] = (df["parent_status"] == "T").astype(int)
df["study_internet_interaction"] = df["study_freq"] * df["internet_support"]
df["support_total"] = (
    df["school_support"]
    + df["family_support"]
    + df["tutoring_support"]
    + df["internet_support"]
)

# 5. Definisi kolom
raw_onehot_cols = ["family_size", "parent_status", "guardian"]
normalize_cols = [
    "mother_education_level",
    "father_education_level",
    "study_freq",
    "failure_rate",
    "family_relation_level",
    "hangout_freq",
    "presence_rate",
    "parent_edu_avg",
    "study_internet_interaction",
    "support_total"
]
additional_binary_cols = ["high_absence_flag", "single_parent_flag"]

# 6. Preprocessing fitur
X_onehot = pd.get_dummies(df[raw_onehot_cols], drop_first=False)
X_normalized = pd.DataFrame(
    MinMaxScaler().fit_transform(df[normalize_cols]),
    columns=normalize_cols,
)
X = pd.concat([
    X_onehot.reset_index(drop=True),
    df[label_cols + additional_binary_cols].reset_index(drop=True),
    X_normalized.reset_index(drop=True),
], axis=1)
y = df["graduate"]

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Model training
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "model": model,
    }

# 9. Simpan model terbaik
best_model_name = max(results, key=lambda x: results[x]["f1_score"])
best_model = results[best_model_name]["model"]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

saved_paths = save_model_artifacts(
    model=best_model,
    normalize_data=df[normalize_cols],
    raw_onehot_columns=raw_onehot_cols,
    encoded_onehot_columns=X_onehot.columns.tolist(),
    label_columns=label_cols,
    normalize_columns=normalize_cols,
    timestamp=timestamp,
    model_name=best_model_name.replace(" ", "_").lower()
)

# 10. Logging
log_model_results(
    results=results,
    best_model_name=best_model_name,
    model_path=saved_paths["model_path"],
)

print(f"Best model saved: {best_model_name}")
print(f"Location: {saved_paths['model_path']}")
