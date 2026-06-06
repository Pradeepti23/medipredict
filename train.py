import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ==========================================
# KIDNEY DISEASE MODEL
# ==========================================

print("Loading Kidney Dataset...")

kidney = pd.read_csv(
    r"C:\Users\HP\Downloads\kidney_disease.csv"
)

kidney.replace("?", pd.NA, inplace=True)

# Fill missing values
for col in kidney.columns:
    kidney[col] = kidney[col].fillna(
        kidney[col].mode()[0]
    )

# Encode categorical columns
for col in kidney.columns:

    if not pd.api.types.is_numeric_dtype(kidney[col]):

        kidney[col] = kidney[col].astype(str).str.strip()

        le = LabelEncoder()

        kidney[col] = le.fit_transform(
            kidney[col]
        )

# ----------------------------
# SELECT ONLY 5 FEATURES
# ----------------------------

X_kidney = kidney[
    ["age", "bp", "sg", "al", "su"]
]

# Target column
y_kidney = kidney["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X_kidney,
    y_kidney,
    test_size=0.2,
    random_state=42
)

kidney_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

kidney_model.fit(X_train, y_train)

kidney_pred = kidney_model.predict(X_test)

print(
    "Kidney Accuracy:",
    round(
        accuracy_score(
            y_test,
            kidney_pred
        ) * 100,
        2
    ),
    "%"
)

joblib.dump(
    kidney_model,
    "kidney_model.pkl"
)

print("Kidney Model Saved Successfully")


# ==========================================
# MENTAL HEALTH MODEL
# ==========================================

print("Loading Mental Health Dataset...")



mental = pd.read_csv(
    r"C:\Users\HP\Downloads\mental_health_survey_dataset_300k.csv"
)

# Show available columns
print("\nAvailable Columns:")
print(mental.columns.tolist())

# Fill missing values
for col in mental.columns:
    mental[col] = mental[col].fillna(
        mental[col].mode()[0]
    )

# Encode categorical columns
for col in mental.columns:
    if not pd.api.types.is_numeric_dtype(mental[col]):
        le = LabelEncoder()
        mental[col] = le.fit_transform(
            mental[col].astype(str)
        )

# ==================================================
# REPLACE THESE COLUMN NAMES WITH THE REAL ONES
# FROM print(mental.columns.tolist())
# ==================================================

X_mental = mental[
    [
        "age",
        "stress_score",
        "sleep_hours",
        "academic_or_job_pressure"
    ]
]

y_mental = mental["mental_health_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X_mental,
    y_mental,
    test_size=0.2,
    random_state=42
)

mental_model = RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

mental_model.fit(X_train, y_train)

mental_pred = mental_model.predict(X_test)

print(
    "Mental Accuracy:",
    round(
        accuracy_score(y_test, mental_pred) * 100,
        2
    ),
    "%"
)

joblib.dump(
    mental_model,
    "mental_model.pkl"
)

print("Mental Model Saved Successfully")
print("\n===================================")
print("ALL MODELS TRAINED SUCCESSFULLY")
print("===================================")