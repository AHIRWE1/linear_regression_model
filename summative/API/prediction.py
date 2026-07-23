import joblib
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Load saved model, scaler and feature names
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DISPLAY_NAMES = {
    "LinearRegression": "Linear Regression",
    "SGDRegressor": "SGD Regressor",
}

model = None
scaler = None
feature_names = None


def reload_artifacts():
    """(Re)load model/scaler/feature_names from disk. Called at import time
    and again after /retrain writes new artifacts, so the running process
    picks up the retrained model without a restart."""
    global model, scaler, feature_names
    model = joblib.load(BASE_DIR / "best_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    feature_names = joblib.load(BASE_DIR / "feature_names.pkl")


reload_artifacts()


def preprocess_input(student_data: dict) -> pd.DataFrame:
    """
    Convert user-friendly input into the exact feature
    vector used during model training.
    """

    processed = {
        "study_time_hours": student_data["study_time_hours"],
        "attendance_percent": student_data["attendance_percent"],
        "sleep_hours": student_data["sleep_hours"],
        "previous_grade": student_data["previous_grade"],

        "gender_Male": 1 if student_data["gender"] == "Male" else 0,

        "parental_education_High School":
            1 if student_data["parental_education"] == "High School" else 0,

        "parental_education_Masters":
            1 if student_data["parental_education"] == "Masters" else 0,

        "parental_education_PhD":
            1 if student_data["parental_education"] == "PhD" else 0,

        "internet_access_Yes":
            1 if student_data["internet_access"] == "Yes" else 0,

        "extracurricular_activities_Yes":
            1 if student_data["extracurricular_activities"] == "Yes" else 0,

        "part_time_job_Yes":
            1 if student_data["part_time_job"] == "Yes" else 0,
    }

    # Reindexed against feature_names so column order always matches what
    # the scaler/model were fit on, even if this dict's order ever drifts.
    return pd.DataFrame([processed])[feature_names]


def predict_exam_score(student_data: dict) -> dict:
    """
    Predict a student's final exam score.
    """

    processed_data = preprocess_input(student_data)

    scaled_data = scaler.transform(processed_data)

    prediction = model.predict(scaled_data)

    model_class_name = type(model).__name__

    return {
        "prediction": round(float(prediction[0]), 2),
        "model": MODEL_DISPLAY_NAMES.get(model_class_name, model_class_name),
    }
