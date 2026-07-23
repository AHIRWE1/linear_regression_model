from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

# Must stay in sync with the feature vector built in prediction.preprocess_input
# and with the one-hot-encoding step in the training notebook.
FEATURE_ORDER = [
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "previous_grade",
    "gender_Male",
    "parental_education_High School",
    "parental_education_Masters",
    "parental_education_PhD",
    "internet_access_Yes",
    "extracurricular_activities_Yes",
    "part_time_job_Yes",
]

REQUIRED_COLUMNS = [
    "gender",
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job",
    "previous_grade",
    "final_exam_score",
]

MIN_ROWS_TO_RETRAIN = 20


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    df = df[REQUIRED_COLUMNS].copy()

    df["parental_education"] = df["parental_education"].fillna(
        df["parental_education"].mode()[0]
    )

    df = pd.get_dummies(
        df,
        columns=[
            "gender",
            "parental_education",
            "internet_access",
            "extracurricular_activities",
            "part_time_job",
        ],
        drop_first=True,
    )

    # Any dummy column not present in this batch (e.g. no PhD parents in the
    # uploaded rows) is filled with 0 so the feature vector always matches
    # what the model/scaler expect.
    for column in FEATURE_ORDER:
        if column not in df.columns:
            df[column] = False

    X = df[FEATURE_ORDER]
    y = df["final_exam_score"]
    return X, y


def retrain_from_csv(csv_path: Path) -> dict:
    """
    Retrain on the given CSV (same schema as the original training dataset)
    and overwrite best_model.pkl / scaler.pkl / feature_names.pkl with
    whichever of {Linear Regression, SGD Regressor} scores higher on a
    held-out test split.
    """
    df = pd.read_csv(csv_path)

    if len(df) < MIN_ROWS_TO_RETRAIN:
        raise ValueError(
            f"Need at least {MIN_ROWS_TO_RETRAIN} rows to retrain reliably, got {len(df)}."
        )

    X, y = _prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "Linear Regression": LinearRegression(),
        "SGD Regressor": SGDRegressor(
            max_iter=1000, learning_rate="constant", eta0=0.01, random_state=42
        ),
    }

    scores = {}
    for name, candidate in candidates.items():
        candidate.fit(X_train_scaled, y_train)
        scores[name] = round(r2_score(y_test, candidate.predict(X_test_scaled)), 4)

    best_name = max(scores, key=scores.get)
    best_model = candidates[best_name]

    joblib.dump(best_model, BASE_DIR / "best_model.pkl")
    joblib.dump(scaler, BASE_DIR / "scaler.pkl")
    joblib.dump(FEATURE_ORDER, BASE_DIR / "feature_names.pkl")

    return {
        "model": best_name,
        "scores": scores,
        "rows_trained_on": len(df),
    }
