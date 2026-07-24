import shutil
import tempfile
from pathlib import Path
from typing import Literal

import prediction
import train
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Student Performance Prediction API",
    description="Predicts students' final examination scores using machine learning.",
    version="1.0.0",
)

# CORS is a browser-enforced mechanism only: the Flutter mobile app (Android/iOS)
# never sends an Origin header, so this list has no effect on it. It matters for
# browser-based callers only — Swagger UI's "Try it out" and a local web build of
# the Flutter app during development — so origins are scoped to those, not "*".
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "http://localhost:58123",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [
        {
            "field": ".".join(str(part) for part in error["loc"][1:]),
            "message": error["msg"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": "validation_error", "details": details},
    )


class StudentData(BaseModel):
    gender: Literal["Male", "Female"]
    study_time_hours: float = Field(
        ..., ge=0, le=24, description="Average daily study time, in hours (0-24)."
    )
    attendance_percent: float = Field(
        ..., ge=0, le=100, description="Class attendance percentage (0-100)."
    )
    sleep_hours: float = Field(
        ..., ge=0, le=24, description="Average daily sleep, in hours (0-24)."
    )
    parental_education: Literal["High School", "Bachelors", "Masters", "PhD"]
    internet_access: Literal["Yes", "No"]
    extracurricular_activities: Literal["Yes", "No"]
    part_time_job: Literal["Yes", "No"]
    previous_grade: float = Field(
        ..., ge=0, le=100, description="Previous grade, on a 0-100 scale."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Male",
                "study_time_hours": 4.0,
                "attendance_percent": 92.5,
                "sleep_hours": 7.0,
                "parental_education": "Bachelors",
                "internet_access": "Yes",
                "extracurricular_activities": "Yes",
                "part_time_job": "No",
                "previous_grade": 78.5,
            }
        }
    }


class PredictionResponse(BaseModel):
    success: bool
    prediction: float
    model: str


class RetrainResponse(BaseModel):
    success: bool
    message: str
    selected_model: str
    r2_scores: dict[str, float]
    rows_trained_on: int


class HomeResponse(BaseModel):
    message: str


@app.get("/", response_model=HomeResponse)
def home():
    return {
        "message": "Student Performance Prediction API is running."
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentData):
    try:
        result = prediction.predict_exam_score(student.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return {
        "success": True,
        "prediction": result["prediction"],
        "model": result["model"],
    }


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(
    file: UploadFile = File(
        ...,
        description=(
            "CSV with the same columns as the original training dataset "
            "(including final_exam_score). Triggers retraining and hot-reloads "
            "the model this API serves."
        ),
    )
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = train.retrain_from_csv(tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    prediction.reload_artifacts()

    return {
        "success": True,
        "message": "Model retrained and reloaded successfully.",
        "selected_model": result["model"],
        "r2_scores": result["scores"],
        "rows_trained_on": result["rows_trained_on"],
    }
