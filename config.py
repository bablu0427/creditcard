from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_credit_card_model.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

TARGET_COLUMN = "approval_status"

CATEGORICAL_FEATURES = [
    "gender",
    "income_type",
    "education_level",
    "payment_risk",
]

NUMERIC_FEATURES = [
    "annual_income",
    "employment_years",
    "existing_loan_balance",
    "credit_inquiries",
    "past_due_count",
]

APP_FEATURES = [
    "gender",
    "income_type",
    "annual_income",
    "employment_years",
    "education_level",
    "existing_loan_balance",
    "credit_inquiries",
    "past_due_count",
    "payment_status",
]
