import sys
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, MODEL_PATH
from predict import predict_applicant, predict_batch


app = Flask(__name__)


def ensure_model_ready():
    if MODEL_PATH.exists():
        return
    try:
        from train_model import train

        train(DATA_DIR / "credit_card_applications.csv")
    except Exception as exc:
        app.logger.warning("Automatic model training failed: %s", exc)


ensure_model_ready()


def parse_float(name, default=0.0):
    value = request.form.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(name, default=0):
    value = request.form.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@app.route("/", methods=["GET"])
def home():
    model_ready = MODEL_PATH.exists()
    return render_template("index.html", model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict():
    applicant = {
        "gender": request.form.get("gender", "Unknown"),
        "income_type": request.form.get("income_type", "Working"),
        "annual_income": parse_float("annual_income"),
        "employment_years": parse_float("employment_years"),
        "education_level": request.form.get("education_level", "Secondary"),
        "existing_loan_balance": parse_float("existing_loan_balance"),
        "credit_inquiries": parse_int("credit_inquiries"),
        "past_due_count": parse_int("past_due_count"),
        "payment_status": request.form.get("payment_status", "0"),
    }
    try:
        result = predict_applicant(applicant)
        return render_template("result.html", applicant=applicant, result=result)
    except Exception as exc:
        return render_template("result.html", applicant=applicant, error=str(exc)), 400


@app.route("/batch", methods=["GET", "POST"])
def batch():
    rows = None
    error = None
    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            error = "Please upload a CSV file."
        else:
            try:
                df = pd.read_csv(file)
                scored = predict_batch(df)
                rows = scored.head(50).to_dict(orient="records")
            except Exception as exc:
                error = str(exc)
    return render_template("batch.html", rows=rows, error=error)


if __name__ == "__main__":
    app.run(debug=True)
