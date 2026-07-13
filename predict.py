import joblib
import pandas as pd

from config import MODEL_PATH
from preprocess import add_payment_risk


def load_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run train_model.py first."
        )
    return joblib.load(model_path)


def predict_applicant(applicant, model_path=MODEL_PATH):
    artifact = load_model(model_path)
    df = pd.DataFrame([applicant])
    df = add_payment_risk(df)
    features = artifact["features"]
    probability = artifact["pipeline"].predict_proba(df[features])[0][1]
    label = int(probability >= 0.5)
    return {
        "prediction": "Approved" if label == 1 else "Rejected",
        "approval_probability": round(float(probability), 4),
        "model_name": artifact["model_name"],
    }


def predict_batch(df, model_path=MODEL_PATH):
    artifact = load_model(model_path)
    prepared = add_payment_risk(df)
    features = artifact["features"]
    probabilities = artifact["pipeline"].predict_proba(prepared[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    output = df.copy()
    output["approval_probability"] = probabilities.round(4)
    output["prediction"] = ["Approved" if item == 1 else "Rejected" for item in predictions]
    return output
