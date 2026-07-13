import argparse
import json
from pathlib import Path

import pandas as pd

from config import (
    APP_FEATURES,
    CATEGORICAL_FEATURES,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from preprocess import RISKY_PAYMENT_CODES, add_payment_risk, normalize_target


def load_dataset(path):
    df = pd.read_csv(path)
    missing = set(APP_FEATURES + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.drop_duplicates()
    df = add_payment_risk(df)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(normalize_target)
    return df


def build_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models(random_state):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    try:
        from xgboost import XGBClassifier
    except Exception:
        XGBClassifier = None

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=3,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    return models


def train(data_path, random_state=42):
    import joblib
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    df = load_dataset(data_path)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    results = {}
    best_name = None
    best_pipeline = None
    best_accuracy = -1.0

    for name, estimator in candidate_models(random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = {
            "accuracy": round(float(accuracy), 4),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                target_names=["Rejected", "Approved"],
                output_dict=True,
                zero_division=0,
            ),
        }
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_name = name
            best_pipeline = pipeline

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model_name": best_name,
        "pipeline": best_pipeline,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "app_features": APP_FEATURES,
        "risk_codes": sorted(RISKY_PAYMENT_CODES),
    }
    joblib.dump(artifact, MODEL_PATH)

    metrics = {
        "best_model": best_name,
        "best_accuracy": round(float(best_accuracy), 4),
        "models": results,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train credit card approval classifiers.")
    parser.add_argument("--data", required=True, type=Path, help="Path to training CSV")
    parser.add_argument("--random-state", default=42, type=int)
    args = parser.parse_args()

    metrics = train(args.data, args.random_state)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved best model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
