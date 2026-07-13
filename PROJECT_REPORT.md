# Project Report

## Problem Statement

Banks and financial institutions receive large volumes of credit card applications. Manual screening is slow, inconsistent, and difficult to scale. This project automates approval prediction with machine learning so analysts, compliance officers, and customers can receive fast eligibility guidance.

## Objectives

- Analyze applicant financial and demographic profiles.
- Engineer clean model-ready features from payment and credit history.
- Train and compare multiple classification algorithms.
- Save the best-performing model for real-time prediction.
- Provide a Flask web interface for single and batch applicant screening.
- Prepare a cloud deployment scaffold for IBM Watson Machine Learning.

## Algorithms Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier when installed
- Scikit-learn Gradient Boosting fallback

## Feature Engineering

The multi-class `payment_status` field is converted into a binary `payment_risk` feature:

- `low_risk`: current or clean repayment behavior
- `risky`: overdue, late, default, or bad-debt status

This supports compliance-oriented screening by making high-risk repayment behavior explicit for the classifier.

## Evaluation Metrics

The training pipeline reports:

- Accuracy score
- Confusion matrix
- Precision, recall, and F1-score
- Best model by validation accuracy

## Deployment Flow

1. Train and save the best model artifact with `joblib`.
2. Load the same preprocessing/model pipeline in Flask.
3. Accept applicant inputs from the UI or uploaded CSV files.
4. Return approval/rejection predictions and approval probabilities.
5. Optionally upload the artifact to IBM Watson Machine Learning for online scoring.

## Ethical and Production Considerations

This is a learning project. Production credit decision systems require fairness audits, explainability, adverse-action reason codes, model monitoring, privacy controls, human review, and compliance with financial regulations.
