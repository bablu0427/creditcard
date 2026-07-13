# Runbook

## Local Demo

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python data/generate_sample_data.py
python train_model.py --data data/credit_card_applications.csv
python app/app.py
```

## Single Prediction from Python

```python
from predict import predict_applicant

result = predict_applicant({
    "gender": "Female",
    "income_type": "Working",
    "annual_income": 72000,
    "employment_years": 6,
    "education_level": "Higher education",
    "existing_loan_balance": 12000,
    "credit_inquiries": 2,
    "past_due_count": 0,
    "payment_status": "0",
})

print(result)
```

## Batch Screening

Upload a CSV through `/batch` or call `predict_batch(df)` from Python. The file must include the same applicant feature columns listed in `DATA_DICTIONARY.md`.
