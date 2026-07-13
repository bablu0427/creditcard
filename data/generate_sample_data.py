from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_PATH = Path(__file__).resolve().parent / "credit_card_applications.csv"


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def main(rows=2500, seed=42):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Female", "Male", "Other"], rows, p=[0.48, 0.49, 0.03])
    income_type = rng.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
        rows,
        p=[0.48, 0.22, 0.15, 0.1, 0.05],
    )
    education_level = rng.choice(
        ["Secondary", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
        rows,
        p=[0.45, 0.33, 0.12, 0.07, 0.03],
    )
    annual_income = rng.lognormal(mean=11.0, sigma=0.45, size=rows).round(-2)
    employment_years = np.clip(rng.normal(6.5, 5.2, rows), 0, 40).round(1)
    existing_loan_balance = np.clip(rng.gamma(2.2, 6500, rows), 0, 120000).round(-2)
    credit_inquiries = rng.poisson(2.0, rows)
    past_due_count = rng.poisson(0.45, rows)
    payment_status = rng.choice(["0", "1", "2", "3", "late", "default"], rows, p=[0.72, 0.12, 0.07, 0.04, 0.03, 0.02])

    risk_flag = np.isin(payment_status, ["1", "2", "3", "late", "default"]).astype(int)
    education_bonus = np.isin(education_level, ["Higher education", "Academic degree"]).astype(float) * 0.35
    income_score = (annual_income - 55000) / 30000
    loan_pressure = existing_loan_balance / np.maximum(annual_income, 1)
    score = (
        -0.15
        + 0.85 * income_score
        + 0.08 * employment_years
        + education_bonus
        - 1.65 * loan_pressure
        - 0.28 * credit_inquiries
        - 0.75 * past_due_count
        - 1.25 * risk_flag
    )
    approval_probability = sigmoid(score)
    approval_status = np.where(rng.random(rows) < approval_probability, "Approved", "Rejected")

    df = pd.DataFrame(
        {
            "gender": gender,
            "income_type": income_type,
            "annual_income": annual_income,
            "employment_years": employment_years,
            "education_level": education_level,
            "existing_loan_balance": existing_loan_balance,
            "credit_inquiries": credit_inquiries,
            "past_due_count": past_due_count,
            "payment_status": payment_status,
            "approval_status": approval_status,
        }
    )
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} rows at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
