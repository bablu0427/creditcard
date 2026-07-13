import numpy as np


RISKY_PAYMENT_CODES = {"1", "2", "3", "4", "5", "late", "past_due", "default", "bad_debt"}


def normalize_target(value):
    text = str(value).strip().lower()
    if text in {"1", "approved", "approve", "yes", "y", "eligible"}:
        return 1
    if text in {"0", "rejected", "reject", "no", "n", "ineligible"}:
        return 0
    raise ValueError(f"Unsupported approval_status value: {value}")


def add_payment_risk(df):
    df = df.copy()
    payment_status = df["payment_status"].fillna("unknown").astype(str).str.strip().str.lower()
    df["payment_risk"] = np.where(payment_status.isin(RISKY_PAYMENT_CODES), "risky", "low_risk")
    return df
