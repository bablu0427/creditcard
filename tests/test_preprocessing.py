import pandas as pd

from preprocess import add_payment_risk, normalize_target


def test_payment_status_is_converted_to_binary_risk_label():
    df = pd.DataFrame({"payment_status": ["0", "1", "late", "default", "C"]})
    output = add_payment_risk(df)
    assert output["payment_risk"].tolist() == [
        "low_risk",
        "risky",
        "risky",
        "risky",
        "low_risk",
    ]


def test_approval_status_normalization():
    assert normalize_target("Approved") == 1
    assert normalize_target("Rejected") == 0
    assert normalize_target("yes") == 1
    assert normalize_target("no") == 0
