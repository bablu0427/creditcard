# EDA Guide

Use this guide as a lightweight notebook outline for credit card approval analysis.

## Suggested Cells

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/credit_card_applications.csv")
df.head()
```

```python
df.info()
df.describe(include="all")
df["approval_status"].value_counts(normalize=True)
```

## Count Plots

```python
for column in ["gender", "income_type", "education_level", "payment_status"]:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=column, hue="approval_status")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()
```

## Distribution Plots

```python
for column in ["annual_income", "employment_years", "existing_loan_balance", "credit_inquiries"]:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=column, hue="approval_status", kde=True)
    plt.tight_layout()
    plt.show()
```

## Correlation View

```python
numeric = df.select_dtypes("number")
sns.heatmap(numeric.corr(), annot=True, cmap="vlag", center=0)
plt.tight_layout()
plt.show()
```
