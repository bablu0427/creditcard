# Data Dictionary

| Column | Type | Description |
| --- | --- | --- |
| `gender` | Categorical | Applicant gender category. |
| `income_type` | Categorical | Employment or income source, such as working, pensioner, student, or state servant. |
| `annual_income` | Numeric | Annual income amount. |
| `employment_years` | Numeric | Number of years employed or financially active. |
| `education_level` | Categorical | Highest education level reported. |
| `existing_loan_balance` | Numeric | Current outstanding loan balance. |
| `credit_inquiries` | Numeric | Recent credit inquiry count. |
| `past_due_count` | Numeric | Count of past-due loan records. |
| `payment_status` | Categorical | Raw payment history status code. Risky values include overdue, late, default, and bad debt states. |
| `payment_risk` | Engineered categorical | Binary label generated from `payment_status`: `low_risk` or `risky`. |
| `approval_status` | Target | Historical application result: `Approved` or `Rejected`. |
