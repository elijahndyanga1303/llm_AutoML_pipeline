"""Creates sample_data/customers.csv (synthetic churn dataset) for trying the system."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1500
tenure = rng.integers(1, 72, n)
monthly = rng.normal(70, 25, n).clip(15, 150).round(2)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
support_calls = rng.poisson(1.5, n)
internet = rng.choice(["Fiber", "DSL", "None"], n, p=[0.45, 0.4, 0.15])
logit = (-1.0 - 0.04 * tenure + 0.015 * monthly + 0.35 * support_calls
         + np.where(contract == "Month-to-month", 1.0, -0.6))
churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame({
    "customer_id": [f"C{i:05d}" for i in range(n)],
    "tenure_months": tenure, "monthly_charges": monthly, "contract": contract,
    "internet_service": internet, "support_calls": support_calls,
    "total_charges": (tenure * monthly).round(2), "churn": churn,
})
df.loc[rng.choice(n, 40, replace=False), "total_charges"] = np.nan
df.to_csv("sample_data/customers.csv", index=False)
print(df.shape, "churn rate:", round(churn.mean(), 3))
