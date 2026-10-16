"""Synthetic monthly bank balance-sheet data for practising Isolation Forest.
Creates:
  bank_balance_sheet.csv  -> bank, month, item, balance_bn   (what you work with)
  answer_key.csv          -> the injected anomalies           (don't open until the end)
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
months = pd.date_range("2019-01-31", periods=72, freq="ME")   # 6 years, month-end

# name: (starting balance in LKR bn, avg monthly growth, monthly noise std)
banks = {"Bank_A": 2000, "Bank_B": 600, "Bank_C": 150}   # very different sizes on purpose
items = {
    "Total_Assets": (1.00, 0.006, 0.008),
    "Loans":        (0.60, 0.007, 0.010),
    "Deposits":     (0.75, 0.006, 0.009),
    "Borrowings":   (0.10, 0.003, 0.030),
}

rows = []
for bank, size in banks.items():
    for item, (share, growth, noise) in items.items():
        level = size * share
        for i, m in enumerate(months):
            seasonal = 0.012 if m.month == 12 else (-0.008 if m.month == 1 else 0)  # year-end bump, Jan dip
            level *= 1 + growth + seasonal + rng.normal(0, noise)
            rows.append((bank, m, item, level))

df = pd.DataFrame(rows, columns=["bank", "month", "item", "balance_bn"])

# Inject anomalies as multiplicative shocks that persist (a real jump in the balance)
anoms = [
    ("Bank_A", "Deposits",     "2021-03-31", 1.12, "sudden deposit surge"),
    ("Bank_A", "Borrowings",   "2022-07-31", 1.60, "large borrowing"),
    ("Bank_A", "Loans",        "2023-04-30", 0.93, "loan book drop"),
    ("Bank_B", "Total_Assets", "2020-04-30", 1.08, "pandemic-style jump"),
    ("Bank_B", "Deposits",     "2022-10-31", 0.90, "deposit outflow"),
    ("Bank_B", "Borrowings",   "2024-02-29", 0.50, "debt repayment"),
    ("Bank_C", "Loans",        "2021-09-30", 1.15, "loan spike"),
    ("Bank_C", "Deposits",     "2023-11-30", 0.88, "deposit run"),
    ("Bank_C", "Borrowings",   "2020-06-30", 1.80, "emergency borrowing"),
]
key = []
for bank, item, date, mult, why in anoms:
    mask = (df.bank == bank) & (df.item == item) & (df.month >= date)
    df.loc[mask, "balance_bn"] *= mult
    key.append((bank, item, date, why))

df["balance_bn"] = df["balance_bn"].round(2)
df.to_csv("bank_balance_sheet.csv", index=False)
pd.DataFrame(key, columns=["bank", "item", "month", "what"]).to_csv("answer_key.csv", index=False)
print(df.shape, "rows written;", len(key), "anomalies hidden")
