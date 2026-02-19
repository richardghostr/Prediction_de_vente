"""Generate synthetic retail sales data (~11k rows).
Writes a cleaned CSV suitable for the pipeline to `data/interim/generated_training_10950_clean.csv`.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/interim/generated_training_10950_clean.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Parameters
n_stores = 30
n_days = 365  # one year
start = pd.Timestamp("2025-01-01")

rows = []
for store in range(1, n_stores + 1):
    # store-level baseline and seasonality
    baseline = np.random.randint(5, 50)
    weekly_amp = np.random.uniform(0.2, 1.5)
    monthly_amp = np.random.uniform(0.1, 0.8)
    for d in range(n_days):
        date = (start + pd.Timedelta(days=d)).date()
        # weekly effect (dow)
        dow = (start + pd.Timedelta(days=d)).weekday()
        dow_effect = (1 + weekly_amp * (1 if dow >=5 else 0))
        # monthly effect via sin
        day_of_year = (start + pd.Timedelta(days=d)).dayofyear
        yearly = 1 + monthly_amp * np.sin(2 * np.pi * day_of_year / 365)
        # trend slowly increasing
        trend = 1 + 0.001 * d
        # random noise
        noise = np.random.normal(0, 5)
        value = max(0, baseline * dow_effect * yearly * trend + noise)
        rows.append({
            "date": date.isoformat(),
            "store_id": f"S{store:03d}",
            "product_id": f"P{np.random.randint(1,20):03d}",
            "quantity": int(max(0, round(value))),
            "price": round(np.random.uniform(5, 100), 2),
        })

# Expand to DataFrame and compute revenue
df = pd.DataFrame(rows)
# ensure > 10k rows
print(f"Generated rows: {len(df)}")
# add revenue column
df["revenue"] = (df["quantity"] * df["price"]).round(2)
# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# Save as cleaned CSV suitable for feature pipeline (columns: date, value)
# Choose 'value' as revenue
out_df = df.rename(columns={"revenue": "value"})[["date", "value", "store_id", "product_id", "quantity", "price"]]
out_df.to_csv(OUT, index=False)
print(f"Written synthetic clean CSV to {OUT} (rows={len(out_df)})")
