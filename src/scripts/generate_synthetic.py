"""Generate realistic synthetic retail sales data (~11k rows).

Each (store_id, product_id) combination forms a continuous daily time series
with coherent temporal patterns:
  - Store-level baseline sales volume
  - Product-level price tier and demand multiplier
  - Weekly seasonality (weekends vs weekdays)
  - Yearly seasonality (summer peak, winter dip)
  - Slow upward trend
  - Autocorrelated noise (AR(1) process)
  - Occasional promotions that boost sales

Writes to `data/interim/generated_training_10950_clean.csv`.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/interim/generated_training_10950_clean.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# Parameters
n_stores = 5
n_products = 3
n_days = 730  # 2 years for enough data
start = pd.Timestamp("2025-01-01")

# Store baselines (units per day)
store_baselines = {
    f"S{s:03d}": np.random.uniform(15, 60)
    for s in range(1, n_stores + 1)
}

# Product characteristics
product_info = {
    f"P{p:03d}": {
        "price": round(np.random.uniform(5, 50), 2),
        "demand_mult": np.random.uniform(0.5, 1.5),
    }
    for p in range(1, n_products + 1)
}

# Day-of-week multipliers (Mon=0 ... Sun=6)
# Retail: higher on weekends, dip on Mon/Tue
dow_mult = np.array([0.85, 0.82, 0.90, 0.95, 1.05, 1.25, 1.18])

rows = []
for store_id, baseline in store_baselines.items():
    for prod_id, pinfo in product_info.items():
        # AR(1) noise state
        ar_state = 0.0
        ar_phi = np.random.uniform(0.3, 0.7)  # autocorrelation coefficient
        ar_sigma = baseline * 0.1  # noise proportional to baseline

        # Promo schedule: random weeks with promotions
        promo_weeks = set(np.random.choice(range(104), size=15, replace=False))

        for d in range(n_days):
            dt = start + pd.Timedelta(days=d)
            date_str = dt.strftime("%Y-%m-%d")
            dow = dt.weekday()
            day_of_year = dt.dayofyear

            # Weekly effect
            weekly = dow_mult[dow]

            # Yearly seasonality: peak in summer (July), dip in January
            yearly = 1.0 + 0.20 * np.sin(2 * np.pi * (day_of_year - 30) / 365)

            # Slow trend (+5% per year)
            trend = 1.0 + 0.05 * (d / 365)

            # Promo effect
            week_num = d // 7
            on_promo = 1 if week_num in promo_weeks else 0
            promo_boost = 1.0 + 0.30 * on_promo  # +30% lift during promo

            # AR(1) noise
            ar_state = ar_phi * ar_state + np.random.normal(0, ar_sigma)

            # Final quantity
            qty = baseline * pinfo["demand_mult"] * weekly * yearly * trend * promo_boost + ar_state
            qty = max(0, round(qty))

            # Price with slight promo discount
            price = pinfo["price"] * (0.85 if on_promo else 1.0)

            rows.append({
                "date": date_str,
                "store_id": store_id,
                "product_id": prod_id,
                "value": round(qty * price, 2),  # revenue as target
                "quantity": int(qty),
                "price": round(price, 2),
                "on_promo": on_promo,
            })

df = pd.DataFrame(rows)
print(f"Generated rows: {len(df)}")
print(f"Unique (store, product) groups: {df.groupby(['store_id', 'product_id']).ngroups}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Value stats: mean={df['value'].mean():.2f}, std={df['value'].std():.2f}")

# Sort by store, product, date for proper temporal ordering
df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
df.to_csv(OUT, index=False)
print(f"Written synthetic clean CSV to {OUT} (rows={len(df)})")
