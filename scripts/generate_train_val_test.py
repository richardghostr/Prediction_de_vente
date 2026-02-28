"""Generate 3 coherent CSV files: train, validation, test.

Temporal split:
  - Train:      2024-01-01 -> 2025-06-30  (18 months, ~9855 rows)
  - Validation:  2025-07-01 -> 2025-09-30  (3 months,  ~1380 rows)
  - Test:        2025-10-01 -> 2025-12-31  (3 months,  ~1380 rows)

Each (store_id, product_id) pair forms a continuous daily time series with:
  - Store-level baseline sales volume (5 stores)
  - Product-level price tier and demand multiplier (3 products)
  - Weekly seasonality (weekends higher)
  - Yearly seasonality (summer peak, winter dip)
  - Trend (+5%/year upward)
  - AR(1) autocorrelated noise
  - Random promotions (~15% of weeks)
  - Holiday effects (Christmas, New Year, Easter, summer sales)
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Output paths
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUT_DIR / "train.csv"
VAL_FILE   = OUT_DIR / "validation.csv"
TEST_FILE  = OUT_DIR / "test.csv"

np.random.seed(42)

# --- Parameters -----------------------------------------------------------
n_stores = 5
n_products = 3
start = pd.Timestamp("2024-01-01")
end   = pd.Timestamp("2025-12-31")
n_days = (end - start).days + 1  # 731 days

# Temporal split dates
TRAIN_END = pd.Timestamp("2025-06-30")
VAL_END   = pd.Timestamp("2025-09-30")
# TEST_END  = end (2025-12-31)

# Store baselines (units per day)
store_baselines = {
    "S001": 45.0,   # High-volume urban store
    "S002": 28.0,   # Medium suburban store
    "S003": 55.0,   # Large flagship store
    "S004": 18.0,   # Small rural store
    "S005": 35.0,   # Medium city-center store
}

# Product characteristics
product_info = {
    "P001": {"price": 12.50,  "demand_mult": 1.3,  "name": "Basic Widget"},
    "P002": {"price": 29.99,  "demand_mult": 0.8,  "name": "Premium Gadget"},
    "P003": {"price": 7.99,   "demand_mult": 1.6,  "name": "Economy Item"},
}

# Day-of-week multipliers (Mon=0 ... Sun=6)
dow_mult = np.array([0.82, 0.80, 0.88, 0.95, 1.08, 1.28, 1.19])

# Holidays (approximate dates that affect retail sales)
HOLIDAYS = {
    # (month, day): multiplier
    (1, 1): 0.40,    # New Year's Day (stores mostly closed)
    (1, 2): 0.60,    # Post-New Year
    (2, 14): 1.35,   # Valentine's Day
    (5, 1): 0.50,    # Labour Day (many stores closed in FR)
    (6, 21): 1.20,   # Fete de la musique / start of summer sales
    (7, 14): 0.55,   # Bastille Day
    (11, 29): 1.45,  # Black Friday (approximate)
    (12, 24): 1.50,  # Christmas Eve
    (12, 25): 0.30,  # Christmas Day
    (12, 26): 1.10,  # Post-Christmas sales
    (12, 31): 0.65,  # New Year's Eve
}


def get_holiday_mult(dt):
    """Return holiday multiplier for a given date."""
    key = (dt.month, dt.day)
    if key in HOLIDAYS:
        return HOLIDAYS[key]
    # Summer sales effect (mid-June to mid-July)
    if dt.month == 6 and dt.day >= 25:
        return 1.15
    if dt.month == 7 and dt.day <= 15:
        return 1.12
    return 1.0


# --- Generate data ---------------------------------------------------------
rows = []
for store_id, baseline in store_baselines.items():
    for prod_id, pinfo in product_info.items():
        # AR(1) noise state (different for each store-product pair)
        ar_state = 0.0
        ar_phi = 0.3 + 0.4 * hash((store_id, prod_id)) % 100 / 100  # between 0.3 and 0.7
        ar_sigma = baseline * 0.08  # noise proportional to baseline

        # Promo schedule: ~15% of weeks get a promotion
        total_weeks = n_days // 7 + 2
        n_promo_weeks = max(1, int(total_weeks * 0.15))
        rng = np.random.RandomState(hash((store_id, prod_id)) % 2**31)
        promo_weeks = set(rng.choice(range(total_weeks), size=n_promo_weeks, replace=False))

        for d in range(n_days):
            dt = start + pd.Timedelta(days=d)
            dow = dt.weekday()
            day_of_year = dt.dayofyear

            # Weekly effect
            weekly = dow_mult[dow]

            # Yearly seasonality: peak in summer (July), dip in January
            yearly = 1.0 + 0.22 * np.sin(2 * np.pi * (day_of_year - 30) / 365)

            # Slow trend (+5% per year)
            trend = 1.0 + 0.05 * (d / 365)

            # Holiday effect
            holiday = get_holiday_mult(dt)

            # Promo effect
            week_num = d // 7
            on_promo = 1 if week_num in promo_weeks else 0
            promo_boost = 1.0 + 0.30 * on_promo

            # AR(1) noise
            ar_state = ar_phi * ar_state + rng.normal(0, ar_sigma)

            # Final quantity
            qty = (baseline * pinfo["demand_mult"]
                   * weekly * yearly * trend * holiday * promo_boost
                   + ar_state)
            qty = max(0, round(qty))

            # Price with slight promo discount (10-20% off during promo)
            discount = rng.uniform(0.80, 0.90) if on_promo else 1.0
            price = round(pinfo["price"] * discount, 2)

            # Revenue = quantity * price
            revenue = round(qty * price, 2)

            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "store_id": store_id,
                "product_id": prod_id,
                "value": revenue,
                "quantity": int(qty),
                "price": price,
                "on_promo": on_promo,
            })

# Build DataFrame
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

print(f"Total rows generated: {len(df)}")
print(f"Unique (store, product) groups: {df.groupby(['store_id', 'product_id']).ngroups}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Value stats: mean={df['value'].mean():.2f}, std={df['value'].std():.2f}, "
      f"min={df['value'].min():.2f}, max={df['value'].max():.2f}")
print()

# --- Temporal split --------------------------------------------------------
train_df = df[df["date"] <= TRAIN_END].copy()
val_df   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].copy()
test_df  = df[df["date"] > VAL_END].copy()

# Convert date back to string for CSV
for split_df in [train_df, val_df, test_df]:
    split_df["date"] = split_df["date"].dt.strftime("%Y-%m-%d")

# --- Summary stats per split -----------------------------------------------
for name, split_df in [("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
    print(f"{name}:")
    print(f"  Rows: {len(split_df)}")
    print(f"  Date range: {split_df['date'].min()} -> {split_df['date'].max()}")
    print(f"  Value: mean={split_df['value'].mean():.2f}, std={split_df['value'].std():.2f}")
    print(f"  Promo rate: {split_df['on_promo'].mean()*100:.1f}%")
    print()

# --- Write to CSV ----------------------------------------------------------
train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print(f"Written: {TRAIN_FILE} ({len(train_df)} rows)")
print(f"Written: {VAL_FILE} ({len(val_df)} rows)")
print(f"Written: {TEST_FILE} ({len(test_df)} rows)")

# Also write the full combined file for backward compatibility with train.py
FULL_FILE = OUT_DIR / "generated_training_10950_clean.csv"
df_full = df.copy()
df_full["date"] = df_full["date"].dt.strftime("%Y-%m-%d") if df_full["date"].dtype != object else df_full["date"]
df_full.to_csv(FULL_FILE, index=False)
print(f"Written: {FULL_FILE} ({len(df_full)} rows, full dataset for train.py)")
