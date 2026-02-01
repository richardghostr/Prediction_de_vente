
import pandas as pd
from pathlib import Path
import joblib
from math import sqrt
import sys

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parents[2] 
sys.path.insert(0, str(ROOT_DIR))

from src.data.features import build_feature_pipeline


FEATURES_PATH = ROOT_DIR / "data/processed/sample_features.csv"
ARTIFACTS_PATH = ROOT_DIR / "models/artifacts"
FORECAST_PATH = ARTIFACTS_PATH / "xgb_forecast.csv"

model_files = sorted(ARTIFACTS_PATH.glob("model_*_baseline.pkl"))
if not model_files:
    raise FileNotFoundError("Aucun modèle trouvé dans models/artifacts/")
MODEL_PATH = model_files[-1]
model = joblib.load(MODEL_PATH)

df = pd.read_csv(FEATURES_PATH, parse_dates=['date'])
df = build_feature_pipeline(df, lags=[1,7], windows=[3,7])
df = df.dropna()

X = df.drop(columns=['value','date'])
y_true = df['value']


y_pred = model.predict(X)
df_forecast = df.reset_index()
df_forecast['yhat'] = y_pred


print("[INFO] Forecast head:")
print(df_forecast[['date','value','yhat']].head())


df_forecast.to_csv(FORECAST_PATH, index=False)
print(f"[INFO] Forecast saved: {FORECAST_PATH}")
