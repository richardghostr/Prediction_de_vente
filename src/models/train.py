

import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
import datetime
import sys
from math import sqrt


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.features import build_feature_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = PROJECT_ROOT / "data/processed/sample_features.csv"
ARTIFACTS_PATH = PROJECT_ROOT / "models/artifacts"
METRICS_PATH = PROJECT_ROOT / "models/metrics"

ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
METRICS_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURES_PATH, parse_dates=['date'])

df = build_feature_pipeline(df, lags=[1, 7], windows=[3, 7])

df = df.dropna()


y = df['value']
X = df.drop(columns=['value', 'date'])


model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X, y)


y_pred_train = model.predict(X)
mae_train = mean_absolute_error(y, y_pred_train)
rmse_train = sqrt(mean_squared_error(y, y_pred_train))

tscv = TimeSeriesSplit(n_splits=5)
cv_mae = []

for train_idx, val_idx in tscv.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model_cv = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model_cv.fit(X_tr, y_tr)
    y_val_pred = model_cv.predict(X_val)
    cv_mae.append(mean_absolute_error(y_val, y_val_pred))

metrics = {
    "MAE_train": float(mae_train),
    "RMSE_train": float(rmse_train),
    "CV_mae_mean": float(pd.Series(cv_mae).mean()),
    "CV_mae_folds": cv_mae,
    "num_samples": len(y)
}

today = datetime.datetime.now().strftime("%Y%m%d")
model_file = ARTIFACTS_PATH / f"model_{today}_baseline.pkl"
metrics_file = METRICS_PATH / f"xgb_metrics_{today}.json"

joblib.dump(model, model_file)
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"[INFO] Model saved: {model_file}")
print(f"[INFO] Metrics saved: {metrics_file}")
print(f"[INFO] MAE_train: {mae_train:.4f}, RMSE_train: {rmse_train:.4f}, CV_mae_mean: {float(pd.Series(cv_mae).mean()):.4f}")
