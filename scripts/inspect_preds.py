from pathlib import Path
import joblib, json
import pandas as pd, numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import build_feature_pipeline
from src.models.predict import _align_features_to_model
from src.config import FEATURE_EXCLUDE

mp = sorted((PROJECT_ROOT / 'models' / 'artifacts').glob('model_*.pkl'))[-1]
print('model', mp)
model = joblib.load(mp)
cfg_path = mp.parent / (mp.stem + '_config.json')
cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
print('cfg keys', list(cfg.keys()))

feat = Path(cfg.get('training_features_path') or PROJECT_ROOT / 'data' / 'processed' / 'train_features.csv')
print('feat exists', feat.exists(), feat)
df = pd.read_csv(feat, parse_dates=['date'])
res = build_feature_pipeline(df, lags=cfg.get('lags'), windows=cfg.get('windows'), encoders=cfg.get('encoders'), is_train=False)
if isinstance(res, tuple):
    df_feat, _ = res
else:
    df_feat = res
print('feat shape before dropna', df_feat.shape)
df_feat = df_feat.dropna().reset_index(drop=True)
print('feat shape after dropna', df_feat.shape)
horizon = int(cfg.get('horizon', 14))
unique_dates = sorted(df_feat['date'].unique())
cut = unique_dates[-horizon]
print('cut', cut)
mask = df_feat['date'] >= cut
exclude = set(FEATURE_EXCLUDE) | {'value', 'date'}
feature_cols = [c for c in df_feat.columns if c not in exclude]
X_test = df_feat.loc[mask, feature_cols].copy()
y_test = df_feat.loc[mask, 'value'].copy()
print('X_test shape', X_test.shape, 'y_test shape', y_test.shape)
X_aligned = _align_features_to_model(X_test, model)
print('X_aligned shape', X_aligned.shape)
for c in X_aligned.columns:
    if X_aligned[c].dtype == 'object':
        X_aligned[c] = pd.Categorical(X_aligned[c]).codes
X_aligned = X_aligned.fillna(0)
print('dtypes sample')
print(X_aligned.dtypes.head(10))
try:
    y_pred = model.predict(X_aligned)
except Exception as e:
    print('predict failed', e)
    raise

print('y_pred min/max/mean:', np.min(y_pred), np.max(y_pred), np.mean(y_pred))
print('y_test min/max/mean:', y_test.min(), y_test.max(), y_test.mean())
print('sample X rows:', X_aligned.head(3).to_dict(orient='records'))
print('sample preds:', y_pred[:5])
