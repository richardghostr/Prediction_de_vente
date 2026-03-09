"""Unified prediction loader & helper.

Exposes `predict(series, horizon, model_name)` which will attempt to load
the requested model from `models/artifacts` (latest by timestamp) and return
predictions in a unified format.
"""
from pathlib import Path
import joblib
import json
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

ARTIFACTS = Path(__file__).resolve().parents[2] / 'models' / 'artifacts'


def _find_latest(model_tag: str) -> Optional[Path]:
    """Find the latest artifact corresponding to a model tag.

    The project notebooks may emit artifacts with different naming
    conventions (e.g. `lgbm_model_*.pkl`, `xgb_model_*.pkl`, or
    `model_2026_lgbm.pkl`). Be permissive and search for any .pkl
    that contains the tag keywords.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    all_pkls = sorted(ARTIFACTS.glob('*.pkl'))
    if not all_pkls:
        return None

    # map friendly tag -> list of keywords to look for
    keywords_map = {
        'LightGBM': ['lgbm', 'lgb', 'lightgbm'],
        'XGBoost': ['xgb', 'xg', 'xgboost'],
        'Prophet': ['prophet'],
        'Auto': [],
    }

    kws = keywords_map.get(model_tag, [model_tag.lower()])
    if model_tag == 'Auto' or not kws:
        # prefer files named with 'model_' prefix if available
        pref = [p for p in all_pkls if p.name.lower().startswith('model_')]
        if pref:
            return pref[-1]
        return all_pkls[-1]

    # find matches containing any keyword
    matches = [p for p in all_pkls if any(kw in p.name.lower() for kw in kws)]
    if matches:
        return matches[-1]

    # fallback: newest pkl
    return all_pkls[-1]


def load_model_for(tag: str) -> Dict[str, Any]:
    mp = _find_latest(tag)
    if mp is None:
        raise FileNotFoundError('No model artifact found')
    model = joblib.load(mp)
    cfg = {}
    cfg_path = mp.with_name(mp.name + '_config.json')
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    return {'model': model, 'path': str(mp), 'config': cfg}


def _align_features(X: pd.DataFrame, model) -> pd.DataFrame:
    # If model exposes feature names, ensure columns exist and ordering
    expected = None
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        try:
            expected = model.get_booster().feature_names
        except Exception:
            expected = None

    out = X.copy()
    if expected is None:
        return out
    for c in expected:
        if c not in out.columns:
            out[c] = 0
    extra = [c for c in out.columns if c not in expected]
    if extra:
        out = out.drop(columns=extra)
    return out[expected]


def predict(series: List[Dict[str, Any]], horizon: int = 7, model_name: str = 'Auto') -> Dict[str, Any]:
    """Unified predict function.

    series: list of dicts with 'date' and optional 'value' and extras.
    model_name: 'LightGBM'|'XGBoost'|'Prophet'|'Auto'
    """
    if not series:
        raise ValueError('Empty series')

    # find and load model
    tag = model_name if model_name in ('LightGBM','XGBoost','Prophet') else 'Auto'
    info = load_model_for(tag)
    model = info['model']
    cfg = info.get('config', {})

    df = pd.DataFrame(series)
    df['date'] = pd.to_datetime(df['date']).dt.floor('D')

    # Prophet branch
    if 'prophet' in str(type(model)).lower() or 'Prophet' in type(model).__name__:
        # build history
        hist = df[['date']].copy()
        hist['y'] = df.get('value') if 'value' in df.columns else pd.NA
        hist = hist.groupby('date', as_index=False).agg({'y':'mean'})
        last = hist['date'].max()
        future = pd.DataFrame({'ds': pd.date_range(start=last + pd.Timedelta(days=1), periods=horizon, freq='D')})
        preds = model.predict(future)
        out = []
        for ds, yhat in zip(preds['ds'], preds['yhat']):
            out.append({'ds': pd.to_datetime(ds).strftime('%Y-%m-%d'), 'yhat': float(yhat)})
        return {'forecast': out, 'model_info': {'path': info['path']}}

    # Generic ML model branch: iterative autoregressive predicting
    history = df.copy().sort_values('date').reset_index(drop=True)
    last_val = None
    for v in reversed(history.get('value', []).tolist()):
        if pd.notna(v):
            last_val = float(v)
            break
    if last_val is None:
        last_val = 0.0

    # If feature builder available, try to use it
    try:
        from src.data.features import build_feature_pipeline
        build_feats = build_feature_pipeline
    except Exception:
        build_feats = None

    forecast = []
    for step in range(1, horizon+1):
        if build_feats is not None:
            try:
                feats, _ = build_feats(history, is_train=False)
                exclude = set(['date','value'])
                X = feats.drop(columns=[c for c in feats.columns if c in exclude], errors='ignore')
                X_last = X.iloc[[-1]].fillna(0)
                X_last = _align_features(X_last, model)
                raw = float(model.predict(X_last)[0])
                yhat = float(raw)
            except Exception:
                yhat = float(last_val)
        else:
            yhat = float(last_val)

        next_date = history['date'].iloc[-1] + pd.Timedelta(days=1)
        forecast.append({'ds': next_date.strftime('%Y-%m-%d'), 'yhat': round(yhat,2)})

        # append to history for next step
        new_row = {'date': next_date, 'value': yhat}
        # keep group columns if present
        for c in ('store_id','product_id','on_promo','price'):
            if c in history.columns:
                new_row[c] = history[c].iloc[-1]
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    return {'forecast': forecast, 'model_info': {'path': info['path']}}


def list_models() -> Dict[str, Optional[str]]:
    # return latest model paths per supported tag
    out = {}
    for tag in ('LightGBM','XGBoost','Prophet'):
        p = _find_latest(tag)
        out[tag] = str(p) if p else None
    return out
