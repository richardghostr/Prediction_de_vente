import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json


def _find_forecast_files():
    # common names to search
    names = ['combined_forecast.csv', 'forecast.csv', 'forecasts.csv']
    candidates = []
    for n in names:
        candidates += list(Path('.').glob(f'**/{n}'))
    # also include any csv under models/ or outputs/
    candidates += list(Path('models').glob('**/*.csv')) if Path('models').exists() else []
    return candidates


def _normalize_date_column(df: pd.DataFrame):
    # choose a date-like column among common candidates
    date_candidates = [c for c in df.columns if c.lower() in ('ds', 'date', 'day', 'timestamp') or 'date' in c.lower()]
    if date_candidates:
        col = date_candidates[0]
        try:
            df[col] = pd.to_datetime(df[col])
            return df, col
        except Exception:
            return df, None
    # try to infer by dtype
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors='coerce')
            if parsed.notna().mean() > 0.3:
                df[c] = parsed
                return df, c
        except Exception:
            continue
    return df, None


def _detect_y_columns(df: pd.DataFrame):
    y_candidates = [c for c in df.columns if c.lower() in ('y', 'y_true', 'value', 'réel', 'real')]
    yhat_candidates = [c for c in df.columns if 'yhat' in c.lower() or 'pred' in c.lower() or 'forecast' in c.lower()]
    return (y_candidates[0] if y_candidates else None), (yhat_candidates[0] if yhat_candidates else None)


def render_visualize_ui():
    st.header("Visualisation des résultats")
    st.write("Visualisez un CSV de prévision ou uploadez votre fichier. L'interface tente de détecter automatiquement les colonnes date / réel / prédit.")

    uploaded = st.file_uploader("Uploader un fichier de prévision (CSV)", type=["csv"])
    df = None
    loaded_from = None

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            loaded_from = 'uploaded'
        except Exception as e:
            st.error(f"Impossible de lire le CSV uploadé: {e}")
            return
    else:
        candidates = _find_forecast_files()
        if candidates:
            # prefer combined_forecast if present
            pick = sorted(candidates, key=lambda p: 'combined' not in p.name)[0]
            try:
                df = pd.read_csv(pick)
                loaded_from = str(pick)
                st.info(f"Chargé: {pick}")
            except Exception as e:
                st.error(f"Impossible de lire {pick}: {e}")
                return
        else:
            st.info("Aucun fichier de prévision trouvé. Exécutez la prédiction ou uploadez un fichier CSV.")
            return

    # show quick diagnostics
    st.subheader("Diagnostics du fichier")
    st.write(f"Source: {loaded_from}")
    st.write("Colonnes détectées:")
    st.write(list(df.columns))
    st.write("Aperçu: ")
    st.dataframe(df.head())

    # If file stores JSON in a 'forecast' column or nested structure, try to normalize
    if 'forecast' in df.columns and df['forecast'].dtype == object:
        try:
            # expand if first cell looks like JSON list/dict
            sample = df['forecast'].dropna().iloc[0]
            parsed = json.loads(sample) if isinstance(sample, str) else sample
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                # build dataframe from list of dicts
                df = pd.DataFrame(parsed)
                st.info('Fichier normalisé depuis colonne `forecast`.')
        except Exception:
            pass

    df, date_col = _normalize_date_column(df)
    if date_col is None:
        st.error("Impossible d'identifier une colonne de date. Assurez-vous qu'il existe une colonne 'ds' ou 'date'.")
        return

    y_col, yhat_col = _detect_y_columns(df)

    fig = go.Figure()
    # historical
    if y_col and y_col in df.columns:
        hist = df[[date_col, y_col]].dropna().groupby(date_col, as_index=False).sum()
        fig.add_trace(go.Scatter(x=hist[date_col], y=hist[y_col], mode='lines', name='Historique'))

    # forecast
    if yhat_col and yhat_col in df.columns:
        fc = df[[date_col, yhat_col]].dropna()
        fig.add_trace(go.Scatter(x=fc[date_col], y=fc[yhat_col], mode='lines', name='Prédiction'))
    else:
        # try common alternative 'yhat' or 'pred' columns
        alt = [c for c in df.columns if 'yhat' in c.lower() or 'pred' in c.lower() or 'forecast' in c.lower()]
        if alt:
            fc = df[[date_col, alt[0]]].dropna()
            fig.add_trace(go.Scatter(x=fc[date_col], y=fc[alt[0]], mode='lines', name=f'Prédiction ({alt[0]})'))

    fig.update_layout(title='Visualisation des prévisions', xaxis_title='Date', yaxis_title='Valeur')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Tableau de la prédiction')
    st.dataframe(df.head())
