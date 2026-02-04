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
            sample = df['forecast'].dropna().iloc[0]
            parsed = json.loads(sample) if isinstance(sample, str) else sample
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                df = pd.DataFrame(parsed)
                st.info('Fichier normalisé depuis colonne `forecast`.')
        except Exception:
            pass

    df, date_col = _normalize_date_column(df)
    if date_col is None:
        st.error("Impossible d'identifier une colonne de date. Assurez-vous qu'il existe une colonne 'ds' ou 'date'.")
        return

    # detect columns
    y_col, yhat_col = _detect_y_columns(df)
    upper_col = next((c for c in df.columns if 'upper' in c.lower()), None)
    lower_col = next((c for c in df.columns if 'lower' in c.lower()), None)

    # offer user to upload historical series if not present
    hist_df = None
    if not y_col:
        st.info("Aucune colonne historique détectée. Vous pouvez uploader un fichier historique (clean) pour comparaison.)")
        uploaded_hist = st.file_uploader("Uploader CSV historique (optionnel)", type=["csv"], key="hist_upload")
        if uploaded_hist is not None:
            try:
                hist_df = pd.read_csv(uploaded_hist)
                hist_df[date_col] = pd.to_datetime(hist_df[date_col])
                for c in ('value', 'y', 'réel'):
                    if c in hist_df.columns:
                        y_col = c
                        break
            except Exception as e:
                st.warning(f"Impossible de lire l'historique uploadé: {e}")
    else:
        hist_df = df[[date_col, y_col]].dropna().copy()

    # forecast df
    if yhat_col and yhat_col in df.columns:
        fc_df = df[[date_col, yhat_col]].dropna().copy()
        fc_df = fc_df.rename(columns={yhat_col: 'yhat'})
    else:
        alt = [c for c in df.columns if 'yhat' in c.lower() or 'pred' in c.lower() or c.lower().startswith('forecast')]
        if alt:
            fc_df = df[[date_col, alt[0]]].dropna().rename(columns={alt[0]: 'yhat'})
        else:
            fc_df = pd.DataFrame(columns=[date_col, 'yhat'])

    # if upper/lower present, bring them
    if upper_col and upper_col in df.columns:
        df[upper_col] = pd.to_numeric(df[upper_col], errors='coerce')
        if upper_col in df.columns:
            fc_df = fc_df.merge(df[[date_col, upper_col]], on=date_col, how='left')
            fc_df = fc_df.rename(columns={upper_col: 'yhat_upper'})
    if lower_col and lower_col in df.columns:
        df[lower_col] = pd.to_numeric(df[lower_col], errors='coerce')
        if lower_col in df.columns:
            fc_df = fc_df.merge(df[[date_col, lower_col]], on=date_col, how='left')
            fc_df = fc_df.rename(columns={lower_col: 'yhat_lower'})

    # Combine historical and forecast for plotting and metrics
    combined = pd.merge(hist_df.rename(columns={date_col: 'ds', y_col: 'y'}) if hist_df is not None else pd.DataFrame(columns=['ds','y']),
                        fc_df.rename(columns={date_col: 'ds'}) if not fc_df.empty else pd.DataFrame(columns=['ds','yhat']),
                        on='ds', how='outer')
    if not combined.empty:
        combined['ds'] = pd.to_datetime(combined['ds'])

    # Controls: date range, aggregation level, anomaly threshold
    st.sidebar.subheader('Affichage')
    if not combined.empty:
        min_date = combined['ds'].min().date()
        max_date = combined['ds'].max().date()
    else:
        min_date = None
        max_date = None
    date_range = st.sidebar.date_input('Période', value=(min_date, max_date) if min_date and max_date else None)
    agg_level = st.sidebar.selectbox('Granularité', options=['D', 'W', 'M'], index=0, format_func=lambda x: {'D':'Journalier','W':'Hebdo','M':'Mensuel'}[x])
    anomaly_pct = st.sidebar.slider('Seuil d\'écart (%) pour alerte si pas d\'intervalle', 0.0, 500.0, 50.0)

    plot_df = combined.copy()
    if date_range and isinstance(date_range, (list, tuple)) and date_range[0] and date_range[1]:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        plot_df = plot_df[(plot_df['ds'] >= start) & (plot_df['ds'] <= end)]

    if not plot_df.empty and agg_level:
        if agg_level == 'D':
            plot_resampled = plot_df.set_index('ds')
        elif agg_level == 'W':
            plot_resampled = plot_df.set_index('ds').resample('W').sum()
        else:
            plot_resampled = plot_df.set_index('ds').resample('M').sum()
        plot_resampled = plot_resampled.reset_index()
    else:
        plot_resampled = plot_df

    # build interactive figure
    fig = go.Figure()
    if 'y' in plot_resampled.columns and not plot_resampled['y'].isna().all():
        fig.add_trace(go.Scatter(x=plot_resampled['ds'], y=plot_resampled['y'], mode='lines+markers', name='Ventes réelles', hovertemplate='Date: %{x}<br>Réel: %{y}<extra></extra>'))
    if 'yhat' in plot_resampled.columns and not plot_resampled['yhat'].isna().all():
        fig.add_trace(go.Scatter(x=plot_resampled['ds'], y=plot_resampled['yhat'], mode='lines+markers', name='Prédiction', line=dict(dash='dash'), hovertemplate='Date: %{x}<br>Préd: %{y}<extra></extra>'))
    if 'yhat_upper' in plot_resampled.columns and 'yhat_lower' in plot_resampled.columns:
        fig.add_trace(go.Scatter(x=plot_resampled['ds'], y=plot_resampled['yhat_upper'], mode='lines', name='Upper', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=plot_resampled['ds'], y=plot_resampled['yhat_lower'], mode='lines', name='Lower', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,165,0,0.2)', showlegend=False))

    fig.update_layout(title='Ventes réelles vs Prédictions', xaxis_title='Date', yaxis_title='Valeur', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # Stats and anomalies
    st.subheader('Statistiques & anomalies')
    if 'y' in combined.columns and 'yhat' in combined.columns and not combined[['y','yhat']].dropna().empty:
        eval_df = combined.dropna(subset=['y','yhat']).copy()
        eval_df['abs_err'] = (eval_df['y'] - eval_df['yhat']).abs()
        mae = float(eval_df['abs_err'].mean())
        mape = float((eval_df['abs_err'] / eval_df['y'].replace({0: pd.NA})).abs().mean() * 100)
        st.metric('MAE', f"{mae:.2f}")
        st.metric('MAPE', f"{mape:.2f}%")

        # anomalies: outside interval if available, else by pct threshold
        if 'yhat_upper' in combined.columns and 'yhat_lower' in combined.columns:
            mask = (combined['y'] > combined['yhat_upper']) | (combined['y'] < combined['yhat_lower'])
            anomalies = combined.loc[mask, ['ds','y','yhat','yhat_lower','yhat_upper']]
        else:
            combined['pct_diff'] = ((combined['y'] - combined['yhat']) / combined['y']).abs() * 100
            anomalies = combined[combined['pct_diff'] > anomaly_pct][['ds','y','yhat','pct_diff']]

        if anomalies is not None and not anomalies.empty:
            st.warning(f"{len(anomalies)} anomalie(s) détectée(s)")
            st.dataframe(anomalies.sort_values('ds').head(50))
        else:
            st.success('Aucune anomalie majeure détectée')

    else:
        st.info('Pas assez de points communs entre historique et prévisions pour calculer des statistiques')

    # granular data access
    st.subheader('Données détaillées')
    st.write('Téléchargez les données combinées (historique + prévisions)')
    if not combined.empty:
        csv_bytes = combined.sort_values('ds').to_csv(index=False).encode('utf-8')
        st.download_button('Télécharger données combinées (CSV)', data=csv_bytes, file_name='combined_forecast_full.csv', mime='text/csv')
    else:
        st.info('Aucune donnée combinée disponible.')
