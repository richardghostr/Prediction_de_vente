import streamlit as st
import pandas as pd
from datetime import timedelta
import json
from pathlib import Path
import os
import plotly.graph_objects as go

# Try to import pipeline helpers so the UI enforces ingest -> clean -> features
try:
    from src.data.clean import clean_dataframe
except Exception:
    clean_dataframe = None
try:
    from src.data.features import build_feature_pipeline
except Exception:
    build_feature_pipeline = None
def detect_columns(df: pd.DataFrame):
    # detect date column: prefer common names, else choose the most-parseable column
    date_candidates = ["date", "Date", "ds", "timestamp", "datetime", "time", "sale_date", "saledate"]
    date_col = None
    for c in date_candidates:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        best_date = None
        best_score = 0
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                score = parsed.notna().mean()
                if score > best_score:
                    best_score = score
                    best_date = col
            except Exception:
                continue
        date_col = best_date

    # detect value column: prefer common names, otherwise numeric column with most non-nulls
    value_candidates = ["value", "y", "sales", "amount", "revenue", "qty", "quantity", "unit_price"]
    value_col = None
    for c in value_candidates:
        if c in df.columns:
            value_col = c
            break

    if value_col is None:
        best_num = None
        best_num_count = -1
        for col in df.select_dtypes(include=["number"]).columns:
            cnt = df[col].notna().sum()
            if cnt > best_num_count:
                best_num_count = cnt
                best_num = col
        value_col = best_num

    return (date_col or "date"), value_col

st.set_page_config(page_title="Prediction UI", layout="wide")

st.title("📈 Prediction_de_vente — Tableau de bord interactif")

st.markdown("Uploadez un fichier CSV avec au minimum une colonne de date et une de valeur. L'application tentera de les détecter automatiquement et vous permettra de filtrer et d'agréger les données.")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

# controls: horizon/offline always visible
col1, col2 = st.columns([2, 1])
with col1:
    st.write("")
with col2:
    horizon = st.slider("Horizon (days)", 1, 365, 30)
    offline = st.checkbox("Offline mode (no API call)")

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    # --- Sidebar for controls ---
    with st.sidebar:
        st.header("⚙️ Contrôles")
        # auto-detect columns and show inputs (provide explicit keys to avoid duplicate widget ids)
        detected_date, detected_value = detect_columns(df)
        default_date = detected_date or "date"

        # Prefer `quantity` when available (model was trained on quantity by default).
        # If `revenue` is detected but `quantity` exists, suggest switching to `quantity`.
        cols_lower = [c.lower() for c in df.columns]
        has_quantity = any('quantity' == c or 'qty' == c for c in cols_lower)

        if has_quantity:
            # Auto-select `quantity` when present (user requested automatic switch).
            default_value = 'quantity'
            # Inform the user that we auto-selected quantity, and how to override
            st.info("Colonne 'quantity' détectée — sélection automatique de 'quantity' (utilisez le champ ci-dessous pour changer). Si vous voulez prédire le revenue/CA, sélectionnez 'revenue' manuellement et entraînez un modèle sur cette cible.")
        else:
            default_value = detected_value or 'value'

        # If revenue-like column is selected explicitly, warn user
        if detected_value and any(k in detected_value.lower() for k in ("rev", "revenue", "price")) and not has_quantity:
            st.warning("Colonne 'revenue' détectée. Le modèle par défaut a été entraîné sur la quantité ('quantity'); les prévisions peuvent être hors échelle pour le revenue. Envisagez d'entraîner un modèle sur 'revenue'.")
        date_col = st.text_input("Colonne de date", value=default_date, key="date_col_input")
        value_col = st.text_input("Colonne de valeur", value=default_value, key="value_col_input")

        # Allow selecting a grouping column (e.g., store_id) and aggregation behavior for duplicate dates
        candidate_group_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns.tolist() if c not in (date_col, value_col)
        ]
        if candidate_group_cols:
            group_col = st.selectbox("Colonne de regroupement (optionnel)", options=["None"] + candidate_group_cols, index=0, key="group_col")
        else:
            group_col = "None"

        group_filter = "All"
        if group_col != "None":
            unique_vals = sorted(df[group_col].dropna().astype(str).unique().tolist())
            group_filter = st.selectbox("Filtrer par groupe ('All' pour tout voir)", options=["All"] + unique_vals, index=0, key="group_filter")

        aggregate = st.checkbox("Agréger les dates dupliquées (par somme)", value=True, key="aggregate_dup")
        archive_outputs = st.checkbox("Archiver les sorties (interim/processed)", value=False, key="archive_outputs")

    st.subheader("Aperçu des données brutes")
    st.dataframe(df.head())

    if date_col not in df.columns:
        st.error(f"Date column '{date_col}' not found in uploaded CSV")
        st.stop()
    if value_col not in df.columns:
        st.warning(f"Value column '{value_col}' not found; values will be set to null")

    # ensure date parsed
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        st.error(f"Could not parse dates in column '{date_col}': {e}")
        st.stop()

    # Work on a copy and optionally filter by group
    df2 = df.copy()
    if group_col != "None" and group_filter != "All":
        df2 = df2[df2[group_col].astype(str) == group_filter]

    # detect duplicates on the date column
    if df2.duplicated(subset=[date_col], keep=False).any():
        if aggregate:
            # build aggregation dict: sum numeric cols, first for non-numeric extras
            numeric_cols = df2.select_dtypes(include=["number"]).columns.tolist()
            extras_cols = [c for c in df2.columns if c not in (date_col, value_col, group_col)]
            agg_dict = {}
            if value_col in df2.columns:
                agg_dict[value_col] = "sum"
            for c in extras_cols:
                if c in numeric_cols:
                    agg_dict[c] = "sum"
                else:
                    agg_dict[c] = "first"
            # perform aggregation
            try:
                agg_df = df2.groupby(date_col, dropna=False).agg(agg_dict).reset_index()
                df_for_series = agg_df
            except Exception:
                st.error("Failed to aggregate duplicate dates; please pre-aggregate your CSV or choose a group filter.")
                st.stop()
        else:
            st.error("Duplicate dates found in the selected data. Choose a group filter or enable aggregation.")
            st.stop()
    else:
        df_for_series = df2

    # Run cleaning + feature pipeline to enforce chronology before building series
    try:
        if clean_dataframe is not None:
            cleaned_df = clean_dataframe(df_for_series, date_col=date_col, value_col=value_col)
        else:
            cleaned_df = df_for_series
    except Exception as e:
        st.error(f"Erreur lors du nettoyage des données: {e}")
        st.stop()

    try:
        if build_feature_pipeline is not None:
            features_df = build_feature_pipeline(cleaned_df)
        else:
            features_df = None
    except Exception as e:
        st.warning(f"Échec du calcul des features: {e}")
        features_df = None

    # Optionally show previews of cleaned/features
    st.subheader("Aperçu après nettoyage")
    st.dataframe(cleaned_df.head())
    if features_df is not None:
        st.subheader("Aperçu des features")
        st.dataframe(features_df.head())

    # If user requested archiving, write interim/processed CSVs
    if archive_outputs:
        try:
            out_interim = Path("data/interim")
            out_processed = Path("data/processed")
            out_interim.mkdir(parents=True, exist_ok=True)
            out_processed.mkdir(parents=True, exist_ok=True)
            interim_name = out_interim / f"uploaded_interim_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            cleaned_df.to_csv(interim_name, index=False)
            if features_df is not None:
                proc_name = out_processed / f"uploaded_features_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
                features_df.to_csv(proc_name, index=False)
            st.success(f"Sorties archivées: {interim_name}{' , ' + str(proc_name) if features_df is not None else ''}")
        except Exception as e:
            st.warning(f"Impossible d'archiver les sorties: {e}")

    # Build series payload from cleaned_df (guarantees chronology applied)
    series = []
    for _, row in cleaned_df.iterrows():
        # ensure date cell is a Timestamp
        try:
            ds = pd.to_datetime(row[date_col])
            date_iso = ds.date().isoformat()
        except Exception:
            date_iso = str(row[date_col])
        item = {"date": date_iso}
        if value_col in df_for_series.columns:
            try:
                item["value"] = None if pd.isna(row[value_col]) else float(row[value_col])
            except Exception:
                item["value"] = None
        # include extras (exclude grouping column)
        extras = {k: v for k, v in row.items() if k not in (date_col, value_col, group_col)}
        item.update(extras)
        # If the user selected a grouping column, include it as `id` so the model
        # receives the entity identifier expected by the feature pipeline.
        try:
            if group_col != "None" and group_col in row.index:
                gid = row[group_col]
                # cast to str for safety
                item["id"] = str(gid) if not pd.isna(gid) else None
        except Exception:
            pass
        series.append(item)

    if st.button("Lancer la prédiction"):
        if not series:
            st.error("Aucune donnée à prédire")
        else:
            if not offline:
                # call API; prefer API_URL env var (inside Docker use service name 'api')
                try:
                    import requests

                    api_base = os.getenv("API_URL", "http://api:8000")
                    url = f"{api_base.rstrip('/')}/predict?horizon={horizon}"
                    payload = {"id": "uploaded_series", "series": series}
                    resp = requests.post(url, json=payload, timeout=20)
                    if resp.status_code != 200:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                        st.stop()
                    j = resp.json()
                except Exception as e:
                    st.error(f"Failed to call API: {e}")
                    st.info(f"Tried API URL: {os.getenv('API_URL', 'http://api:8000')}")
                    st.stop()
            else:
                # try advanced offline: use local model wrapper if available
                try:
                    from src.models import predict as predict_module

                    if hasattr(predict_module, "predict_series"):
                        j = predict_module.predict_series(series, horizon=horizon)
                    else:
                        raise ImportError("predict_series not available")
                except Exception:
                    # fallback naive: last-value carry forward
                    last = None
                    for s in reversed(series):
                        if s.get("value") is not None:
                            last = s.get("value")
                            last_date = pd.to_datetime(s.get("date"))
                            break
                    if last is None:
                        st.error("No observed `value` in series to base offline forecast on")
                        st.stop()
                    j = {"id": "uploaded_series", "forecast": []}
                    for i in range(1, horizon + 1):
                        ds = (last_date + timedelta(days=i)).date().isoformat()
                        j["forecast"].append({"ds": ds, "yhat": float(last)})

            # --- Enhanced Display with Tabs ---
            st.subheader("Résultats de la prédiction")
            
            fc = pd.DataFrame(j.get("forecast", []))
            if fc.empty:
                st.warning("La prédiction n'a retourné aucun résultat.")
                st.stop()

            fc["ds"] = pd.to_datetime(fc["ds"]).dt.date
            
            # Prepare historical data for plotting
            hist = df_for_series[[date_col, value_col]].dropna() if value_col in df_for_series.columns else pd.DataFrame()
            if not hist.empty:
                hist = hist.rename(columns={date_col: "ds", value_col: "y"})
                hist["ds"] = pd.to_datetime(hist["ds"]).dt.date
                if hist["ds"].duplicated().any():
                    hist = hist.groupby("ds", as_index=False).agg({"y": "sum"})
            
            # Merge historical and forecast data
            combined = pd.merge(hist, fc, on="ds", how="outer").sort_values("ds").reset_index(drop=True)
            combined['ds'] = pd.to_datetime(combined['ds'])

            # --- Anomaly/Alert detection ---
            alerts = []
            if 'yhat_upper' in combined.columns and 'yhat_lower' in combined.columns:
                # convert to numeric safely (None or non-numeric -> NaN) before comparisons
                y_num = pd.to_numeric(combined['y'], errors='coerce')
                yhat_upper_num = pd.to_numeric(combined['yhat_upper'], errors='coerce')
                yhat_lower_num = pd.to_numeric(combined['yhat_lower'], errors='coerce')

                mask_upper = y_num.notna() & yhat_upper_num.notna() & (y_num > yhat_upper_num)
                mask_lower = y_num.notna() & yhat_lower_num.notna() & (y_num < yhat_lower_num)
                anomalies = combined[mask_upper | mask_lower]
                if not anomalies.empty:
                    alerts.append(f"{len(anomalies)} anomalie(s) détectée(s) où la valeur réelle est hors de l'intervalle de confiance.")
            
            # --- Tabs for different views ---
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Vue d'ensemble", "🗓️ Analyse Mensuelle", "📊 Détails des données", "🚨 Alertes"])

            with tab1:
                st.header("Courbes de prédiction interactive")
                fig = go.Figure()
                # Historical data
                fig.add_trace(go.Scatter(x=combined['ds'], y=combined['y'], mode='lines', name='Ventes réelles', line=dict(color='blue')))
                # Forecast
                fig.add_trace(go.Scatter(x=combined['ds'], y=combined['yhat'], mode='lines', name='Prédiction', line=dict(color='orange', dash='dash')))
                # Confidence intervals
                if 'yhat_upper' in combined.columns and 'yhat_lower' in combined.columns:
                    fig.add_trace(go.Scatter(
                        x=combined['ds'], y=combined['yhat_upper'], mode='lines', name='Intervalle sup.', line=dict(width=0), showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=combined['ds'], y=combined['yhat_lower'], mode='lines', name='Intervalle inf.', line=dict(width=0), showlegend=False,
                        fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)'
                    ))
                
                fig.update_layout(
                    title="Ventes réelles vs. Prédictions",
                    xaxis_title="Date",
                    yaxis_title="Valeur",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.header("Analyse mensuelle agrégée")
                monthly_df = combined.set_index('ds').resample('ME').sum().reset_index()
                monthly_df['month'] = monthly_df['ds'].dt.strftime('%Y-%m')
                
                fig_monthly = go.Figure()
                fig_monthly.add_trace(go.Bar(x=monthly_df['month'], y=monthly_df['y'], name='Ventes réelles mensuelles'))
                fig_monthly.add_trace(go.Bar(x=monthly_df['month'], y=monthly_df['yhat'], name='Prédictions mensuelles'))
                st.plotly_chart(fig_monthly, use_container_width=True)

                st.write("Statistiques mensuelles :")
                st.dataframe(monthly_df[['month', 'y', 'yhat']].rename(columns={'y': 'Total réel', 'yhat': 'Total prédit'}).round(2))

            with tab3:
                st.header("Détails des données et téléchargement")
                st.dataframe(combined.rename(columns={'y': 'réel', 'yhat': 'prédit'}).set_index('ds').round(2))
                csv_bytes = combined.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Télécharger les données combinées (CSV)", data=csv_bytes, file_name="combined_forecast.csv", mime="text/csv")

            with tab4:
                st.header("Alertes et déviations")
                if alerts:
                    for alert in alerts:
                        st.warning(alert)
                else:
                    st.success("Aucune anomalie majeure détectée.")
                
                combined['deviation'] = ((combined['y'] - combined['yhat']) / combined['y']).replace([float('inf'), -float('inf')], 0).fillna(0) * 100
                st.write("Écart en % par rapport à la prédiction (réel - prédit) / réel:")
                st.dataframe(combined[['ds', 'y', 'yhat', 'deviation']].rename(columns={'y': 'réel', 'yhat': 'prédit', 'deviation': 'écart (%)'}).set_index('ds').dropna().round(2))
