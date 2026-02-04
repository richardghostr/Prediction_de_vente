import streamlit as st
from pathlib import Path
import pandas as pd
import os

try:
    from src.models.predict import predict_series, get_model
except Exception:
    predict_series = None
    get_model = None

try:
    from src.serve.schemas import SeriesRequest, ForecastResponse
except Exception:
    SeriesRequest = None
    ForecastResponse = None


def build_series_from_df(df: pd.DataFrame, date_col: str = "date", value_col: str = "value", id_col: str = None):
    """
    Build a list of datapoints suitable for the API `SeriesRequest`.

    - Aggregates duplicate dates by summing numeric columns (including `value`) and taking first
      non-null for non-numeric extras.
    - Returns a tuple `(series_list, series_id)` where `series_id` is the common id value
      if `id_col` is present and constant, otherwise `None`.
    """
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    df[date_col] = pd.to_datetime(df[date_col])

    # determine series_id if possible
    series_id = None
    if id_col and id_col in df.columns:
        try:
            uniq = df[id_col].dropna().unique()
            if len(uniq) == 1:
                series_id = str(uniq[0])
        except Exception:
            series_id = None

    # aggregate duplicates on date
    agg_dict = {}
    for c in df.columns:
        if c == date_col:
            continue
        if c == id_col:
            # ignore id column during aggregation
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            agg_dict[c] = 'sum'
        else:
            agg_dict[c] = 'first'

    grouped = df.groupby(df[date_col].dt.floor('D')).agg(agg_dict).reset_index()
    grouped = grouped.sort_values(by=date_col)

    series = []
    for _, row in grouped.iterrows():
        item = {"date": pd.to_datetime(row[date_col]).date().isoformat()}
        if value_col in grouped.columns:
            val = row[value_col]
            item["value"] = None if pd.isna(val) else float(val)
        # include extras (all columns except date and value)
        for c in grouped.columns:
            if c in (date_col, value_col):
                continue
            v = row[c]
            if pd.isna(v):
                continue
            # coerce numeric values to float for JSON friendliness
            if isinstance(v, (int, float)):
                item[c] = float(v)
            else:
                item[c] = str(v)
        series.append(item)

    # compute aggregation info: number of original rows vs number of unique dates
    try:
        orig_count = int(df.shape[0])
        unique_dates = int(df[date_col].dt.floor('D').nunique())
        grouped_dates = int(grouped.shape[0])
        agg_info = {"orig_rows": orig_count, "unique_dates": unique_dates, "grouped_rows": grouped_dates}
    except Exception:
        agg_info = {"orig_rows": len(df), "unique_dates": len(grouped), "grouped_rows": len(grouped)}
    return series, series_id, agg_info


def render_predict_ui():
    st.header("Analyse et Prédiction")
    st.write("Utilise `src/models/predict.py` pour effectuer des prédictions offline basées sur les données nettoyées / de features.")

    # Let user pick a clean or features file
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")

    candidates = []
    if interim_dir.exists():
        candidates += sorted(interim_dir.glob("*_clean.csv"))
    if processed_dir.exists():
        candidates += sorted(processed_dir.glob("*_features.csv"))

    if not candidates:
        st.info("Aucun fichier clean/features trouvé. Exécutez les étapes précédentes ou archive des fichiers dans data/interim / data/processed.")
        return

    sel = st.selectbox("Choisir un fichier pour la prédiction", options=[str(p) for p in candidates])
    try:
        df = pd.read_csv(sel)
    except Exception as e:
        st.error(f"Impossible de lire le fichier sélectionné: {e}")
        return

    st.subheader("Aperçu des données sélectionnées")
    st.dataframe(df.head())

    horizon = st.slider("Horizon (jours)", 1, 365, 30)

    # detect id column
    id_col = None
    for c in df.columns:
        if c.lower() in ("id", "store", "storeid", "productid", "sku"):
            id_col = c
            break

    mode = st.radio("Mode d'exécution", options=["Offline (local)", "Online (API)"], index=0)

    if mode == "Offline (local)":
        if st.button("Lancer la prédiction (offline) "):
            if predict_series is None:
                st.error("`predict_series` non disponible dans l'environnement")
                return
            try:
                payload_series, suggested_id, agg_info = build_series_from_df(df, id_col=id_col)
                st.info(f"Aggregation: {agg_info}")
                res = predict_series(payload_series, horizon=horizon)
                st.success("Prédiction terminée")
                fc = pd.DataFrame(res.get("forecast", []))
                if not fc.empty:
                    st.subheader("Aperçu de la prédiction")
                    st.dataframe(fc.head())
                    csv_bytes = fc.to_csv(index=False).encode("utf-8")
                    st.download_button("Télécharger la prédiction (CSV)", data=csv_bytes, file_name="forecast.csv", mime="text/csv")
                else:
                    st.warning("La prédiction n'a retourné aucun résultat")
            except Exception as e:
                st.error(f"Erreur durant la prédiction: {e}")

    else:
        # Online API mode
        api_base = st.text_input("API base URL", value=os.getenv("API_URL", "http://api:8000"))
        series_id = None
        # if id_col exists and is constant, propose to use it as series id
        if id_col:
            try:
                uniq = df[id_col].dropna().unique()
                if len(uniq) == 1:
                    series_id = str(uniq[0])
            except Exception:
                series_id = None
        series_id = st.text_input("Series ID (optional)", value=series_id or "")

        if st.button("Lancer la prédiction (API)"):
            try:
                import requests

                payload_series, suggested_id, agg_info = build_series_from_df(df, id_col=id_col)
                st.info(f"Aggregation: {agg_info}")
                payload = {"id": series_id or suggested_id or None, "series": payload_series}

                # Validate against API schema if available
                if SeriesRequest is not None:
                    try:
                        SeriesRequest.model_validate(payload)
                    except Exception as e:
                        st.error(f"Payload invalide selon le schéma API: {e}")
                        st.write(payload)
                        return

                url = f"{api_base.rstrip('/')}/predict?horizon={horizon}"
                resp = requests.post(url, json=payload, timeout=30)
                if resp.status_code != 200:
                    st.error(f"API error {resp.status_code}: {resp.text}")
                    return
                j = resp.json()

                # Validate response
                if ForecastResponse is not None:
                    try:
                        ForecastResponse.model_validate(j)
                    except Exception as e:
                        st.warning(f"Réponse API non conforme au schéma: {e}")

                st.success("Prédiction via API terminée")
                fc = pd.DataFrame(j.get("forecast", []))
                if not fc.empty:
                    st.subheader("Aperçu de la prédiction")
                    st.dataframe(fc.head())
                    csv_bytes = fc.to_csv(index=False).encode("utf-8")
                    st.download_button("Télécharger la prédiction (CSV)", data=csv_bytes, file_name="forecast_api.csv", mime="text/csv")
                else:
                    st.warning("La prédiction n'a retourné aucun résultat")
                if j.get("metrics"):
                    st.subheader("Metrics retournées par l'API")
                    st.json(j.get("metrics"))

            except Exception as e:
                st.error(f"Erreur lors de l'appel API: {e}")
