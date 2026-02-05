"""
Reusable prediction UI component for Streamlit.

Can be imported and called from the main dashboard or used standalone.
Provides both online (API) and offline (local model) prediction modes
with rich interactive visualizations.

Usage:
    # As a standalone page
    streamlit run src/ui/predict_ui.py

    # As a component in the main dashboard
    from src.ui.predict_ui import render_predict_ui
    render_predict_ui()
"""
import os
import sys
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.models.predict import predict_series, get_model, get_model_path
except Exception:
    predict_series = None
    get_model = None
    get_model_path = None

try:
    from src.serve.schemas import SeriesRequest, ForecastResponse
except Exception:
    SeriesRequest = None
    ForecastResponse = None

try:
    from src.data.clean import clean_dataframe
except Exception:
    clean_dataframe = None


PALETTE = {
    "primary": "#1B6B93",
    "secondary": "#F39C12",
    "success": "#27AE60",
    "danger": "#E74C3C",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_series_from_df(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "value",
    id_col: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str], Dict[str, int]]:
    """
    Build a list of datapoints suitable for the API / predict_series.

    Returns (series_list, series_id, aggregation_info).
    """
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    df[date_col] = pd.to_datetime(df[date_col])

    # Determine series_id
    series_id = None
    if id_col and id_col in df.columns:
        uniq = df[id_col].dropna().unique()
        if len(uniq) == 1:
            series_id = str(uniq[0])

    # Aggregate duplicates
    agg_dict = {}
    for c in df.columns:
        if c in (date_col, id_col):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            agg_dict[c] = "sum"
        else:
            agg_dict[c] = "first"

    grouped = df.groupby(df[date_col].dt.floor("D")).agg(agg_dict).reset_index()
    grouped = grouped.sort_values(by=date_col)

    series = []
    for _, row in grouped.iterrows():
        item = {"date": pd.to_datetime(row[date_col]).date().isoformat()}
        if value_col in grouped.columns:
            val = row[value_col]
            item["value"] = None if pd.isna(val) else float(val)
        # Extras
        for c in grouped.columns:
            if c in (date_col, value_col):
                continue
            v = row[c]
            if pd.isna(v):
                continue
            item[c] = float(v) if isinstance(v, (int, float)) else str(v)
        series.append(item)

    agg_info = {
        "original_rows": len(df),
        "unique_dates": int(df[date_col].dt.floor("D").nunique()),
        "aggregated_rows": len(grouped),
    }
    return series, series_id, agg_info


def run_offline_prediction(
    series: List[Dict[str, Any]],
    horizon: int,
) -> Dict[str, Any]:
    """Run prediction using the local model or naive fallback."""
    # Try trained model
    if predict_series is not None:
        try:
            return predict_series(series, horizon=horizon)
        except Exception as e:
            st.warning(f"Modele local indisponible ({e}), repli vers la prediction naive.")

    # Naive fallback
    last_val = None
    last_date = None
    for s in reversed(series):
        if s.get("value") is not None:
            last_val = s["value"]
            last_date = pd.to_datetime(s["date"])
            break

    if last_val is None:
        raise ValueError("Aucune valeur observee dans les donnees.")

    forecast = []
    for i in range(1, horizon + 1):
        ds = (last_date + timedelta(days=i)).date().isoformat()
        forecast.append({"ds": ds, "yhat": float(last_val)})

    return {"id": None, "forecast": forecast, "metrics": {"method": "naive_last_value"}}


def run_api_prediction(
    series: List[Dict[str, Any]],
    horizon: int,
    api_base: str,
    series_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run prediction via the API endpoint."""
    import requests as req_lib

    payload = {"id": series_id, "series": series}

    # Optional schema validation
    if SeriesRequest is not None:
        try:
            SeriesRequest.model_validate(payload)
        except Exception as e:
            raise ValueError(f"Payload invalide : {e}")

    url = f"{api_base.rstrip('/')}/predict?horizon={horizon}"
    resp = req_lib.post(url, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Erreur API {resp.status_code}: {resp.text}")

    result = resp.json()

    # Validate response
    if ForecastResponse is not None:
        try:
            ForecastResponse.model_validate(result)
        except Exception as e:
            st.warning(f"Reponse API non conforme au schema : {e}")

    return result


def display_forecast_results(
    forecast_result: Dict[str, Any],
    hist_df: Optional[pd.DataFrame] = None,
) -> None:
    """Display forecast results with rich visualizations."""
    fc = pd.DataFrame(forecast_result.get("forecast", []))
    if fc.empty:
        st.warning("La prediction n'a retourne aucun resultat.")
        return

    fc["ds"] = pd.to_datetime(fc["ds"])

    # Metrics
    if forecast_result.get("metrics"):
        st.subheader("Metriques")
        metrics = forecast_result["metrics"]
        cols = st.columns(min(len(metrics), 4))
        for i, (k, v) in enumerate(metrics.items()):
            with cols[i % len(cols)]:
                if isinstance(v, (int, float)):
                    st.metric(k, f"{v:.4f}")
                else:
                    st.metric(k, str(v))

    # Summary KPIs
    st.subheader("Resume de la prevision")
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Nb de jours prevus", len(fc))
    with kpi_cols[1]:
        st.metric("Moyenne", f"{fc['yhat'].mean():.2f}")
    with kpi_cols[2]:
        st.metric("Min", f"{fc['yhat'].min():.2f}")
    with kpi_cols[3]:
        st.metric("Max", f"{fc['yhat'].max():.2f}")

    # Chart
    st.subheader("Graphique de prevision")
    fig = go.Figure()

    # Historical
    if hist_df is not None and not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df["ds"], y=hist_df["y"],
            mode="lines", name="Historique",
            line=dict(color=PALETTE["primary"], width=2),
        ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=fc["ds"], y=fc["yhat"],
        mode="lines+markers", name="Prediction",
        line=dict(color=PALETTE["secondary"], width=2, dash="dash"),
        marker=dict(size=4),
    ))

    # Confidence intervals
    if "yhat_upper" in fc.columns and "yhat_lower" in fc.columns:
        upper = pd.to_numeric(fc["yhat_upper"], errors="coerce")
        lower = pd.to_numeric(fc["yhat_lower"], errors="coerce")
        if upper.notna().any() and lower.notna().any():
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=upper, mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=lower, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(243,156,18,0.15)",
                name="Intervalle de confiance",
            ))

    fig.update_layout(
        height=450,
        xaxis_title="Date",
        yaxis_title="Valeur",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Deviation analysis
    if hist_df is not None and not hist_df.empty:
        combined = pd.merge(hist_df, fc, on="ds", how="inner")
        if not combined.empty and "y" in combined.columns and "yhat" in combined.columns:
            st.subheader("Analyse des ecarts (sur la zone de recouvrement)")
            combined["deviation_%"] = (
                (combined["y"] - combined["yhat"]) / combined["y"].replace(0, float("nan")) * 100
            ).round(2)
            st.dataframe(
                combined[["ds", "y", "yhat", "deviation_%"]].rename(
                    columns={"y": "Reel", "yhat": "Predit", "deviation_%": "Ecart (%)"}
                ),
                use_container_width=True,
            )

    # Data table
    st.subheader("Donnees de prevision")
    st.dataframe(fc, use_container_width=True)

    # Download
    csv = fc.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Telecharger les previsions (CSV)",
        data=csv, file_name="forecast.csv", mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_predict_ui() -> None:
    """Render the full prediction UI panel."""
    st.header("Prediction avancee")
    st.write(
        "Selectionnez un fichier source (nettoye ou avec features) ou chargez un CSV "
        "pour lancer une prediction en mode hors-ligne ou en ligne via l'API."
    )

    # Source selection
    source_option = st.radio(
        "Source des donnees",
        options=["Fichiers du pipeline", "Charger un CSV"],
        horizontal=True,
        key="predict_source_opt",
    )

    df = None
    date_col = "date"
    value_col = "value"

    if source_option == "Fichiers du pipeline":
        # Discover existing pipeline files
        interim_dir = Path("data/interim")
        processed_dir = Path("data/processed")
        candidates = []
        if interim_dir.exists():
            candidates += sorted(interim_dir.glob("*.csv"))
        if processed_dir.exists():
            candidates += sorted(processed_dir.glob("*.csv"))

        if not candidates:
            st.info(
                "Aucun fichier trouve dans `data/interim` ou `data/processed`. "
                "Executez les etapes d'ingestion et de nettoyage, ou chargez un CSV manuellement."
            )
            return

        selected = st.selectbox(
            "Fichier source",
            options=[str(p) for p in candidates],
            key="predict_file_sel",
        )
        try:
            df = pd.read_csv(selected)
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")
            return

    else:
        uploaded = st.file_uploader("Charger un CSV", type=["csv"], key="predict_csv_upload")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Erreur de lecture : {e}")
                return

    if df is None:
        return

    # Column detection
    date_candidates = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "ds"))]
    value_candidates = df.select_dtypes(include=["number"]).columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox(
            "Colonne de date",
            options=date_candidates if date_candidates else df.columns.tolist(),
            key="predict_date_col",
        )
    with c2:
        value_col = st.selectbox(
            "Colonne de valeur",
            options=value_candidates if value_candidates else df.columns.tolist(),
            key="predict_value_col",
        )

    # Preview
    st.subheader("Apercu des donnees")
    st.dataframe(df.head(10), use_container_width=True)

    # Config
    c1, c2 = st.columns(2)
    with c1:
        horizon = st.slider("Horizon de prediction (jours)", 1, 365, 30, key="predict_horizon")
    with c2:
        mode = st.radio("Mode", options=["Hors-ligne", "En ligne (API)"], horizontal=True, key="predict_mode")

    # ID column detection
    id_col = None
    for c in df.columns:
        if c.lower() in ("id", "store", "store_id", "storeid", "product_id", "productid", "sku"):
            id_col = c
            break

    # API settings (only shown for online mode)
    api_base = None
    if mode == "En ligne (API)":
        api_base = st.text_input(
            "URL de l'API",
            value=os.getenv("API_URL", "http://api:8000"),
            key="predict_api_url",
        )

    # Run
    if st.button("Lancer la prediction", type="primary", use_container_width=True, key="predict_run"):
        with st.spinner("Preparation et prediction en cours..."):
            try:
                # Clean data first
                if clean_dataframe is not None:
                    df_clean = clean_dataframe(df, date_col=date_col, value_col=value_col)
                else:
                    df_clean = df.copy()
                    df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors="coerce")
                    df_clean = df_clean.dropna(subset=[date_col]).sort_values(date_col)

                # Build series payload
                series, series_id, agg_info = build_series_from_df(
                    df_clean, date_col=date_col, value_col=value_col, id_col=id_col,
                )

                st.caption(
                    f"Donnees : {agg_info['original_rows']} lignes -> "
                    f"{agg_info['aggregated_rows']} dates uniques"
                )

                # Build historical df for overlay
                hist_df = df_clean[[date_col, value_col]].dropna().copy()
                hist_df = hist_df.rename(columns={date_col: "ds", value_col: "y"})
                hist_df["ds"] = pd.to_datetime(hist_df["ds"])
                if hist_df["ds"].duplicated().any():
                    hist_df = hist_df.groupby("ds", as_index=False).agg({"y": "sum"})

                # Run prediction
                if mode == "Hors-ligne":
                    result = run_offline_prediction(series, horizon=horizon)
                else:
                    result = run_api_prediction(
                        series, horizon=horizon,
                        api_base=api_base,
                        series_id=series_id,
                    )

                # Display
                st.success("Prediction terminee")
                display_forecast_results(result, hist_df=hist_df)

            except Exception as e:
                st.error(f"Erreur : {e}")


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Prediction", layout="wide")
    render_predict_ui()
