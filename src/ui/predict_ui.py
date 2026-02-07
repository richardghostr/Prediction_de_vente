"""
Interface Streamlit avancee pour la prediction.

Supporte la prediction hors-ligne (modele local XGBoost) et en ligne (API).
Fournit des graphiques interactifs, des metriques, un resume de prevision
et des options d'export.

Usage:
    # En tant que composant du dashboard
    from src.ui.predict_ui import render_predict_ui
    render_predict_ui()

    # En standalone
    streamlit run src/ui/predict_ui.py
"""
import os
import sys
from pathlib import Path
from datetime import timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import io

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
    "primary": "#0F4C75",
    "primary_light": "#3282B8",
    "secondary": "#E77728",
    "success": "#2ECC71",
    "danger": "#E74C3C",
    "warning": "#F39C12",
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
    """Build a list of datapoints suitable for the API / predict_series."""
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Colonne de date manquante : {date_col}")
    df[date_col] = pd.to_datetime(df[date_col])

    series_id = None
    if id_col and id_col in df.columns:
        uniq = df[id_col].dropna().unique()
        if len(uniq) == 1:
            series_id = str(uniq[0])

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
    if predict_series is not None:
        try:
            return predict_series(series, horizon=horizon)
        except Exception as e:
            st.warning(f"Modele local indisponible ({e}), repli vers la prediction naive.")

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
    """Display forecast results with rich interactive visualizations."""
    fc = pd.DataFrame(forecast_result.get("forecast", []))
    if fc.empty:
        st.warning("La prediction n'a retourne aucun resultat.")
        return

    fc["ds"] = pd.to_datetime(fc["ds"])

    # Metrics
    if forecast_result.get("metrics"):
        st.subheader("Metriques du modele")
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
        st.metric("Jours prevus", len(fc))
    with kpi_cols[1]:
        st.metric("Moyenne", f"{fc['yhat'].mean():.2f}")
    with kpi_cols[2]:
        st.metric("Min", f"{fc['yhat'].min():.2f}")
    with kpi_cols[3]:
        st.metric("Max", f"{fc['yhat'].max():.2f}")

    # Trend indicator
    if hist_df is not None and not hist_df.empty:
        last_hist_val = hist_df["y"].iloc[-1]
        avg_fc = fc["yhat"].mean()
        if last_hist_val != 0:
            pct_change = ((avg_fc - last_hist_val) / last_hist_val) * 100
            trend = "hausse" if pct_change > 0 else "baisse"
            st.markdown(
                f"<div style='background: {'#EAFAF1' if pct_change > 0 else '#FDEDEC'}; "
                f"border-left: 4px solid {PALETTE['success'] if pct_change > 0 else PALETTE['danger']}; "
                f"padding: 0.6rem 1rem; border-radius: 0 0.5rem 0.5rem 0; margin: 0.5rem 0;'>"
                f"<strong>Tendance :</strong> {pct_change:+.1f}% en {trend} par rapport a la derniere valeur historique"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Main chart
    st.subheader("Graphique de prevision")
    fig = go.Figure()

    if hist_df is not None and not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df["ds"], y=hist_df["y"],
            mode="lines", name="Historique",
            line=dict(color=PALETTE["primary"], width=2),
            hovertemplate="Date: %{x}<br>Historique: %{y:,.2f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=fc["ds"], y=fc["yhat"],
        mode="lines+markers", name="Prediction",
        line=dict(color=PALETTE["secondary"], width=2.5, dash="dash"),
        marker=dict(size=5),
        hovertemplate="Date: %{x}<br>Prediction: %{y:,.2f}<extra></extra>",
    ))

    if "yhat_upper" in fc.columns and "yhat_lower" in fc.columns:
        upper = pd.to_numeric(fc["yhat_upper"], errors="coerce")
        lower = pd.to_numeric(fc["yhat_lower"], errors="coerce")
        if upper.notna().any() and lower.notna().any():
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=upper, mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=lower, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(231, 119, 40, 0.15)",
                name="Intervalle de confiance",
            ))

    # Transition line
    if hist_df is not None and not hist_df.empty:
        last_date = hist_df["ds"].iloc[-1]
        last_val = hist_df["y"].iloc[-1]
        first_fc_date = fc["ds"].iloc[0]
        first_fc_val = fc["yhat"].iloc[0]
        fig.add_trace(go.Scatter(
            x=[last_date, first_fc_date],
            y=[last_val, first_fc_val],
            mode="lines",
            line=dict(color="#95A5A6", width=1.5, dash="dot"),
            showlegend=False,
        ))

    fig.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Valeur",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Weekly breakdown
    if len(fc) > 7:
        st.divider()
        st.subheader("Prevision par semaine")
        fc_weekly = fc.set_index("ds").resample("W")["yhat"].agg(["sum", "mean", "count"]).reset_index()
        fc_weekly.columns = ["Semaine", "Total", "Moyenne", "Nb jours"]

        c1, c2 = st.columns(2)
        with c1:
            fig_weekly = px.bar(
                fc_weekly, x="Semaine", y="Total",
                color_discrete_sequence=[PALETTE["secondary"]],
            )
            fig_weekly.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_weekly, use_container_width=True)
        with c2:
            st.dataframe(fc_weekly.round(2), use_container_width=True, hide_index=True, height=350)

    # Deviation analysis
    if hist_df is not None and not hist_df.empty:
        combined = pd.merge(hist_df, fc, on="ds", how="inner")
        if not combined.empty and "y" in combined.columns and "yhat" in combined.columns:
            st.divider()
            st.subheader("Analyse des ecarts (zone de recouvrement)")
            combined["ecart_abs"] = (combined["y"] - combined["yhat"]).abs().round(2)
            combined["ecart_pct"] = (
                (combined["y"] - combined["yhat"]) / combined["y"].replace(0, float("nan")) * 100
            ).round(2)
            st.dataframe(
                combined[["ds", "y", "yhat", "ecart_abs", "ecart_pct"]].rename(
                    columns={"y": "Reel", "yhat": "Predit", "ecart_abs": "Ecart abs.", "ecart_pct": "Ecart (%)"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    # Data table & download
    st.divider()
    st.subheader("Donnees de prevision")
    st.dataframe(fc, use_container_width=True, hide_index=True)

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
    st.markdown(
        "Selectionnez un fichier source ou chargez un CSV "
        "pour lancer une prediction en mode hors-ligne (modele local) ou en ligne (API)."
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
        interim_dir = Path("data/interim")
        processed_dir = Path("data/processed")
        candidates = []
        if processed_dir.exists():
            candidates += sorted(processed_dir.glob("*.csv"), reverse=True)
        if interim_dir.exists():
            candidates += sorted(interim_dir.glob("*.csv"), reverse=True)

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

        # Try to restore uploaded from dashboard_state cache
        try:
            if uploaded is None and "dashboard_state" in st.session_state:
                ufiles = st.session_state["dashboard_state"].get("uploaded_files", {})
                if ufiles and ufiles.get("predict"):
                    data = ufiles["predict"].get("data")
                    name = ufiles["predict"].get("name")
                    if data:
                        uploaded = io.BytesIO(data)
                        uploaded.name = name or "restored_predict.csv"
        except Exception:
            pass

        if uploaded is not None:
            try:
                try:
                    uploaded.seek(0)
                except Exception:
                    pass
                data_bytes = uploaded.read()
                # Persist to dashboard_state
                try:
                    st.session_state.setdefault("dashboard_state", {}).setdefault("uploaded_files", {})["predict"] = {
                        "name": getattr(uploaded, "name", "predict.csv"),
                        "data": data_bytes,
                    }
                except Exception:
                    pass
                # durable copy
                try:
                    cache_dir = PROJECT_ROOT / "data" / "interim" / ".dashboard_cache"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    if data_bytes is not None:
                        with open(cache_dir / "predict_uploaded.csv", "wb") as _f:
                            _f.write(data_bytes)
                except Exception:
                    pass
                # Load dataframe from bytes
                df = pd.read_csv(io.BytesIO(data_bytes))
            except Exception as e:
                st.error(f"Erreur de lecture : {e}")
                return

    # Fallback: restore previously cached parsed dataframe if available
    if df is None:
        try:
            last = st.session_state.get("dashboard_state", {}).get("last_results", {})
            p = last.get("predict_df")
            if p:
                try:
                    df = pd.read_pickle(p)
                except Exception:
                    try:
                        df = pd.read_csv(p)
                    except Exception:
                        df = None
        except Exception:
            df = None

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
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Lignes", f"{len(df):,}")
    with c2:
        st.metric("Colonnes", f"{len(df.columns)}")
    with c3:
        if value_col in df.columns:
            st.metric(f"Moy. {value_col}", f"{df[value_col].mean():.2f}")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # Prediction config
    st.divider()
    st.subheader("Configuration de la prediction")
    c1, c2 = st.columns(2)
    with c1:
        horizon = st.slider("Horizon de prediction (jours)", 1, 365, 30, key="predict_horizon")
    with c2:
        mode = st.radio("Mode", options=["Hors-ligne", "En ligne (API)"], horizontal=True, key="predict_mode")

    # Model status
    if get_model_path:
        try:
            mp = get_model_path()
            if mp:
                st.markdown(
                    f"<div style='background: #EAFAF1; border-left: 4px solid {PALETTE['success']}; "
                    f"padding: 0.5rem 1rem; border-radius: 0 0.5rem 0.5rem 0;'>"
                    f"<strong>Modele disponible :</strong> {Path(mp).name}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Aucun artefact de modele trouve. La prediction naive sera utilisee.")
        except Exception:
            st.warning("Impossible de verifier le statut du modele.")

    # ID column detection
    id_col = None
    for c in df.columns:
        if c.lower() in ("id", "store", "store_id", "storeid", "product_id", "productid", "sku"):
            id_col = c
            break

    # API settings
    api_base = None
    if mode == "En ligne (API)":
        api_base = st.text_input(
            "URL de l'API",
            value=os.getenv("API_URL", "http://api:8000"),
            key="predict_api_url",
        )

    # Run prediction
    if st.button("Lancer la prediction", type="primary", use_container_width=True, key="predict_run"):
        with st.spinner("Preparation et prediction en cours..."):
            try:
                # Clean data
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

                st.success("Prediction terminee !")
                display_forecast_results(result, hist_df=hist_df)

            except Exception as e:
                st.error(f"Erreur : {e}")


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Prediction", layout="wide")
    st.markdown("""
    <style>
        .block-container { padding-top: 1rem; }
        h1 { color: #0F4C75; font-weight: 700; }
        h2 { color: #1B2631; font-weight: 600; border-bottom: 2px solid #0F4C75; padding-bottom: 0.3rem; }
        div[data-testid="stMetric"] {
            background: #FFFFFF; padding: 1rem; border-radius: 0.75rem;
            border-left: 4px solid #0F4C75; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
    </style>
    """, unsafe_allow_html=True)
    render_predict_ui()
