"""
Interface Streamlit pour la visualisation des resultats de prediction.

Charge un fichier de prevision (CSV) et affiche des graphiques interactifs
comparant l'historique et les predictions avec des metriques d'evaluation,
une detection d'anomalies et des options de granularite.
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PALETTE = {
    "primary": "#0F4C75",
    "primary_light": "#3282B8",
    "secondary": "#E77728",
    "success": "#2ECC71",
    "danger": "#E74C3C",
    "warning": "#F39C12",
}


def _find_forecast_files():
    """Search for forecast CSV files in common locations."""
    names = ["combined_forecast.csv", "forecast.csv", "forecasts.csv"]
    candidates = []
    for n in names:
        candidates += list(Path(".").glob(f"**/{n}"))
    if Path("models").exists():
        candidates += list(Path("models").glob("**/*.csv"))
    # Remove duplicates
    seen = set()
    unique = []
    for c in candidates:
        if str(c) not in seen:
            seen.add(str(c))
            unique.append(c)
    return unique


def _normalize_date_column(df: pd.DataFrame):
    """Detect and parse a date column in the DataFrame."""
    date_candidates = [c for c in df.columns if c.lower() in ("ds", "date", "day", "timestamp") or "date" in c.lower()]
    if date_candidates:
        col = date_candidates[0]
        try:
            df[col] = pd.to_datetime(df[col])
            return df, col
        except Exception:
            return df, None

    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.3:
                df[c] = parsed
                return df, c
        except Exception:
            continue
    return df, None


def _detect_y_columns(df: pd.DataFrame):
    """Auto-detect actual and predicted value columns."""
    y_candidates = [c for c in df.columns if c.lower() in ("y", "y_true", "value", "reel", "real")]
    yhat_candidates = [c for c in df.columns if "yhat" in c.lower() or "pred" in c.lower() or "forecast" in c.lower()]
    return (y_candidates[0] if y_candidates else None), (yhat_candidates[0] if yhat_candidates else None)


def render_visualize_ui():
    """Render the results visualization UI panel."""
    st.header("Visualisation des resultats")
    st.markdown(
        "Explorez les resultats de prediction avec des graphiques interactifs. "
        "Chargez un fichier de prevision ou utilisez les fichiers disponibles dans le projet."
    )

    # File source
    uploaded = st.file_uploader(
        "Charger un fichier de prevision (CSV)",
        type=["csv"],
        help="CSV contenant au moins une colonne de date et une colonne de prediction (yhat/forecast)",
    )

    df = None
    loaded_from = None

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            loaded_from = f"upload: {uploaded.name}"
        except Exception as e:
            st.error(f"Impossible de lire le CSV : {e}")
            return
    else:
        candidates = _find_forecast_files()
        if candidates:
            selected = st.selectbox(
                "Ou selectionner un fichier existant",
                options=["-- Selectionner --"] + [str(p) for p in candidates],
                key="viz_file_sel",
            )
            if selected and selected != "-- Selectionner --":
                try:
                    df = pd.read_csv(selected)
                    loaded_from = selected
                except Exception as e:
                    st.error(f"Impossible de lire {selected} : {e}")
                    return
        else:
            st.info(
                "Aucun fichier de prevision trouve dans le projet. "
                "Executez la prediction ou chargez un fichier CSV."
            )
            return

    if df is None:
        return

    # Diagnostics
    st.divider()
    st.subheader("Diagnostics du fichier")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Lignes", f"{len(df):,}")
    with c2:
        st.metric("Colonnes", f"{len(df.columns)}")
    with c3:
        st.metric("Source", loaded_from[:30] if loaded_from else "N/A")
    with c4:
        missing = df.isna().mean().mean() * 100
        st.metric("% manquant", f"{missing:.1f}%")

    tab_preview, tab_columns = st.tabs(["Apercu", "Colonnes detectees"])
    with tab_preview:
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    with tab_columns:
        st.markdown(", ".join([f"`{c}`" for c in df.columns]))

    # Normalize nested forecast column if present
    if "forecast" in df.columns and df["forecast"].dtype == object:
        try:
            sample = df["forecast"].dropna().iloc[0]
            parsed = json.loads(sample) if isinstance(sample, str) else sample
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                df = pd.DataFrame(parsed)
                st.info("Fichier normalise depuis la colonne `forecast`.")
        except Exception:
            pass

    # Date normalization
    df, date_col = _normalize_date_column(df)
    if date_col is None:
        st.error("Impossible d'identifier une colonne de date. Verifiez qu'il existe une colonne 'ds' ou 'date'.")
        return

    # Detect columns
    y_col, yhat_col = _detect_y_columns(df)
    upper_col = next((c for c in df.columns if "upper" in c.lower()), None)
    lower_col = next((c for c in df.columns if "lower" in c.lower()), None)

    # Optional historical upload
    hist_df = None
    if not y_col:
        st.markdown("""
        <div style="background: #FEF9E7; border-left: 4px solid #F39C12; 
                    padding: 0.6rem 1rem; border-radius: 0 0.5rem 0.5rem 0; margin: 0.5rem 0;">
            <strong>Aucune colonne historique detectee.</strong> 
            Vous pouvez charger un fichier historique pour la comparaison.
        </div>
        """, unsafe_allow_html=True)
        uploaded_hist = st.file_uploader(
            "Charger un CSV historique (optionnel)",
            type=["csv"],
            key="hist_upload",
        )
        if uploaded_hist is not None:
            try:
                hist_df = pd.read_csv(uploaded_hist)
                hist_df[date_col] = pd.to_datetime(hist_df[date_col])
                for c in ("value", "y", "reel"):
                    if c in hist_df.columns:
                        y_col = c
                        break
            except Exception as e:
                st.warning(f"Impossible de lire l'historique : {e}")
    else:
        hist_df = df[[date_col, y_col]].dropna().copy()

    # Forecast df
    if yhat_col and yhat_col in df.columns:
        fc_df = df[[date_col, yhat_col]].dropna().copy()
        fc_df = fc_df.rename(columns={yhat_col: "yhat"})
    else:
        alt = [c for c in df.columns if "yhat" in c.lower() or "pred" in c.lower() or c.lower().startswith("forecast")]
        if alt:
            fc_df = df[[date_col, alt[0]]].dropna().rename(columns={alt[0]: "yhat"})
        else:
            fc_df = pd.DataFrame(columns=[date_col, "yhat"])

    # Add confidence intervals
    if upper_col and upper_col in df.columns:
        df[upper_col] = pd.to_numeric(df[upper_col], errors="coerce")
        fc_df = fc_df.merge(df[[date_col, upper_col]], on=date_col, how="left")
        fc_df = fc_df.rename(columns={upper_col: "yhat_upper"})
    if lower_col and lower_col in df.columns:
        df[lower_col] = pd.to_numeric(df[lower_col], errors="coerce")
        fc_df = fc_df.merge(df[[date_col, lower_col]], on=date_col, how="left")
        fc_df = fc_df.rename(columns={lower_col: "yhat_lower"})

    # Combine historical and forecast
    hist_renamed = (
        hist_df.rename(columns={date_col: "ds", y_col: "y"})
        if hist_df is not None and y_col
        else pd.DataFrame(columns=["ds", "y"])
    )
    fc_renamed = (
        fc_df.rename(columns={date_col: "ds"})
        if not fc_df.empty
        else pd.DataFrame(columns=["ds", "yhat"])
    )
    combined = pd.merge(hist_renamed, fc_renamed, on="ds", how="outer")
    if not combined.empty:
        combined["ds"] = pd.to_datetime(combined["ds"])
        combined = combined.sort_values("ds")

    # Visualization controls
    st.divider()
    st.subheader("Parametres de visualisation")
    c1, c2, c3 = st.columns(3)

    with c1:
        if not combined.empty:
            min_date = combined["ds"].min().date()
            max_date = combined["ds"].max().date()
            date_range = st.date_input(
                "Periode",
                value=(min_date, max_date),
                key="viz_date_range",
            )
        else:
            date_range = None

    with c2:
        agg_level = st.selectbox(
            "Granularite",
            options=["D", "W", "M"],
            format_func=lambda x: {"D": "Journalier", "W": "Hebdomadaire", "M": "Mensuel"}[x],
            key="viz_agg",
        )

    with c3:
        anomaly_pct = st.slider(
            "Seuil d'ecart (%) pour alerte",
            0.0, 200.0, 50.0,
            help="Pourcentage d'ecart entre reel et predit pour signaler une anomalie",
            key="viz_anomaly",
        )

    # Apply date filter
    plot_df = combined.copy()
    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        plot_df = plot_df[(plot_df["ds"] >= start) & (plot_df["ds"] <= end)]

    # Apply aggregation
    if not plot_df.empty and agg_level != "D":
        numeric_cols = plot_df.select_dtypes(include=["number"]).columns.tolist()
        plot_resampled = plot_df.set_index("ds")[numeric_cols].resample(agg_level).mean().reset_index()
    else:
        plot_resampled = plot_df

    # Main chart
    st.divider()
    st.subheader("Ventes reelles vs Predictions")

    fig = go.Figure()
    if "y" in plot_resampled.columns and not plot_resampled["y"].isna().all():
        fig.add_trace(go.Scatter(
            x=plot_resampled["ds"], y=plot_resampled["y"],
            mode="lines+markers", name="Ventes reelles",
            line=dict(color=PALETTE["primary"], width=2),
            marker=dict(size=4),
            hovertemplate="Date: %{x}<br>Reel: %{y:,.2f}<extra></extra>",
        ))
    if "yhat" in plot_resampled.columns and not plot_resampled["yhat"].isna().all():
        fig.add_trace(go.Scatter(
            x=plot_resampled["ds"], y=plot_resampled["yhat"],
            mode="lines+markers", name="Prediction",
            line=dict(color=PALETTE["secondary"], width=2, dash="dash"),
            marker=dict(size=4),
            hovertemplate="Date: %{x}<br>Prediction: %{y:,.2f}<extra></extra>",
        ))
    if "yhat_upper" in plot_resampled.columns and "yhat_lower" in plot_resampled.columns:
        fig.add_trace(go.Scatter(
            x=plot_resampled["ds"], y=plot_resampled["yhat_upper"],
            mode="lines", name="Upper", line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=plot_resampled["ds"], y=plot_resampled["yhat_lower"],
            mode="lines", name="Lower", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(231, 119, 40, 0.15)", showlegend=False,
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

    # Error distribution chart
    if "y" in combined.columns and "yhat" in combined.columns:
        eval_df = combined.dropna(subset=["y", "yhat"]).copy()
        if not eval_df.empty:
            eval_df["error"] = eval_df["y"] - eval_df["yhat"]
            eval_df["abs_error"] = eval_df["error"].abs()
            eval_df["pct_error"] = (eval_df["error"] / eval_df["y"].replace(0, np.nan)).abs() * 100

            st.divider()

            # Metrics
            st.subheader("Metriques d'evaluation")
            mae = float(eval_df["abs_error"].mean())
            rmse = float(np.sqrt((eval_df["error"] ** 2).mean()))
            mape = float(eval_df["pct_error"].dropna().mean())
            r2 = 1 - (eval_df["error"] ** 2).sum() / ((eval_df["y"] - eval_df["y"].mean()) ** 2).sum()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("MAE", f"{mae:.2f}")
            with c2:
                st.metric("RMSE", f"{rmse:.2f}")
            with c3:
                st.metric("MAPE", f"{mape:.2f}%")
            with c4:
                st.metric("R2", f"{float(r2):.4f}")

            # Error analysis charts
            st.subheader("Analyse des erreurs")
            c1, c2 = st.columns(2)
            with c1:
                fig_err_hist = px.histogram(
                    eval_df, x="error", nbins=30,
                    color_discrete_sequence=[PALETTE["primary"]],
                    labels={"error": "Erreur (Reel - Predit)"},
                    marginal="box",
                )
                fig_err_hist.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_err_hist, use_container_width=True)

            with c2:
                fig_err_scatter = px.scatter(
                    eval_df, x="y", y="yhat",
                    color_discrete_sequence=[PALETTE["primary"]],
                    labels={"y": "Valeur reelle", "yhat": "Prediction"},
                    opacity=0.7,
                )
                # Add perfect prediction line
                min_val = min(eval_df["y"].min(), eval_df["yhat"].min())
                max_val = max(eval_df["y"].max(), eval_df["yhat"].max())
                fig_err_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode="lines", name="Prediction parfaite",
                    line=dict(color=PALETTE["danger"], dash="dash", width=1.5),
                ))
                fig_err_scatter.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_err_scatter, use_container_width=True)

            # Anomaly detection
            st.divider()
            st.subheader("Detection d'anomalies")

            if "yhat_upper" in combined.columns and "yhat_lower" in combined.columns:
                mask = (combined["y"] > combined["yhat_upper"]) | (combined["y"] < combined["yhat_lower"])
                anomalies = combined.loc[mask, ["ds", "y", "yhat", "yhat_lower", "yhat_upper"]].dropna()
            else:
                combined_temp = combined.dropna(subset=["y", "yhat"]).copy()
                combined_temp["pct_diff"] = ((combined_temp["y"] - combined_temp["yhat"]) / combined_temp["y"].replace(0, np.nan)).abs() * 100
                anomalies = combined_temp[combined_temp["pct_diff"] > anomaly_pct][["ds", "y", "yhat", "pct_diff"]]

            if not anomalies.empty:
                st.markdown(
                    f"<div style='background: #FDEDEC; border-left: 4px solid {PALETTE['danger']}; "
                    f"padding: 0.6rem 1rem; border-radius: 0 0.5rem 0.5rem 0;'>"
                    f"<strong>{len(anomalies)} anomalie(s) detectee(s)</strong> "
                    f"(seuil d'ecart: {anomaly_pct:.0f}%)</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(anomalies.sort_values("ds").head(50), use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    f"<div style='background: #EAFAF1; border-left: 4px solid {PALETTE['success']}; "
                    f"padding: 0.6rem 1rem; border-radius: 0 0.5rem 0.5rem 0;'>"
                    f"<strong>Aucune anomalie majeure detectee</strong></div>",
                    unsafe_allow_html=True,
                )
    else:
        st.info("Pas assez de points communs entre historique et previsions pour calculer des metriques.")

    # Download section
    st.divider()
    st.subheader("Export des donnees")

    if not combined.empty:
        c1, c2 = st.columns(2)
        with c1:
            csv_combined = combined.sort_values("ds").to_csv(index=False).encode("utf-8")
            st.download_button(
                "Telecharger donnees combinees (CSV)",
                data=csv_combined,
                file_name="combined_forecast_full.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            if "y" in combined.columns and "yhat" in combined.columns:
                eval_data = combined.dropna(subset=["y", "yhat"]).copy()
                if not eval_data.empty:
                    eval_data["error"] = eval_data["y"] - eval_data["yhat"]
                    eval_data["abs_error"] = eval_data["error"].abs()
                    csv_eval = eval_data.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Telecharger analyse d'erreurs (CSV)",
                        data=csv_eval,
                        file_name="error_analysis.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
    else:
        st.info("Aucune donnee combinee disponible pour l'export.")
