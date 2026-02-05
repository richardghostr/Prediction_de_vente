"""
Prediction de Vente -- Tableau de bord interactif ultra-complet.

Streamlit dashboard that provides:
  1. Data Upload & Auto-detection
  2. Exploratory Data Analysis (statistics, distributions, correlations)
  3. Time-series visualisation (trends, seasonality, decomposition)
  4. Prediction panel (online via API or offline via local model)
  5. Model info & evaluation metrics
  6. Data export

Launch:
    streamlit run src/ui/streamlit_app.py
"""
import os
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Optional imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from src.data.clean import clean_dataframe
except Exception:
    clean_dataframe = None

try:
    from src.data.features import build_feature_pipeline
except Exception:
    build_feature_pipeline = None

try:
    from src.models.predict import predict_series, get_model, get_model_path
except Exception:
    predict_series = None
    get_model = None
    get_model_path = None


# =====================================================================
# CONFIGURATION
# =====================================================================
PALETTE = {
    "primary": "#1B6B93",
    "secondary": "#F39C12",
    "success": "#27AE60",
    "danger": "#E74C3C",
    "bg": "#F8F9FA",
    "text": "#2C3E50",
}

st.set_page_config(
    page_title="Prediction de Vente",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a polished look
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stMetric > div { background: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    div[data-testid="stSidebar"] { background: #f0f4f8; }
    h1 { color: #1B6B93; }
    h2, h3 { color: #2C3E50; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# HELPERS
# =====================================================================

def detect_columns(df: pd.DataFrame):
    """Auto-detect date and value columns from a DataFrame."""
    date_candidates = [
        "date", "Date", "ds", "timestamp", "datetime", "time",
        "sale_date", "saledate", "order_date", "created_at",
    ]
    date_col = None
    for c in date_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        best_date, best_score = None, 0
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

    value_candidates = [
        "quantity", "qty", "value", "y", "sales",
        "amount", "revenue", "unit_price",
    ]
    value_col = None
    for c in value_candidates:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        best_num, best_cnt = None, -1
        for col in df.select_dtypes(include=["number"]).columns:
            cnt = df[col].notna().sum()
            if cnt > best_cnt:
                best_cnt = cnt
                best_num = col
        value_col = best_num

    return (date_col or "date"), value_col


def compute_stats(series: pd.Series) -> dict:
    """Compute descriptive statistics for a numeric series."""
    return {
        "Count": int(series.count()),
        "Mean": round(float(series.mean()), 2),
        "Std": round(float(series.std()), 2),
        "Min": round(float(series.min()), 2),
        "25%": round(float(series.quantile(0.25)), 2),
        "Median": round(float(series.median()), 2),
        "75%": round(float(series.quantile(0.75)), 2),
        "Max": round(float(series.max()), 2),
        "Skewness": round(float(series.skew()), 3),
        "Kurtosis": round(float(series.kurtosis()), 3),
    }


def prepare_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_col: str = None,
    group_filter: str = "All",
    aggregate: bool = True,
) -> pd.DataFrame:
    """Clean, filter, and aggregate data into a proper time series."""
    ts = df.copy()

    # Filter by group if needed
    if group_col and group_col != "None" and group_filter != "All":
        ts = ts[ts[group_col].astype(str) == group_filter]

    # Parse dates
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])

    # Aggregate duplicate dates
    if ts.duplicated(subset=[date_col], keep=False).any() and aggregate:
        agg_dict = {}
        for c in ts.columns:
            if c == date_col:
                continue
            if pd.api.types.is_numeric_dtype(ts[c]):
                agg_dict[c] = "sum"
            else:
                agg_dict[c] = "first"
        ts = ts.groupby(date_col, dropna=False).agg(agg_dict).reset_index()

    ts = ts.sort_values(date_col).reset_index(drop=True)
    return ts


# =====================================================================
# SIDEBAR
# =====================================================================

with st.sidebar:
    st.title("Prediction de Vente")
    st.caption("Tableau de bord interactif")
    st.divider()

    uploaded = st.file_uploader("Charger un fichier CSV", type=["csv"])

    st.divider()
    st.subheader("Parametres de prediction")
    horizon = st.slider("Horizon (jours)", 1, 365, 30)
    offline = st.checkbox("Mode hors-ligne (pas d'appel API)", value=True)

    st.divider()
    st.subheader("Options avancees")
    aggregate = st.checkbox("Agreger les dates dupliquees (somme)", value=True, key="agg")
    archive_outputs = st.checkbox("Archiver les sorties", value=False, key="archive")

    st.divider()
    # Model status
    st.subheader("Statut du modele")
    if get_model_path:
        try:
            mp = get_model_path()
            if mp:
                st.success(f"Modele: {Path(mp).name}")
            else:
                st.warning("Aucun artefact de modele trouve")
        except Exception:
            st.warning("Impossible de charger le modele")
    else:
        st.info("Module de prediction non disponible")


# =====================================================================
# MAIN CONTENT
# =====================================================================

if uploaded is None:
    st.title("Bienvenue sur le Dashboard de Prediction de Vente")
    st.markdown("""
    Ce tableau de bord vous permet de :

    - **Visualiser** vos donnees de vente avec des graphiques interactifs
    - **Analyser** les tendances, la saisonnalite et les distributions
    - **Predire** les ventes futures avec un modele XGBoost entraine
    - **Exporter** les resultats et les previsions

    **Pour commencer**, chargez un fichier CSV dans la barre laterale.

    Le fichier doit contenir au minimum une colonne de date et une colonne numerique (ventes, quantite, etc.).
    """)

    # Show sample data info
    sample_path = PROJECT_ROOT / "sample.csv"
    if sample_path.exists():
        st.info(f"Un fichier d'exemple est disponible dans le projet : `{sample_path.name}`")

    st.stop()


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Erreur de lecture du CSV : {e}")
    st.stop()

if df_raw.empty:
    st.error("Le fichier CSV est vide.")
    st.stop()


# ---------------------------------------------------------------------------
# Column detection & sidebar controls
# ---------------------------------------------------------------------------
detected_date, detected_value = detect_columns(df_raw)

with st.sidebar:
    st.divider()
    st.subheader("Colonnes detectees")
    date_col = st.selectbox(
        "Colonne de date",
        options=df_raw.columns.tolist(),
        index=df_raw.columns.tolist().index(detected_date) if detected_date in df_raw.columns else 0,
        key="date_col_sel",
    )
    # Default to quantity-like column if available
    value_default = detected_value or df_raw.select_dtypes(include=["number"]).columns[0] if not df_raw.select_dtypes(include=["number"]).empty else df_raw.columns[0]
    numeric_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()
    value_col = st.selectbox(
        "Colonne de valeur",
        options=numeric_cols if numeric_cols else df_raw.columns.tolist(),
        index=numeric_cols.index(value_default) if value_default in numeric_cols else 0,
        key="value_col_sel",
    )

    # Group column
    cat_cols = [c for c in df_raw.select_dtypes(include=["object", "category"]).columns if c != date_col]
    if cat_cols:
        group_col = st.selectbox(
            "Colonne de regroupement",
            options=["None"] + cat_cols,
            index=0,
            key="group_col_sel",
        )
    else:
        group_col = "None"

    group_filter = "All"
    if group_col != "None":
        unique_vals = sorted(df_raw[group_col].dropna().astype(str).unique().tolist())
        group_filter = st.selectbox(
            "Filtrer par groupe",
            options=["All"] + unique_vals,
            index=0,
            key="group_filter_sel",
        )


# ---------------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------------
df_ts = prepare_time_series(
    df_raw, date_col, value_col,
    group_col=group_col if group_col != "None" else None,
    group_filter=group_filter,
    aggregate=aggregate,
)

if df_ts.empty:
    st.error("Aucune donnee disponible apres filtrage et aggregation.")
    st.stop()

# Clean data
if clean_dataframe:
    try:
        df_clean = clean_dataframe(df_ts, date_col=date_col, value_col=value_col)
    except Exception:
        df_clean = df_ts.copy()
else:
    df_clean = df_ts.copy()

# Build features
features_df = None
if build_feature_pipeline:
    try:
        features_df = build_feature_pipeline(
            df_clean, date_col=date_col, value_col=value_col,
            lags=[1, 7], windows=[3, 7],
        )
    except Exception:
        features_df = None

# Archive if requested
if archive_outputs:
    try:
        interim_dir = Path("data/interim")
        processed_dir = Path("data/processed")
        interim_dir.mkdir(parents=True, exist_ok=True)
        ts_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        df_clean.to_csv(interim_dir / f"uploaded_clean_{ts_str}.csv", index=False)
        if features_df is not None:
            processed_dir.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(processed_dir / f"uploaded_features_{ts_str}.csv", index=False)
        st.sidebar.success("Sorties archivees")
    except Exception as e:
        st.sidebar.warning(f"Archivage echoue : {e}")


# =====================================================================
# TABS
# =====================================================================

tab_overview, tab_eda, tab_ts, tab_predict, tab_model, tab_data = st.tabs([
    "Vue d'ensemble",
    "Analyse exploratoire",
    "Series temporelles",
    "Prediction",
    "Modele & Metriques",
    "Donnees & Export",
])


# =====================================================================
# TAB 1 : VUE D'ENSEMBLE
# =====================================================================
with tab_overview:
    st.header("Vue d'ensemble des donnees")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nb de lignes", f"{len(df_raw):,}")
    with col2:
        st.metric("Nb de colonnes", f"{len(df_raw.columns)}")
    with col3:
        if value_col in df_clean.columns:
            total = df_clean[value_col].sum()
            st.metric("Total (valeur)", f"{total:,.0f}")
    with col4:
        if date_col in df_clean.columns:
            date_range = (df_clean[date_col].max() - df_clean[date_col].min()).days
            st.metric("Plage temporelle", f"{date_range} jours")

    st.divider()

    # Quick summary
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Apercu des donnees brutes")
        st.dataframe(df_raw.head(20), use_container_width=True, height=400)

    with c2:
        st.subheader("Types de colonnes")
        type_info = pd.DataFrame({
            "Colonne": df_raw.columns,
            "Type": df_raw.dtypes.astype(str).values,
            "Non-null": df_raw.notna().sum().values,
            "% manquant": (df_raw.isna().mean() * 100).round(1).values,
        })
        st.dataframe(type_info, use_container_width=True, height=400)

    # Missing values heatmap
    if df_raw.isna().any().any():
        st.subheader("Carte des valeurs manquantes")
        missing_pct = df_raw.isna().mean() * 100
        missing_df = missing_pct[missing_pct > 0].sort_values(ascending=False)
        if not missing_df.empty:
            fig_missing = px.bar(
                x=missing_df.index, y=missing_df.values,
                labels={"x": "Colonne", "y": "% manquant"},
                color=missing_df.values,
                color_continuous_scale=["#27AE60", "#F39C12", "#E74C3C"],
            )
            fig_missing.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_missing, use_container_width=True)


# =====================================================================
# TAB 2 : ANALYSE EXPLORATOIRE
# =====================================================================
with tab_eda:
    st.header("Analyse exploratoire des donnees")

    if value_col in df_clean.columns:
        val_series = df_clean[value_col].dropna()

        # Statistics
        st.subheader("Statistiques descriptives")
        stats = compute_stats(val_series)
        stat_cols = st.columns(5)
        for i, (k, v) in enumerate(stats.items()):
            with stat_cols[i % 5]:
                st.metric(k, v)

        st.divider()

        # Distribution
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Distribution de '{value_col}'")
            fig_hist = px.histogram(
                df_clean, x=value_col, nbins=50,
                color_discrete_sequence=[PALETTE["primary"]],
                marginal="box",
            )
            fig_hist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.subheader("Box plot")
            if group_col != "None" and group_col in df_ts.columns:
                fig_box = px.box(
                    df_ts, x=group_col, y=value_col,
                    color=group_col,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
            else:
                fig_box = px.box(df_clean, y=value_col, color_discrete_sequence=[PALETTE["primary"]])
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)

        # Correlation matrix
        num_df = df_clean.select_dtypes(include=["number"])
        if len(num_df.columns) > 1:
            st.subheader("Matrice de correlation")
            corr = num_df.corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                aspect="auto",
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

        # Category breakdown
        if group_col != "None" and group_col in df_raw.columns:
            st.subheader(f"Repartition par '{group_col}'")
            c1, c2 = st.columns(2)
            with c1:
                cat_counts = df_raw[group_col].value_counts()
                fig_pie = px.pie(
                    values=cat_counts.values,
                    names=cat_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                if value_col in df_raw.columns:
                    grp_agg = df_raw.groupby(group_col)[value_col].sum().sort_values(ascending=False)
                    fig_bar = px.bar(
                        x=grp_agg.index, y=grp_agg.values,
                        labels={"x": group_col, "y": f"Total {value_col}"},
                        color=grp_agg.index,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_bar.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning(f"La colonne '{value_col}' n'a pas ete trouvee dans les donnees nettoyees.")


# =====================================================================
# TAB 3 : SERIES TEMPORELLES
# =====================================================================
with tab_ts:
    st.header("Analyse des series temporelles")

    if date_col in df_clean.columns and value_col in df_clean.columns:
        ts_data = df_clean[[date_col, value_col]].dropna().sort_values(date_col)
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])

        # Main time series chart
        st.subheader("Evolution temporelle")
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts_data[date_col], y=ts_data[value_col],
            mode="lines+markers",
            name="Valeur",
            line=dict(color=PALETTE["primary"], width=2),
            marker=dict(size=3),
        ))
        # Add 7-day rolling average
        if len(ts_data) > 7:
            ts_data["rolling_7"] = ts_data[value_col].rolling(7, min_periods=1).mean()
            fig_ts.add_trace(go.Scatter(
                x=ts_data[date_col], y=ts_data["rolling_7"],
                mode="lines",
                name="Moyenne mobile (7j)",
                line=dict(color=PALETTE["secondary"], width=2, dash="dash"),
            ))
        fig_ts.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title=value_col,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Aggregated views
        st.subheader("Agregations temporelles")
        agg_period = st.radio(
            "Periode d'agregation",
            options=["Journalier", "Hebdomadaire", "Mensuel"],
            horizontal=True,
            key="ts_agg_period",
        )

        freq_map = {"Journalier": "D", "Hebdomadaire": "W", "Mensuel": "ME"}
        freq = freq_map[agg_period]

        ts_agg = ts_data.set_index(date_col).resample(freq)[value_col].agg(["sum", "mean", "count"]).reset_index()
        ts_agg.columns = ["Date", "Total", "Moyenne", "Nb transactions"]

        c1, c2 = st.columns(2)
        with c1:
            fig_agg_bar = px.bar(
                ts_agg, x="Date", y="Total",
                color_discrete_sequence=[PALETTE["primary"]],
                labels={"Total": f"Total {value_col}"},
            )
            fig_agg_bar.update_layout(height=400)
            st.plotly_chart(fig_agg_bar, use_container_width=True)

        with c2:
            fig_agg_line = px.line(
                ts_agg, x="Date", y="Moyenne",
                color_discrete_sequence=[PALETTE["secondary"]],
                labels={"Moyenne": f"Moyenne {value_col}"},
            )
            fig_agg_line.update_layout(height=400)
            st.plotly_chart(fig_agg_line, use_container_width=True)

        # Day-of-week / Month analysis
        st.subheader("Saisonnalite")
        c1, c2 = st.columns(2)
        with c1:
            ts_data["dow"] = ts_data[date_col].dt.day_name()
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_agg = ts_data.groupby("dow")[value_col].mean().reindex(dow_order)
            fig_dow = px.bar(
                x=dow_agg.index, y=dow_agg.values,
                labels={"x": "Jour de la semaine", "y": f"Moyenne {value_col}"},
                color_discrete_sequence=[PALETTE["primary"]],
            )
            fig_dow.update_layout(height=350, title="Moyenne par jour de la semaine")
            st.plotly_chart(fig_dow, use_container_width=True)

        with c2:
            ts_data["month_name"] = ts_data[date_col].dt.month_name()
            month_order = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            present_months = [m for m in month_order if m in ts_data["month_name"].values]
            month_agg = ts_data.groupby("month_name")[value_col].mean().reindex(present_months)
            fig_month = px.bar(
                x=month_agg.index, y=month_agg.values,
                labels={"x": "Mois", "y": f"Moyenne {value_col}"},
                color_discrete_sequence=[PALETTE["secondary"]],
            )
            fig_month.update_layout(height=350, title="Moyenne par mois")
            st.plotly_chart(fig_month, use_container_width=True)

        # Heatmap: day-of-week vs week number
        if len(ts_data) > 14:
            st.subheader("Heatmap activite (semaine x jour)")
            ts_data["week"] = ts_data[date_col].dt.isocalendar().week.astype(int)
            ts_data["dow_num"] = ts_data[date_col].dt.dayofweek
            heatmap_data = ts_data.pivot_table(
                index="dow_num", columns="week", values=value_col, aggfunc="sum"
            ).fillna(0)
            fig_hm = px.imshow(
                heatmap_data,
                labels=dict(x="Semaine", y="Jour (0=Lun)", color=value_col),
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_hm.update_layout(height=300)
            st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.warning("Colonnes de date/valeur introuvables apres nettoyage.")


# =====================================================================
# TAB 4 : PREDICTION
# =====================================================================
with tab_predict:
    st.header("Prevision des ventes")

    if value_col not in df_clean.columns or date_col not in df_clean.columns:
        st.error("Les colonnes de date et de valeur doivent etre presentes pour la prediction.")
        st.stop()

    st.info(
        f"Mode : {'Hors-ligne (modele local)' if offline else 'En ligne (API)'}  |  "
        f"Horizon : {horizon} jours  |  "
        f"Points de donnees : {len(df_clean)}"
    )

    if st.button("Lancer la prediction", type="primary", use_container_width=True):
        # Build series payload from cleaned data
        series_payload = []
        for _, row in df_clean.iterrows():
            try:
                ds = pd.to_datetime(row[date_col])
                date_iso = ds.date().isoformat()
            except Exception:
                date_iso = str(row[date_col])
            item = {"date": date_iso}
            if value_col in df_clean.columns:
                val = row[value_col]
                item["value"] = None if pd.isna(val) else float(val)
            # Include extra numeric columns
            for c in df_clean.select_dtypes(include=["number"]).columns:
                if c != value_col and pd.notna(row[c]):
                    item[c] = float(row[c])
            series_payload.append(item)

        forecast_result = None

        if offline:
            # Offline prediction
            with st.spinner("Prediction en cours (mode hors-ligne)..."):
                if predict_series is not None:
                    try:
                        forecast_result = predict_series(series_payload, horizon=horizon)
                    except Exception as e:
                        st.warning(f"Modele local echoue : {e}. Utilisation du repli naive.")

                # Naive fallback
                if forecast_result is None:
                    last_val = None
                    for s in reversed(series_payload):
                        if s.get("value") is not None:
                            last_val = s["value"]
                            last_date = pd.to_datetime(s["date"])
                            break
                    if last_val is None:
                        st.error("Aucune valeur observee pour la prediction.")
                        st.stop()
                    forecast_result = {"id": "offline", "forecast": []}
                    for i in range(1, horizon + 1):
                        ds = (last_date + timedelta(days=i)).date().isoformat()
                        forecast_result["forecast"].append({"ds": ds, "yhat": float(last_val)})
        else:
            # Online via API
            with st.spinner("Appel API en cours..."):
                try:
                    import requests as req_lib
                    api_base = os.getenv("API_URL", "http://api:8000")
                    url = f"{api_base.rstrip('/')}/predict?horizon={horizon}"
                    payload = {"id": "dashboard_upload", "series": series_payload}
                    resp = req_lib.post(url, json=payload, timeout=30)
                    if resp.status_code != 200:
                        st.error(f"Erreur API {resp.status_code}: {resp.text}")
                        st.stop()
                    forecast_result = resp.json()
                except Exception as e:
                    st.error(f"Erreur lors de l'appel API : {e}")
                    st.stop()

        if forecast_result is None:
            st.error("La prediction a echoue.")
            st.stop()

        # Parse forecast
        fc = pd.DataFrame(forecast_result.get("forecast", []))
        if fc.empty:
            st.warning("La prediction n'a retourne aucun resultat.")
            st.stop()

        fc["ds"] = pd.to_datetime(fc["ds"])

        # Historical data
        hist = df_clean[[date_col, value_col]].dropna().copy()
        hist = hist.rename(columns={date_col: "ds", value_col: "y"})
        hist["ds"] = pd.to_datetime(hist["ds"])
        if hist["ds"].duplicated().any():
            hist = hist.groupby("ds", as_index=False).agg({"y": "sum"})

        # Store in session state for other tabs
        st.session_state["forecast_result"] = forecast_result
        st.session_state["fc"] = fc
        st.session_state["hist"] = hist

        # ---- Display results ----
        st.success(f"Prediction terminee : {len(fc)} points generes")

        # Metrics (if available)
        if forecast_result.get("metrics"):
            st.subheader("Metriques du modele")
            met_cols = st.columns(min(len(forecast_result["metrics"]), 4))
            for i, (k, v) in enumerate(forecast_result["metrics"].items()):
                with met_cols[i % len(met_cols)]:
                    if isinstance(v, (int, float)):
                        st.metric(k, f"{v:.4f}")
                    else:
                        st.metric(k, str(v))

        # Main forecast chart
        st.subheader("Graphique de prevision")
        fig_fc = go.Figure()

        # Historical
        fig_fc.add_trace(go.Scatter(
            x=hist["ds"], y=hist["y"],
            mode="lines",
            name="Donnees historiques",
            line=dict(color=PALETTE["primary"], width=2),
        ))

        # Forecast
        fig_fc.add_trace(go.Scatter(
            x=fc["ds"], y=fc["yhat"],
            mode="lines+markers",
            name="Prediction",
            line=dict(color=PALETTE["secondary"], width=2, dash="dash"),
            marker=dict(size=4),
        ))

        # Confidence intervals
        if "yhat_upper" in fc.columns and "yhat_lower" in fc.columns:
            fig_fc.add_trace(go.Scatter(
                x=fc["ds"], y=fc["yhat_upper"],
                mode="lines", line=dict(width=0),
                showlegend=False,
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc["ds"], y=fc["yhat_lower"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(243, 156, 18, 0.15)",
                name="Intervalle de confiance",
            ))

        # Transition line
        if not hist.empty:
            last_hist_date = hist["ds"].iloc[-1]
            last_hist_val = hist["y"].iloc[-1]
            first_fc_date = fc["ds"].iloc[0]
            first_fc_val = fc["yhat"].iloc[0]
            fig_fc.add_trace(go.Scatter(
                x=[last_hist_date, first_fc_date],
                y=[last_hist_val, first_fc_val],
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
            ))

        fig_fc.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Valeur",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast summary
        st.subheader("Resume de la prevision")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Moyenne prevue", f"{fc['yhat'].mean():.1f}")
        with c2:
            st.metric("Min prevu", f"{fc['yhat'].min():.1f}")
        with c3:
            st.metric("Max prevu", f"{fc['yhat'].max():.1f}")
        with c4:
            if not hist.empty:
                last_val = hist["y"].iloc[-1]
                avg_fc = fc["yhat"].mean()
                pct_change = ((avg_fc - last_val) / last_val * 100) if last_val != 0 else 0
                st.metric("Variation (%)", f"{pct_change:+.1f}%")

        # Monthly breakdown of forecast
        if len(fc) > 7:
            st.subheader("Prevision par periode")
            fc_monthly = fc.set_index("ds").resample("W")["yhat"].agg(["sum", "mean", "count"]).reset_index()
            fc_monthly.columns = ["Semaine", "Total", "Moyenne", "Nb jours"]
            fig_fc_bar = px.bar(
                fc_monthly, x="Semaine", y="Total",
                color_discrete_sequence=[PALETTE["secondary"]],
            )
            fig_fc_bar.update_layout(height=350)
            st.plotly_chart(fig_fc_bar, use_container_width=True)

        # Download forecast
        st.divider()
        csv_bytes = fc.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Telecharger les previsions (CSV)",
            data=csv_bytes,
            file_name="forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =====================================================================
# TAB 5 : MODELE & METRIQUES
# =====================================================================
with tab_model:
    st.header("Informations sur le modele et metriques")

    # Model info
    st.subheader("Modele charge")
    if get_model is not None:
        try:
            model = get_model()
            mp = get_model_path()
            st.success(f"Modele : {Path(mp).name if mp else 'N/A'}")
            st.write(f"**Type** : `{type(model).__name__}`")

            # Feature importance
            feature_names = None
            importances = None
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                if hasattr(model, "feature_names_in_"):
                    feature_names = list(model.feature_names_in_)
                elif hasattr(model, "get_booster"):
                    try:
                        feature_names = model.get_booster().feature_names
                    except Exception:
                        pass

            if importances is not None and feature_names is not None:
                st.subheader("Importance des features")
                imp_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances,
                }).sort_values("Importance", ascending=True)

                fig_imp = px.bar(
                    imp_df, x="Importance", y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=["#F8F9FA", PALETTE["primary"]],
                )
                fig_imp.update_layout(height=max(300, len(feature_names) * 25), showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)
        except FileNotFoundError:
            st.warning("Aucun artefact de modele trouve dans `models/artifacts/`.")
        except Exception as e:
            st.error(f"Erreur lors du chargement du modele : {e}")
    else:
        st.info("Module `predict` non disponible.")

    # Features preview
    if features_df is not None:
        st.divider()
        st.subheader("Apercu des features calculees")
        st.dataframe(features_df.head(20), use_container_width=True)

    # Saved metrics files
    st.divider()
    st.subheader("Metriques sauvegardees")
    metrics_dir = Path(config.MODELS_METRICS_DIR) if hasattr(config, "MODELS_METRICS_DIR") else Path("models/metrics")
    if metrics_dir.exists():
        json_files = sorted(metrics_dir.glob("*.json"), reverse=True)
        if json_files:
            selected_metric = st.selectbox(
                "Selectionner un fichier de metriques",
                options=[f.name for f in json_files],
                key="metric_file_sel",
            )
            if selected_metric:
                try:
                    import json
                    with open(metrics_dir / selected_metric) as f:
                        data = json.load(f)
                    st.json(data)
                except Exception as e:
                    st.error(f"Erreur de lecture : {e}")
        else:
            st.info("Aucun fichier de metriques trouve.")
    else:
        st.info("Le dossier de metriques n'existe pas encore.")


# =====================================================================
# TAB 6 : DONNEES & EXPORT
# =====================================================================
with tab_data:
    st.header("Donnees detaillees et export")

    data_view = st.radio(
        "Vue des donnees",
        options=["Brutes", "Nettoyees", "Avec features", "Previsions"],
        horizontal=True,
        key="data_view_sel",
    )

    if data_view == "Brutes":
        st.subheader("Donnees brutes")
        st.dataframe(df_raw, use_container_width=True, height=500)
        csv = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button("Telecharger (CSV)", csv, "donnees_brutes.csv", "text/csv")

    elif data_view == "Nettoyees":
        st.subheader("Donnees nettoyees")
        st.dataframe(df_clean, use_container_width=True, height=500)
        csv = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button("Telecharger (CSV)", csv, "donnees_nettoyees.csv", "text/csv")

    elif data_view == "Avec features":
        if features_df is not None:
            st.subheader("Donnees avec features")
            st.dataframe(features_df, use_container_width=True, height=500)
            csv = features_df.to_csv(index=False).encode("utf-8")
            st.download_button("Telecharger (CSV)", csv, "donnees_features.csv", "text/csv")
        else:
            st.info("Les features n'ont pas pu etre calculees.")

    elif data_view == "Previsions":
        if "fc" in st.session_state and "hist" in st.session_state:
            fc = st.session_state["fc"]
            hist = st.session_state["hist"]
            combined = pd.merge(hist, fc, on="ds", how="outer").sort_values("ds")
            st.subheader("Donnees combinees (historique + prevision)")
            st.dataframe(combined, use_container_width=True, height=500)
            csv = combined.to_csv(index=False).encode("utf-8")
            st.download_button("Telecharger (CSV)", csv, "combined_forecast.csv", "text/csv")
        else:
            st.info("Lancez d'abord une prediction dans l'onglet 'Prediction'.")
