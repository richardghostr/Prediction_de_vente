"""
Prediction de Vente -- Tableau de bord interactif ultra-complet.

Streamlit dashboard unifie qui fournit :
  1. Accueil avec KPIs et vue d'ensemble
  2. Upload & Auto-detection de colonnes
  3. Analyse exploratoire (statistiques, distributions, correlations)
  4. Visualisation de series temporelles (tendances, saisonnalite, decomposition)
  5. Prediction (online via API ou offline via modele local)
  6. Modele & Metriques (importance des features, evaluation)
  7. Donnees & Export

Lancement :
    streamlit run src/ui/streamlit_app.py
"""
import os
import sys
from pathlib import Path
from datetime import timedelta
import csv
import io

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

try:
    from src import config
except Exception:
    config = None


# =====================================================================
# CONFIGURATION & THEME
# =====================================================================
PALETTE = {
    "primary": "#0F4C75",
    "primary_light": "#3282B8",
    "secondary": "#E77728",
    "success": "#2ECC71",
    "danger": "#E74C3C",
    "warning": "#F39C12",
    "bg": "#F7F9FC",
    "card": "#FFFFFF",
    "text": "#1B2631",
    "text_muted": "#5D6D7E",
    "border": "#D5DBDB",
}

st.set_page_config(
    page_title="Prediction de Vente | Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a polished, professional look
st.markdown("""
<style>
    /* Global */
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    
    /* Sidebar */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F4C75 0%, #1B2631 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown h1,
    div[data-testid="stSidebar"] .stMarkdown h2,
    div[data-testid="stSidebar"] .stMarkdown h3,
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown label,
    div[data-testid="stSidebar"] .stMarkdown span {
        color: #ECF0F1 !important;
    }
    div[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15); }
    
    /* Headers */
    h1 { color: #0F4C75; font-weight: 700; }
    h2 { color: #1B2631; font-weight: 600; border-bottom: 2px solid #0F4C75; padding-bottom: 0.3rem; }
    h3 { color: #2C3E50; font-weight: 600; }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        padding: 1rem 1.2rem;
        border-radius: 0.75rem;
        border-left: 4px solid #0F4C75;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label { color: #5D6D7E !important; font-size: 0.85rem; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #1B2631 !important; font-weight: 700; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #F7F9FC;
        padding: 0.3rem;
        border-radius: 0.75rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem;
        font-weight: 500;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background: #0F4C75 !important;
        color: white !important;
    }
    
    /* DataFrames */
    .stDataFrame { border-radius: 0.5rem; overflow: hidden; }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background: #0F4C75;
        border: none;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background: #3282B8;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: #0F4C75 !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
    }
    
    /* Dividers */
    hr { border-color: #D5DBDB; }
    
    /* Info/Warning boxes */
    .stAlert { border-radius: 0.5rem; }
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
        "Nombre": int(series.count()),
        "Moyenne": round(float(series.mean()), 2),
        "Ecart-type": round(float(series.std()), 2),
        "Min": round(float(series.min()), 2),
        "Q1 (25%)": round(float(series.quantile(0.25)), 2),
        "Mediane": round(float(series.median()), 2),
        "Q3 (75%)": round(float(series.quantile(0.75)), 2),
        "Max": round(float(series.max()), 2),
        "Asymetrie": round(float(series.skew()), 3),
        "Kurtosis": round(float(series.kurtosis()), 3),
    }


def prepare_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    group_col: str = None,
    group_filter: str = "Tous",
    aggregate: bool = True,
) -> pd.DataFrame:
    """Clean, filter, and aggregate data into a proper time series."""
    ts = df.copy()

    if group_col and group_col != "Aucun" and group_filter != "Tous":
        ts = ts[ts[group_col].astype(str) == group_filter]

    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col])

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


def format_number(n, decimals=0):
    """Format a number with thousands separator."""
    if decimals == 0:
        return f"{n:,.0f}".replace(",", " ")
    return f"{n:,.{decimals}f}".replace(",", " ")


# =====================================================================
# SIDEBAR
# =====================================================================

with st.sidebar:
    st.markdown("## Prediction de Vente")
    st.caption("Tableau de bord interactif et complet")
    st.divider()

    uploaded = st.file_uploader(
        "Charger un fichier CSV",
        type=["csv"],
        help="Chargez vos donnees de vente au format CSV. Le fichier doit contenir au moins une colonne de date et une colonne numerique.",
    )

    st.divider()
    st.markdown("### Parametres de prediction")
    horizon = st.slider(
        "Horizon (jours)", 1, 365, 30,
        help="Nombre de jours a predire dans le futur",
    )
    offline = st.checkbox(
        "Mode hors-ligne",
        value=True,
        help="Utiliser le modele local au lieu de l'API",
    )

    st.divider()
    st.markdown("### Options avancees")
    aggregate = st.checkbox(
        "Agreger les dates dupliquees (somme)",
        value=True,
        key="agg",
        help="Si plusieurs lignes ont la meme date, elles seront sommees",
    )
    archive_outputs = st.checkbox(
        "Archiver les sorties",
        value=False,
        key="archive",
        help="Sauvegarder les resultats dans les dossiers du projet",
    )

    st.divider()
    st.markdown("### Statut du modele")
    if get_model_path:
        try:
            mp = get_model_path()
            if mp:
                st.success(f"Modele : {Path(mp).name}")
            else:
                st.warning("Aucun artefact trouve")
        except Exception:
            st.warning("Impossible de charger le modele")
    else:
        st.info("Module de prediction non disponible")

    st.divider()
    st.markdown(
        "<div style='text-align:center; color: rgba(255,255,255,0.4); font-size:0.75rem;'>"
        "Prediction de Vente v2.0<br>Dashboard Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# LANDING PAGE (no file uploaded)
# =====================================================================

if uploaded is None:
    st.title("Bienvenue sur le Dashboard de Prediction de Vente")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0F4C75 0%, #3282B8 100%); 
                padding: 2rem; border-radius: 1rem; color: white; margin-bottom: 2rem;">
        <h3 style="color: white; margin-top: 0; border: none;">Votre outil complet d'analyse et de prevision</h3>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 0;">
            Chargez un fichier CSV dans la barre laterale pour commencer l'analyse de vos donnees de vente.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; 
                    border-left: 4px solid #0F4C75; box-shadow: 0 2px 8px rgba(0,0,0,0.06); height: 160px;">
            <h4 style="color: #0F4C75; margin-top: 0;">Visualisation</h4>
            <p style="color: #5D6D7E; font-size: 0.9rem;">
                Explorez vos donnees avec des graphiques interactifs et des tableaux de bord dynamiques.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; 
                    border-left: 4px solid #3282B8; box-shadow: 0 2px 8px rgba(0,0,0,0.06); height: 160px;">
            <h4 style="color: #3282B8; margin-top: 0;">Analyse</h4>
            <p style="color: #5D6D7E; font-size: 0.9rem;">
                Statistiques descriptives, distributions, correlations et detection d'anomalies.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; 
                    border-left: 4px solid #E77728; box-shadow: 0 2px 8px rgba(0,0,0,0.06); height: 160px;">
            <h4 style="color: #E77728; margin-top: 0;">Prediction</h4>
            <p style="color: #5D6D7E; font-size: 0.9rem;">
                Previsions basees sur XGBoost avec intervalle de confiance et metriques detaillees.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; 
                    border-left: 4px solid #2ECC71; box-shadow: 0 2px 8px rgba(0,0,0,0.06); height: 160px;">
            <h4 style="color: #2ECC71; margin-top: 0;">Export</h4>
            <p style="color: #5D6D7E; font-size: 0.9rem;">
                Exportez les resultats, previsions et rapports en format CSV pour vos equipes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Show sample info
    sample_path = PROJECT_ROOT / "sample.csv"
    if sample_path.exists():
        st.info(
            f"Un fichier d'exemple est disponible dans le projet : `{sample_path.name}` "
            f"(~1000 lignes de ventes simulees avec date, magasin, produit, categorie, quantite, prix, revenu)."
        )

    # Quick feature overview
    st.markdown("### Fonctionnalites du Dashboard")
    features_data = {
        "Fonctionnalite": [
            "Vue d'ensemble", "Analyse exploratoire", "Series temporelles",
            "Prediction", "Modele & Metriques", "Donnees & Export",
        ],
        "Description": [
            "KPIs cles, apercu des donnees, types de colonnes, valeurs manquantes",
            "Statistiques descriptives, histogrammes, box plots, matrice de correlation, repartition par categorie",
            "Evolution temporelle, moyenne mobile, agregation jour/semaine/mois, saisonnalite, heatmap",
            "Prevision XGBoost ou naive, mode hors-ligne/API, graphique interactif, resume, export",
            "Informations du modele, importance des features, metriques sauvegardees",
            "Navigation entre donnees brutes/nettoyees/features/previsions, telechargement CSV",
        ],
        "Statut": [
            "Disponible", "Disponible", "Disponible",
            "Disponible", "Disponible", "Disponible",
        ],
    }
    st.dataframe(pd.DataFrame(features_data), use_container_width=True, hide_index=True)

    st.stop()


# ---------------------------------------------------------------------------
# Load data (robust to different delimiters)
# ---------------------------------------------------------------------------
try:
    # Read whole upload (stream may be bytes or text)
    try:
        uploaded.seek(0)
    except Exception:
        pass
    raw = uploaded.read()

    # Prepare a sample text for delimiter sniffing
    if isinstance(raw, bytes):
        sample_text = raw[:8192].decode("utf-8", errors="ignore")
    else:
        sample_text = str(raw)[:8192]

    # Detect delimiter
    try:
        dialect = csv.Sniffer().sniff(sample_text)
        sep = dialect.delimiter
    except Exception:
        sep = ';' if ';' in sample_text and sample_text.count(';') > sample_text.count(',') else ','

    # Try common encodings if pandas raises decoding error
    encodings_to_try = [None, "utf-8", "cp1252", "latin-1"]
    df_raw = None
    for enc in encodings_to_try:
        try:
            if isinstance(raw, bytes):
                # Use BytesIO and pass encoding when not None
                bio = io.BytesIO(raw)
                if enc:
                    df_raw = pd.read_csv(bio, sep=sep, encoding=enc)
                else:
                    df_raw = pd.read_csv(bio, sep=sep)
            else:
                # raw is str
                s = io.StringIO(raw)
                df_raw = pd.read_csv(s, sep=sep)
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            # If parsing fails for other reasons, keep trying encodings
            continue

    if df_raw is None:
        raise ValueError("Impossible de decoder/parse le fichier CSV avec les encodages tries")
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
    st.markdown("### Colonnes detectees")
    date_col = st.selectbox(
        "Colonne de date",
        options=df_raw.columns.tolist(),
        index=df_raw.columns.tolist().index(detected_date) if detected_date in df_raw.columns else 0,
        key="date_col_sel",
    )

    numeric_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()
    value_default = detected_value or (numeric_cols[0] if numeric_cols else df_raw.columns[0])
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
            options=["Aucun"] + cat_cols,
            index=0,
            key="group_col_sel",
        )
    else:
        group_col = "Aucun"

    group_filter = "Tous"
    if group_col != "Aucun":
        unique_vals = sorted(df_raw[group_col].dropna().astype(str).unique().tolist())
        group_filter = st.selectbox(
            "Filtrer par groupe",
            options=["Tous"] + unique_vals,
            index=0,
            key="group_filter_sel",
        )


# ---------------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------------
df_ts = prepare_time_series(
    df_raw, date_col, value_col,
    group_col=group_col if group_col != "Aucun" else None,
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
# HEADER BAR
# =====================================================================
st.title("Dashboard de Prediction de Vente")

# Quick KPIs in header
hcol1, hcol2, hcol3, hcol4, hcol5 = st.columns(5)
with hcol1:
    st.metric("Lignes", format_number(len(df_raw)))
with hcol2:
    st.metric("Colonnes", f"{len(df_raw.columns)}")
with hcol3:
    if value_col in df_clean.columns:
        total = df_clean[value_col].sum()
        st.metric("Total", format_number(total))
with hcol4:
    if date_col in df_clean.columns:
        try:
            date_range = (df_clean[date_col].max() - df_clean[date_col].min()).days
            st.metric("Plage (jours)", format_number(date_range))
        except Exception:
            st.metric("Plage", "N/A")
with hcol5:
    missing_pct = df_raw.isna().mean().mean() * 100
    st.metric("Manquant (%)", f"{missing_pct:.1f}%")


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

    # Data preview
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Apercu des donnees brutes")
        st.dataframe(df_raw.head(25), use_container_width=True, height=400)

    with c2:
        st.subheader("Types et qualite des colonnes")
        type_info = pd.DataFrame({
            "Colonne": df_raw.columns,
            "Type": df_raw.dtypes.astype(str).values,
            "Non-null": df_raw.notna().sum().values,
            "Null": df_raw.isna().sum().values,
            "% manquant": (df_raw.isna().mean() * 100).round(1).values,
            "Unique": [df_raw[c].nunique() for c in df_raw.columns],
        })
        st.dataframe(type_info, use_container_width=True, height=400, hide_index=True)

    st.divider()

    # Numeric summary
    num_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        st.subheader("Resume statistique rapide")
        st.dataframe(
            df_raw[num_cols].describe().round(2).T.reset_index().rename(columns={"index": "Colonne"}),
            use_container_width=True,
            hide_index=True,
        )

    # Missing values visualization
    if df_raw.isna().any().any():
        st.divider()
        st.subheader("Carte des valeurs manquantes")
        missing_pct = df_raw.isna().mean() * 100
        missing_df = missing_pct[missing_pct > 0].sort_values(ascending=True)
        if not missing_df.empty:
            fig_missing = px.bar(
                x=missing_df.values, y=missing_df.index,
                orientation="h",
                labels={"x": "% manquant", "y": "Colonne"},
                color=missing_df.values,
                color_continuous_scale=[[0, PALETTE["success"]], [0.5, PALETTE["warning"]], [1, PALETTE["danger"]]],
            )
            fig_missing.update_layout(
                showlegend=False, height=max(200, len(missing_df) * 40),
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=10),
            )
            st.plotly_chart(fig_missing, use_container_width=True)

    # Category breakdown (if applicable)
    if group_col != "Aucun" and group_col in df_raw.columns:
        st.divider()
        st.subheader(f"Repartition par '{group_col}'")
        c1, c2 = st.columns(2)
        with c1:
            cat_counts = df_raw[group_col].value_counts()
            fig_pie = px.pie(
                values=cat_counts.values,
                names=cat_counts.index,
                color_discrete_sequence=[PALETTE["primary"], PALETTE["primary_light"], PALETTE["secondary"], PALETTE["success"], PALETTE["warning"]],
                hole=0.4,
            )
            fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            if value_col in df_raw.columns:
                grp_agg = df_raw.groupby(group_col)[value_col].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
                grp_agg = grp_agg.reset_index()
                grp_agg.columns = [group_col, "Total", "Moyenne", "Nombre"]
                st.dataframe(grp_agg.round(2), use_container_width=True, hide_index=True, height=350)


# =====================================================================
# TAB 2 : ANALYSE EXPLORATOIRE
# =====================================================================
with tab_eda:
    st.header("Analyse exploratoire des donnees")

    if value_col in df_clean.columns:
        val_series = df_clean[value_col].dropna()

        # Descriptive Statistics
        st.subheader("Statistiques descriptives")
        stats = compute_stats(val_series)
        stat_cols = st.columns(5)
        for i, (k, v) in enumerate(stats.items()):
            with stat_cols[i % 5]:
                st.metric(k, v)

        st.divider()

        # Distribution Analysis
        st.subheader("Analyse de distribution")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Distribution de '{value_col}'**")
            nbins = st.slider("Nombre de bins", 10, 100, 40, key="eda_nbins")
            fig_hist = px.histogram(
                df_clean, x=value_col, nbins=nbins,
                color_discrete_sequence=[PALETTE["primary"]],
                marginal="box",
            )
            fig_hist.update_layout(
                height=420, showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title=value_col,
                yaxis_title="Frequence",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            st.markdown("**Box Plot**")
            if group_col != "Aucun" and group_col in df_ts.columns:
                fig_box = px.box(
                    df_ts, x=group_col, y=value_col,
                    color=group_col,
                    color_discrete_sequence=[PALETTE["primary"], PALETTE["primary_light"], PALETTE["secondary"], PALETTE["success"], PALETTE["warning"]],
                )
            else:
                fig_box = px.box(
                    df_clean, y=value_col,
                    color_discrete_sequence=[PALETTE["primary"]],
                )
            fig_box.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Violin plot for multiple numeric columns
        if len(num_cols) > 1:
            st.divider()
            st.subheader("Comparaison des distributions")
            selected_num_cols = st.multiselect(
                "Colonnes a comparer",
                options=num_cols,
                default=num_cols[:3],
                key="eda_multi_cols",
            )
            if selected_num_cols:
                melted = df_clean[selected_num_cols].melt(var_name="Variable", value_name="Valeur")
                fig_violin = px.violin(
                    melted, x="Variable", y="Valeur",
                    color="Variable",
                    box=True, points="outliers",
                    color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"], PALETTE["success"], PALETTE["warning"], PALETTE["danger"]],
                )
                fig_violin.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_violin, use_container_width=True)

        # Correlation matrix
        num_df = df_clean.select_dtypes(include=["number"])
        if len(num_df.columns) > 1:
            st.divider()
            st.subheader("Matrice de correlation")
            corr_method = st.radio(
                "Methode de correlation",
                options=["pearson", "spearman", "kendall"],
                horizontal=True,
                key="eda_corr_method",
            )
            corr = num_df.corr(method=corr_method)
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                aspect="auto",
            )
            fig_corr.update_layout(
                height=max(400, len(corr.columns) * 50),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # Scatter plot explorer
        if len(num_cols) >= 2:
            st.divider()
            st.subheader("Explorateur de nuage de points")
            sc1, sc2 = st.columns(2)
            with sc1:
                scatter_x = st.selectbox("Axe X", options=num_cols, index=0, key="scatter_x")
            with sc2:
                scatter_y = st.selectbox("Axe Y", options=num_cols, index=min(1, len(num_cols) - 1), key="scatter_y")

            scatter_color = None
            if group_col != "Aucun" and group_col in df_ts.columns:
                scatter_color = group_col

            # Use trendline if statsmodels is available (plotly will import it when trendline='ols')
            try:
                import statsmodels.api  # type: ignore
                trendline_arg = "ols"
            except Exception:
                trendline_arg = None
                st.info("`statsmodels` non installé — la ligne de tendance est désactivée.")

            fig_scatter = px.scatter(
                df_ts, x=scatter_x, y=scatter_y,
                color=scatter_color,
                color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"], PALETTE["success"], PALETTE["warning"], PALETTE["danger"]],
                opacity=0.7,
                trendline=trendline_arg,
            )
            fig_scatter.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Category breakdown
        if group_col != "Aucun" and group_col in df_raw.columns:
            st.divider()
            st.subheader(f"Analyse par '{group_col}'")
            c1, c2 = st.columns(2)
            with c1:
                cat_counts = df_raw[group_col].value_counts()
                fig_cat_bar = px.bar(
                    x=cat_counts.index, y=cat_counts.values,
                    labels={"x": group_col, "y": "Nombre d'enregistrements"},
                    color=cat_counts.index,
                    color_discrete_sequence=[PALETTE["primary"], PALETTE["primary_light"], PALETTE["secondary"], PALETTE["success"], PALETTE["warning"]],
                )
                fig_cat_bar.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_cat_bar, use_container_width=True)

            with c2:
                if value_col in df_raw.columns:
                    grp_sum = df_raw.groupby(group_col)[value_col].sum().sort_values(ascending=False)
                    fig_grp_bar = px.bar(
                        x=grp_sum.index, y=grp_sum.values,
                        labels={"x": group_col, "y": f"Total {value_col}"},
                        color=grp_sum.index,
                        color_discrete_sequence=[PALETTE["primary"], PALETTE["primary_light"], PALETTE["secondary"], PALETTE["success"], PALETTE["warning"]],
                    )
                    fig_grp_bar.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_grp_bar, use_container_width=True)
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
        show_rolling = st.checkbox("Afficher la moyenne mobile", value=True, key="ts_show_rolling")
        rolling_window = st.slider("Fenetre de moyenne mobile (jours)", 3, 30, 7, key="ts_rolling_window") if show_rolling else 7

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts_data[date_col], y=ts_data[value_col],
            mode="lines+markers",
            name="Valeur observee",
            line=dict(color=PALETTE["primary"], width=2),
            marker=dict(size=3),
            hovertemplate="Date: %{x}<br>Valeur: %{y:,.2f}<extra></extra>",
        ))

        if show_rolling and len(ts_data) > rolling_window:
            ts_data["rolling"] = ts_data[value_col].rolling(rolling_window, min_periods=1).mean()
            fig_ts.add_trace(go.Scatter(
                x=ts_data[date_col], y=ts_data["rolling"],
                mode="lines",
                name=f"Moyenne mobile ({rolling_window}j)",
                line=dict(color=PALETTE["secondary"], width=3, dash="dash"),
                hovertemplate="Date: %{x}<br>Moy. mobile: %{y:,.2f}<extra></extra>",
            ))

        fig_ts.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title=value_col,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Aggregated views
        st.divider()
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
            fig_agg_bar.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_agg_bar, use_container_width=True)

        with c2:
            fig_agg_line = px.area(
                ts_agg, x="Date", y="Moyenne",
                color_discrete_sequence=[PALETTE["secondary"]],
                labels={"Moyenne": f"Moyenne {value_col}"},
            )
            fig_agg_line.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_agg_line, use_container_width=True)

        # Transactions count
        st.markdown(f"**Nombre de transactions par periode ({agg_period.lower()})**")
        fig_count = px.bar(
            ts_agg, x="Date", y="Nb transactions",
            color_discrete_sequence=[PALETTE["success"]],
        )
        fig_count.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_count, use_container_width=True)

        # Seasonality
        st.divider()
        st.subheader("Saisonnalite")
        c1, c2 = st.columns(2)
        with c1:
            ts_data["dow"] = ts_data[date_col].dt.day_name()
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_labels = {"Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
                         "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"}
            dow_agg = ts_data.groupby("dow")[value_col].mean().reindex(dow_order)
            fig_dow = px.bar(
                x=[dow_labels.get(d, d) for d in dow_agg.index], y=dow_agg.values,
                labels={"x": "Jour de la semaine", "y": f"Moyenne {value_col}"},
                color_discrete_sequence=[PALETTE["primary"]],
            )
            fig_dow.update_layout(height=350, title="Moyenne par jour de la semaine", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_dow, use_container_width=True)

        with c2:
            ts_data["month_name"] = ts_data[date_col].dt.month_name()
            month_order = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            month_labels = {"January": "Janv", "February": "Fev", "March": "Mars", "April": "Avr",
                           "May": "Mai", "June": "Juin", "July": "Juil", "August": "Aout",
                           "September": "Sept", "October": "Oct", "November": "Nov", "December": "Dec"}
            present_months = [m for m in month_order if m in ts_data["month_name"].values]
            month_agg = ts_data.groupby("month_name")[value_col].mean().reindex(present_months)
            fig_month = px.bar(
                x=[month_labels.get(m, m) for m in month_agg.index], y=month_agg.values,
                labels={"x": "Mois", "y": f"Moyenne {value_col}"},
                color_discrete_sequence=[PALETTE["secondary"]],
            )
            fig_month.update_layout(height=350, title="Moyenne par mois", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_month, use_container_width=True)

        # Heatmap: day-of-week vs week number
        if len(ts_data) > 14:
            st.divider()
            st.subheader("Heatmap d'activite (semaine x jour)")
            ts_data["week"] = ts_data[date_col].dt.isocalendar().week.astype(int)
            ts_data["dow_num"] = ts_data[date_col].dt.dayofweek
            heatmap_data = ts_data.pivot_table(
                index="dow_num", columns="week", values=value_col, aggfunc="sum"
            ).fillna(0)
            fig_hm = px.imshow(
                heatmap_data,
                labels=dict(x="Semaine", y="Jour (0=Lun, 6=Dim)", color=value_col),
                color_continuous_scale="Blues",
                aspect="auto",
                y=["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][:len(heatmap_data)],
            )
            fig_hm.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_hm, use_container_width=True)

        # Cumulative chart
        st.divider()
        st.subheader("Evolution cumulative")
        ts_cum = ts_data[[date_col, value_col]].copy()
        ts_cum["cumul"] = ts_cum[value_col].cumsum()
        fig_cum = px.area(
            ts_cum, x=date_col, y="cumul",
            labels={"cumul": f"Cumul {value_col}"},
            color_discrete_sequence=[PALETTE["primary_light"]],
        )
        fig_cum.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cum, use_container_width=True)

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

    # Info banner
    mode_label = "Hors-ligne (modele local)" if offline else "En ligne (API)"
    st.markdown(
        f"""<div style="background: #EBF5FB; border-left: 4px solid {PALETTE['primary']}; 
                padding: 0.8rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin-bottom: 1rem;">
            <strong>Mode :</strong> {mode_label} &nbsp; | &nbsp;
            <strong>Horizon :</strong> {horizon} jours &nbsp; | &nbsp;
            <strong>Points de donnees :</strong> {len(df_clean)}
        </div>""",
        unsafe_allow_html=True,
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
            for c in df_clean.select_dtypes(include=["number"]).columns:
                if c != value_col and pd.notna(row[c]):
                    item[c] = float(row[c])
            series_payload.append(item)

        forecast_result = None

        if offline:
            with st.spinner("Prediction en cours (mode hors-ligne)..."):
                if predict_series is not None:
                    try:
                        forecast_result = predict_series(series_payload, horizon=horizon)
                    except Exception as e:
                        st.warning(f"Modele local echoue : {e}. Utilisation du repli naive.")

                # Naive fallback
                if forecast_result is None:
                    last_val = None
                    last_date = None
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

        # Store in session state
        st.session_state["forecast_result"] = forecast_result
        st.session_state["fc"] = fc
        st.session_state["hist"] = hist

        # Results
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
            hovertemplate="Date: %{x}<br>Historique: %{y:,.2f}<extra></extra>",
        ))

        # Forecast
        fig_fc.add_trace(go.Scatter(
            x=fc["ds"], y=fc["yhat"],
            mode="lines+markers",
            name="Prediction",
            line=dict(color=PALETTE["secondary"], width=2.5, dash="dash"),
            marker=dict(size=5),
            hovertemplate="Date: %{x}<br>Prediction: %{y:,.2f}<extra></extra>",
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
                fill="tonexty", fillcolor="rgba(231, 119, 40, 0.15)",
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
                line=dict(color="#95A5A6", width=1.5, dash="dot"),
                showlegend=False,
            ))

        fig_fc.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Valeur",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast KPIs
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
                delta_color = "normal" if pct_change >= 0 else "inverse"
                st.metric("Variation (%)", f"{pct_change:+.1f}%")

        # Weekly breakdown
        if len(fc) > 7:
            st.divider()
            st.subheader("Prevision par semaine")
            fc_weekly = fc.set_index("ds").resample("W")["yhat"].agg(["sum", "mean", "count"]).reset_index()
            fc_weekly.columns = ["Semaine", "Total", "Moyenne", "Nb jours"]

            c1, c2 = st.columns(2)
            with c1:
                fig_fc_bar = px.bar(
                    fc_weekly, x="Semaine", y="Total",
                    color_discrete_sequence=[PALETTE["secondary"]],
                )
                fig_fc_bar.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_fc_bar, use_container_width=True)

            with c2:
                st.dataframe(fc_weekly.round(2), use_container_width=True, hide_index=True, height=350)

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

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Type** : `{type(model).__name__}`")
                st.markdown(f"**Chemin** : `{mp}`")
            with c2:
                if hasattr(model, "n_estimators"):
                    st.markdown(f"**N estimators** : `{model.n_estimators}`")
                if hasattr(model, "max_depth"):
                    st.markdown(f"**Max depth** : `{model.max_depth}`")
                if hasattr(model, "learning_rate"):
                    st.markdown(f"**Learning rate** : `{model.learning_rate}`")

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
                st.divider()
                st.subheader("Importance des features")
                imp_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances,
                }).sort_values("Importance", ascending=True)

                fig_imp = px.bar(
                    imp_df, x="Importance", y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale=[[0, "#EBF5FB"], [1, PALETTE["primary"]]],
                )
                fig_imp.update_layout(
                    height=max(300, len(feature_names) * 30),
                    showlegend=False,
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        except FileNotFoundError:
            st.warning("Aucun artefact de modele trouve dans `models/artifacts/`.")
        except Exception as e:
            st.error(f"Erreur lors du chargement du modele : {e}")
    else:
        st.info("Module `predict` non disponible dans l'environnement actuel.")

    # Features preview
    if features_df is not None:
        st.divider()
        st.subheader("Apercu des features calculees")
        st.markdown(f"**Dimensions** : {features_df.shape[0]} lignes x {features_df.shape[1]} colonnes")
        st.dataframe(features_df.head(20), use_container_width=True, hide_index=True)

        # Feature statistics
        feat_num = features_df.select_dtypes(include=["number"])
        if not feat_num.empty:
            st.markdown("**Statistiques des features numeriques**")
            st.dataframe(
                feat_num.describe().round(3).T.reset_index().rename(columns={"index": "Feature"}),
                use_container_width=True,
                hide_index=True,
            )

    # Saved metrics files
    st.divider()
    st.subheader("Metriques sauvegardees")
    metrics_dir = Path(config.MODELS_METRICS_DIR) if config and hasattr(config, "MODELS_METRICS_DIR") else Path("models/metrics")
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
                    
                    # Display metrics in a nice layout if possible
                    if isinstance(data, dict):
                        # Extract common metric keys
                        metric_keys = [k for k in data if isinstance(data[k], (int, float))]
                        if metric_keys:
                            metric_cols = st.columns(min(len(metric_keys), 4))
                            for i, k in enumerate(metric_keys):
                                with metric_cols[i % len(metric_cols)]:
                                    st.metric(k, f"{data[k]:.4f}" if isinstance(data[k], float) else str(data[k]))
                        
                        st.markdown("**Donnees completes :**")
                    st.json(data)
                except Exception as e:
                    st.error(f"Erreur de lecture : {e}")
        else:
            st.info("Aucun fichier de metriques trouve.")
    else:
        st.info("Le dossier de metriques n'existe pas encore. Entrainez un modele pour generer des metriques.")


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
        st.markdown(f"**Dimensions** : {df_raw.shape[0]} lignes x {df_raw.shape[1]} colonnes")

        # Search / filter
        search_term = st.text_input("Rechercher dans les donnees", key="search_raw", placeholder="Tapez pour filtrer...")
        display_df = df_raw
        if search_term:
            mask = df_raw.astype(str).apply(lambda col: col.str.contains(search_term, case=False, na=False)).any(axis=1)
            display_df = df_raw[mask]
            st.caption(f"{len(display_df)} lignes trouvees sur {len(df_raw)}")

        st.dataframe(display_df, use_container_width=True, height=500, hide_index=True)
        csv = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button("Telecharger les donnees brutes (CSV)", csv, "donnees_brutes.csv", "text/csv")

    elif data_view == "Nettoyees":
        st.subheader("Donnees nettoyees")
        st.markdown(f"**Dimensions** : {df_clean.shape[0]} lignes x {df_clean.shape[1]} colonnes")
        st.dataframe(df_clean, use_container_width=True, height=500, hide_index=True)
        csv = df_clean.to_csv(index=False).encode("utf-8")
        st.download_button("Telecharger les donnees nettoyees (CSV)", csv, "donnees_nettoyees.csv", "text/csv")

    elif data_view == "Avec features":
        if features_df is not None:
            st.subheader("Donnees avec features")
            st.markdown(f"**Dimensions** : {features_df.shape[0]} lignes x {features_df.shape[1]} colonnes")
            st.dataframe(features_df, use_container_width=True, height=500, hide_index=True)
            csv = features_df.to_csv(index=False).encode("utf-8")
            st.download_button("Telecharger les donnees features (CSV)", csv, "donnees_features.csv", "text/csv")
        else:
            st.info("Les features n'ont pas pu etre calculees. Verifiez que le module `features.py` est disponible.")

    elif data_view == "Previsions":
        if "fc" in st.session_state and "hist" in st.session_state:
            fc = st.session_state["fc"]
            hist = st.session_state["hist"]
            combined = pd.merge(hist, fc, on="ds", how="outer").sort_values("ds")

            st.subheader("Donnees combinees (historique + prevision)")
            st.markdown(f"**Dimensions** : {combined.shape[0]} lignes x {combined.shape[1]} colonnes")

            # Add source label
            combined_display = combined.copy()
            combined_display["source"] = combined_display.apply(
                lambda r: "Historique" if pd.notna(r.get("y")) and pd.isna(r.get("yhat")) else (
                    "Prevision" if pd.isna(r.get("y")) and pd.notna(r.get("yhat")) else "Overlap"
                ), axis=1
            )

            st.dataframe(combined_display, use_container_width=True, height=500, hide_index=True)
            csv = combined.to_csv(index=False).encode("utf-8")
            st.download_button("Telecharger les donnees combinees (CSV)", csv, "combined_forecast.csv", "text/csv")
        else:
            st.info("Lancez d'abord une prediction dans l'onglet 'Prediction' pour voir les resultats ici.")
