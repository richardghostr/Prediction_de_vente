"""
Dashboard orchestrateur -- Pipeline complet de prediction de vente.

Ce fichier sert d'orchestrateur pour le pipeline complet :
  Ingestion -> Nettoyage -> Feature Engineering -> Prediction -> Visualisation

Chaque phase est un module UI independant dans src/ui/ et est appele
depuis cette interface unifiee avec une navigation par sidebar.

Lancement :
    streamlit run src/ui/dashboard.py
"""
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui import ingest_ui, clean_ui, predict_ui, visualize_ui, features_ui


# =====================================================================
# INITIALIZE SESSION STATE FOR PERSISTENCE
# =====================================================================
if "dashboard_state" not in st.session_state:
    st.session_state["dashboard_state"] = {
        "last_page": "home",
        "uploaded_files": {},  # Store uploaded files by module
        "module_params": {},  # Store parameters for each module
        "last_results": {},  # Store last results from each module
    }

# Restore last visited page if it exists
if "restore_page" not in st.session_state:
    st.session_state["restore_page"] = True

# On-disk cache dir for dashboard (helps survive hard reloads)
CACHE_DIR = PROJECT_ROOT / "data" / "interim" / ".dashboard_cache"
try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    CACHE_DIR = None

# Optionally restore previous dashboard_state from disk (pickle)
if CACHE_DIR is not None:
    try:
        ds_path = CACHE_DIR / "dashboard_state.pkl"
        if ds_path.exists():
            import pickle
            with open(ds_path, "rb") as _f:
                loaded = pickle.load(_f)
            # Only replace if loaded is a dict with expected keys
            if isinstance(loaded, dict) and "last_page" in loaded:
                st.session_state["dashboard_state"] = loaded
    except Exception:
        pass


# =====================================================================
# CONFIGURATION
# =====================================================================
PALETTE = {
    "primary": "#0F4C75",
    "primary_light": "#3282B8",
    "secondary": "#E77728",
    "success": "#2ECC71",
    "text": "#1B2631",
}

st.set_page_config(
    page_title="Pipeline Dashboard | Prediction de Vente",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS - coherent with streamlit_app.py
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    
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
    
    h1 { color: #0F4C75; font-weight: 700; }
    h2 { color: #1B2631; font-weight: 600; border-bottom: 2px solid #0F4C75; padding-bottom: 0.3rem; }
    h3 { color: #2C3E50; font-weight: 600; }
    
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        padding: 1rem 1.2rem;
        border-radius: 0.75rem;
        border-left: 4px solid #0F4C75;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label { color: #5D6D7E !important; font-size: 0.85rem; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #1B2631 !important; font-weight: 700; }
    
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
    
    .stButton > button[kind="primary"] {
        background: #0F4C75;
        border: none;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background: #3282B8;
    }
    
    .stDownloadButton > button {
        background: #0F4C75 !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
    }
    
    hr { border-color: #D5DBDB; }
    .stAlert { border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# NAVIGATION PAGES
# =====================================================================
PAGES = {
    "Accueil": "home",
    "1. Ingestion des donnees": "ingest",
    "2. Nettoyage des donnees": "clean",
    "3. Feature Engineering": "features",
    "4. Prediction": "predict",
    "5. Visualisation des resultats": "visualize",
}

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("## Pipeline de Prediction")
    st.caption("Navigation par phase du pipeline")
    st.divider()

    # Get last visited page index
    last_page_key = st.session_state["dashboard_state"]["last_page"]
    page_keys = list(PAGES.keys())
    try:
        default_index = page_keys.index([k for k, v in PAGES.items() if v == last_page_key][0])
    except (IndexError, ValueError):
        default_index = 0

    page = st.radio(
        "Phase du pipeline",
        options=page_keys,
        index=default_index if st.session_state.get("restore_page", True) else 0,
        key="pipeline_page",
    )
    
    # Save current page to session state
    st.session_state["dashboard_state"]["last_page"] = PAGES[page]
    st.session_state["restore_page"] = True

    st.divider()

    # Pipeline status indicator
    st.markdown("### Statut du pipeline")
    
    # Check data directories
    raw_dir = Path("data/raw")
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")
    models_dir = Path("models/artifacts")

    raw_files = list(raw_dir.glob("*_ingest.csv")) if raw_dir.exists() else []
    clean_files = list(interim_dir.glob("*_clean.csv")) if interim_dir.exists() else []
    feature_files = list(processed_dir.glob("*_features.csv")) if processed_dir.exists() else []
    model_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []

    steps = [
        ("Ingestion", len(raw_files) > 0),
        ("Nettoyage", len(clean_files) > 0),
        ("Features", len(feature_files) > 0),
        ("Modele", len(model_files) > 0),
    ]

    for step_name, done in steps:
        if done:
            st.markdown(f"<span style='color: #2ECC71;'>&#10003;</span> {step_name}", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color: #95A5A6;'>&#9675;</span> {step_name}", unsafe_allow_html=True)

    st.divider()
    
    # Show persistence status
    st.markdown("### 💾 État de la session")
    
    # Always show session info
    n_files = len(st.session_state["dashboard_state"]["uploaded_files"])
    n_results = len(st.session_state["dashboard_state"]["last_results"])
    n_params = len(st.session_state["dashboard_state"]["module_params"])
    
    if n_files > 0 or n_results > 0 or n_params > 0:
        if n_files > 0:
            st.success(f"📂 {n_files} fichier(s) en mémoire")
        if n_results > 0:
            st.info(f"📊 {n_results} résultat(s) sauvegardé(s)")
        if n_params > 0:
            st.info(f"⚙️ {n_params} module(s) configuré(s)")
        
        if st.button("🗑️ Réinitialiser tout", key="reset_all_dashboard", use_container_width=True):
            st.session_state["dashboard_state"] = {
                "last_page": "home",
                "uploaded_files": {},
                "module_params": {},
                "last_results": {},
            }

            # Remove on-disk cache if present
            try:
                if CACHE_DIR is not None and CACHE_DIR.exists():
                    for f in CACHE_DIR.iterdir():
                        try:
                            f.unlink()
                        except Exception:
                            pass
            except Exception:
                pass

            st.session_state["restore_page"] = False
            st.success("✅ Session réinitialisée!")
            st.rerun()
    else:
        st.caption("Aucune donnée en cache")
    
    st.divider()
    st.markdown(
        "<div style='text-align:center; color: rgba(255,255,255,0.4); font-size:0.75rem;'>"
        "Prediction de Vente v2.0<br>Pipeline Dashboard"
        "</div>",
        unsafe_allow_html=True,
    )


# =====================================================================
# MAIN CONTENT
# =====================================================================
selected_page = PAGES[page]

# Persist dashboard state to disk so a hard reload retains uploaded files and module params
try:
    if CACHE_DIR is not None:
        ds_path = CACHE_DIR / "dashboard_state.pkl"
        import pickle
        with open(ds_path, "wb") as _f:
            pickle.dump(st.session_state.get("dashboard_state", {}), _f)
except Exception:
    pass

if selected_page == "home":
    st.title("Pipeline de Prediction de Vente")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #0F4C75 0%, #3282B8 100%); 
                padding: 2rem; border-radius: 1rem; color: white; margin-bottom: 2rem;">
        <h3 style="color: white; margin-top: 0; border: none;">Pipeline complet d'analyse et de prevision</h3>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 0;">
            Naviguez a travers chaque phase du pipeline pour transformer vos donnees brutes 
            en previsions exploitables. Selectionnez une phase dans la barre laterale pour commencer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline overview cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; 
                    border-left: 4px solid {PALETTE['primary']}; box-shadow: 0 2px 8px rgba(0,0,0,0.06); min-height: 130px;">
            <h4 style="color: {PALETTE['primary']}; margin-top: 0;">1. Ingestion</h4>
            <p style="color: #5D6D7E; font-size: 0.9rem;">
                Chargez un fichier CSV, detection automatique des colonnes, mapping intelligent et validation.
            </p>
            <p style="color: #5D6D7E; font-size: 0.8rem;"><strong>{len(raw_files)}</strong> fichier(s) ingere(s)</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; 
                    border-left: 4px solid {PALETTE['primary_light']}; box-shadow: 0 2px 8px rgba(0,0,0,0.06); min-height: 130px;">
            <h4 style="color: {PALETTE['primary_light']}; margin-top: 0;">2. Nettoyage & Features</h4>
            <p style="color: #5D6D7E; font-size: 0.9rem;">
                Harmonisation des dates, remplissage des manquants, suppression d'outliers, features temporelles.
            </p>
            <p style="color: #5D6D7E; font-size: 0.8rem;"><strong>{len(clean_files)}</strong> nettoye(s), <strong>{len(feature_files)}</strong> avec features</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; 
                    border-left: 4px solid {PALETTE['secondary']}; box-shadow: 0 2px 8px rgba(0,0,0,0.06); min-height: 130px;">
            <h4 style="color: {PALETTE['secondary']}; margin-top: 0;">3. Prediction & Visualisation</h4>
            <p style="color: #5D6D7E; font-size: 0.9rem;">
                Prevision XGBoost avec horizon configurable, graphiques interactifs, metriques et export.
            </p>
            <p style="color: #5D6D7E; font-size: 0.8rem;"><strong>{len(model_files)}</strong> modele(s) disponible(s)</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick start guide
    st.subheader("Guide de demarrage rapide")
    st.markdown("""
    | Etape | Action | Description |
    |-------|--------|-------------|
    | 1 | **Ingestion** | Chargez votre CSV dans l'onglet "Ingestion des donnees" |
    | 2 | **Nettoyage** | Selectionnez le fichier ingere et lancez le nettoyage |
    | 3 | **Features** | Generez les features temporelles (lags, moyennes mobiles) |
    | 4 | **Prediction** | Choisissez un fichier et lancez la prediction |
    | 5 | **Visualisation** | Explorez les resultats avec les graphiques interactifs |
    """)

    st.info(
        "Astuce : Vous pouvez aussi utiliser le dashboard unifie (`streamlit run src/ui/streamlit_app.py`) "
        "pour une experience tout-en-un avec upload direct de CSV."
    )

elif selected_page == "ingest":
    ingest_ui.render_ingest_ui()

elif selected_page == "clean":
    tab_clean, tab_features = st.tabs(["Nettoyage", "Feature Engineering"])
    with tab_clean:
        clean_ui.render_clean_ui()
    with tab_features:
        features_ui.render_features_ui()

elif selected_page == "features":
    features_ui.render_features_ui()

elif selected_page == "predict":
    predict_ui.render_predict_ui()

elif selected_page == "visualize":
    visualize_ui.render_visualize_ui()
