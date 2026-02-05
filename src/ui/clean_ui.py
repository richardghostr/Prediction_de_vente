"""
Interface Streamlit pour le nettoyage des donnees.

Utilise src/data/clean.py pour harmoniser et nettoyer les series temporelles.
Fournit une visualisation avant/apres nettoyage avec des diagnostics detailles.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.clean import clean_dataframe, save_interim
except Exception:
    clean_dataframe = None
    save_interim = None

PALETTE = {
    "primary": "#0F4C75",
    "primary_light": "#3282B8",
    "secondary": "#E77728",
    "success": "#2ECC71",
    "danger": "#E74C3C",
    "warning": "#F39C12",
}


def render_clean_ui():
    """Render the data cleaning UI panel."""
    st.header("Nettoyage des donnees")
    st.markdown(
        "Selectionnez un fichier ingere pour lancer le nettoyage automatique. "
        "Le processus harmonise les dates, remplit les valeurs manquantes et supprime les outliers."
    )

    # File selection
    raw_dir = Path("data/raw")
    files = []
    if raw_dir.exists():
        files = sorted(raw_dir.glob("*.csv"), reverse=True)

    if not files:
        st.info(
            "Aucun fichier CSV trouve dans `data/raw/`. "
            "Lancez d'abord l'ingestion ou placez un fichier dans ce dossier."
        )
        return

    selected = st.selectbox(
        "Choisir un fichier a nettoyer",
        options=["-- Selectionnez un fichier --"] + [str(p) for p in files],
        key="clean_file_sel",
    )

    if not selected or selected == "-- Selectionnez un fichier --":
        return

    # Load the file
    try:
        df = pd.read_csv(selected)
    except Exception as e:
        st.error(f"Impossible de lire le fichier : {e}")
        return

    # Preview & diagnostics
    st.divider()
    st.subheader("Diagnostics du fichier selectionne")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Lignes", f"{len(df):,}")
    with c2:
        st.metric("Colonnes", f"{len(df.columns)}")
    with c3:
        missing_pct = df.isna().mean().mean() * 100
        st.metric("% manquant global", f"{missing_pct:.1f}%")
    with c4:
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) > 0:
            outlier_count = 0
            for col in num_cols:
                s = df[col].dropna()
                if s.std() > 0:
                    z = ((s - s.mean()) / s.std()).abs()
                    outlier_count += (z > 3).sum()
            st.metric("Outliers (z>3)", f"{outlier_count}")
        else:
            st.metric("Outliers", "N/A")

    # Data preview
    tab_preview, tab_missing, tab_dist = st.tabs(["Apercu", "Valeurs manquantes", "Distribution"])

    with tab_preview:
        st.dataframe(df.head(15), use_container_width=True, hide_index=True)

    with tab_missing:
        missing = df.isna().mean() * 100
        missing_nonzero = missing[missing > 0].sort_values(ascending=True)
        if not missing_nonzero.empty:
            fig = px.bar(
                x=missing_nonzero.values, y=missing_nonzero.index,
                orientation="h",
                labels={"x": "% manquant", "y": "Colonne"},
                color=missing_nonzero.values,
                color_continuous_scale=[[0, PALETTE["success"]], [0.5, PALETTE["warning"]], [1, PALETTE["danger"]]],
            )
            fig.update_layout(
                height=max(200, len(missing_nonzero) * 35),
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("Aucune valeur manquante.")

    with tab_dist:
        num_cols_list = df.select_dtypes(include=["number"]).columns.tolist()
        if num_cols_list:
            dist_col = st.selectbox("Colonne", options=num_cols_list, key="clean_dist_col")
            fig_dist = px.histogram(
                df, x=dist_col, nbins=40,
                color_discrete_sequence=[PALETTE["primary"]],
                marginal="box",
            )
            fig_dist.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Aucune colonne numerique disponible.")

    # Cleaning parameters
    st.divider()
    st.subheader("Parametres de nettoyage")

    c1, c2, c3 = st.columns(3)
    with c1:
        date_col = st.selectbox(
            "Colonne de date",
            options=df.columns.tolist(),
            index=df.columns.tolist().index("date") if "date" in df.columns else 0,
            key="clean_date_col",
        )
    with c2:
        value_col_options = df.select_dtypes(include=["number"]).columns.tolist() or df.columns.tolist()
        default_val = "value" if "value" in value_col_options else value_col_options[0] if value_col_options else df.columns[0]
        value_col = st.selectbox(
            "Colonne de valeur",
            options=value_col_options,
            index=value_col_options.index(default_val) if default_val in value_col_options else 0,
            key="clean_value_col",
        )
    with c3:
        fill_strategy = st.selectbox(
            "Strategie de remplissage",
            options=["ffill", "bfill", "mean", "zero"],
            format_func=lambda x: {
                "ffill": "Forward fill (propager vers l'avant)",
                "bfill": "Backward fill (propager vers l'arriere)",
                "mean": "Moyenne de la colonne",
                "zero": "Remplacer par 0",
            }.get(x, x),
            key="clean_fill_strategy",
        )

    outlier = st.number_input(
        "Seuil z-score pour outliers (0 pour desactiver)",
        value=3.0,
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        help="Les valeurs avec un z-score superieur a ce seuil seront supprimees",
        key="clean_outlier",
    )

    # Launch cleaning
    if st.button("Lancer le nettoyage", type="primary", use_container_width=True, key="clean_run"):
        if clean_dataframe is None:
            st.error("`clean_dataframe` non disponible dans l'environnement.")
            return
        try:
            with st.spinner("Nettoyage en cours..."):
                cleaned = clean_dataframe(
                    df,
                    date_col=date_col,
                    value_col=value_col,
                    fill_strategy=fill_strategy,
                    outlier_threshold=(None if outlier == 0 else outlier),
                )

            st.success(f"Nettoyage termine ! {len(df)} -> {len(cleaned)} lignes.")

            # Before/After comparison
            st.subheader("Comparaison avant / apres")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Lignes avant", f"{len(df):,}")
            with c2:
                st.metric("Lignes apres", f"{len(cleaned):,}")
            with c3:
                diff = len(cleaned) - len(df)
                st.metric("Lignes supprimees", f"{abs(diff)}")

            # Before/after missing values
            before_missing = df.isna().sum().sum()
            after_missing = cleaned.isna().sum().sum()
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Valeurs manquantes (avant)", f"{before_missing:,}")
            with c2:
                st.metric("Valeurs manquantes (apres)", f"{after_missing:,}")

            # Preview cleaned data
            st.subheader("Apercu des donnees nettoyees")
            st.dataframe(cleaned.head(15), use_container_width=True, hide_index=True)

            # Before/after value distribution
            if value_col in df.columns and value_col in cleaned.columns:
                st.subheader("Distribution avant / apres nettoyage")
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["Avant nettoyage", "Apres nettoyage"],
                )
                fig.add_trace(
                    go.Histogram(x=df[value_col].dropna(), name="Avant", marker_color=PALETTE["danger"], opacity=0.7),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Histogram(x=cleaned[value_col].dropna(), name="Apres", marker_color=PALETTE["success"], opacity=0.7),
                    row=1, col=2,
                )
                fig.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

            # Save option
            st.divider()
            if save_interim is not None:
                save = st.checkbox("Enregistrer le resultat dans `data/interim/`", value=True, key="clean_save")
                if save:
                    outp = Path("data/interim")
                    outp.mkdir(parents=True, exist_ok=True)
                    fname = outp / f"{Path(selected).stem.replace('_ingest', '')}_clean.csv"
                    save_interim(cleaned, str(fname))
                    st.info(f"Fichier nettoye enregistre : `{fname}`")

            # Download
            csv = cleaned.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Telecharger les donnees nettoyees (CSV)",
                data=csv,
                file_name="donnees_nettoyees.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Erreur pendant le nettoyage : {e}")

    # Existing cleaned files
    st.divider()
    st.subheader("Fichiers nettoyes existants")
    interim_dir = Path("data/interim")
    if interim_dir.exists():
        clean_files = sorted(interim_dir.glob("*_clean.csv"), reverse=True)
        if clean_files:
            st.markdown(f"**{len(clean_files)} fichier(s) nettoye(s) dans `data/interim/`**")
            for f in clean_files:
                with st.expander(f.name, expanded=False):
                    try:
                        df_existing = pd.read_csv(f)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Lignes", f"{len(df_existing):,}")
                        with c2:
                            st.metric("Colonnes", f"{len(df_existing.columns)}")
                        with c3:
                            size_kb = f.stat().st_size / 1024
                            st.metric("Taille", f"{size_kb:.1f} KB")
                        st.dataframe(df_existing.head(5), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.warning(f"Impossible de lire : {e}")
        else:
            st.info("Aucun fichier `*_clean.csv` trouve.")
    else:
        st.info("Dossier `data/interim/` introuvable.")
