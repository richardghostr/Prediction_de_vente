"""
Interface Streamlit pour le Feature Engineering.

Utilise src/data/features.py pour generer les features temporelles a partir
des fichiers nettoyes. Visualise les features generees et permet l'export.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.features import build_feature_pipeline
except Exception:
    build_feature_pipeline = None

PALETTE = {
    "primary": "#0F4C75",
    "primary_light": "#3282B8",
    "secondary": "#E77728",
    "success": "#2ECC71",
    "danger": "#E74C3C",
    "warning": "#F39C12",
}


def render_features_ui():
    """Render the feature engineering UI panel."""
    st.header("Feature Engineering")
    st.markdown(
        "Generez des features temporelles (lags, moyennes mobiles, indicateurs calendaires) "
        "a partir des fichiers nettoyes pour alimenter le modele de prediction."
    )

    # File selection
    interim_dir = Path("data/interim")
    files = []
    if interim_dir.exists():
        files = sorted(interim_dir.glob("*_clean.csv"), reverse=True)

    if not files:
        st.info(
            "Aucun fichier nettoye trouve dans `data/interim/`. "
            "Executez d'abord le nettoyage dans l'onglet precedent."
        )
        return

    # Restore previous selection if present in dashboard_state
    saved_sel = None
    try:
        saved_sel = st.session_state.get("dashboard_state", {}).get("module_params", {}).get("features", {}).get("selected")
    except Exception:
        saved_sel = None

    options = [str(p) for p in files]
    index = options.index(saved_sel) if saved_sel in options else 0
    selected = st.selectbox(
        "Choisir un fichier nettoye",
        options=options,
        index=index,
        key="features_file_sel",
    )

    # persist selection
    try:
        st.session_state.setdefault("dashboard_state", {}).setdefault("module_params", {}).setdefault("features", {})["selected"] = selected
    except Exception:
        pass

    # Load selected file, prefer cached pickle
    df = None
    try:
        cache_dir = PROJECT_ROOT / "data" / "interim" / ".dashboard_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        cache_dir = None

    try:
        if cache_dir is not None:
            pkl = cache_dir / f"features_{Path(selected).stem}.pkl"
            if pkl.exists():
                try:
                    df = pd.read_pickle(pkl)
                except Exception:
                    df = None
        if df is None:
            df = pd.read_csv(selected)
            try:
                if cache_dir is not None:
                    pkl = cache_dir / f"features_{Path(selected).stem}.pkl"
                    df.to_pickle(pkl)
                    st.session_state.setdefault("dashboard_state", {}).setdefault("last_results", {})["features_df"] = str(pkl)
            except Exception:
                pass
    except Exception as e:
        st.error(f"Impossible de lire le fichier : {e}")
        return

    # Preview
    st.divider()
    st.subheader("Apercu du fichier selectionne")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Lignes", f"{len(df):,}")
    with c2:
        st.metric("Colonnes", f"{len(df.columns)}")
    with c3:
        num_count = len(df.select_dtypes(include=["number"]).columns)
        st.metric("Colonnes numeriques", f"{num_count}")

    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # Feature parameters
    st.divider()
    st.subheader("Parametres des features")

    st.markdown("""
    <div style="background: #EBF5FB; border-left: 4px solid #0F4C75; 
                padding: 0.8rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin-bottom: 1rem;">
        <strong>Lags :</strong> Valeurs passees de la serie (ex: lag 1 = valeur de la veille, lag 7 = il y a une semaine)<br>
        <strong>Rolling windows :</strong> Moyennes et ecarts-types glissants sur N jours
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        lags_input = st.text_input(
            "Lags (separes par virgule)",
            value="1,7",
            help="Ex: 1,7,14 pour les lags a 1, 7 et 14 jours",
            key="features_lags",
        )
    with c2:
        windows_input = st.text_input(
            "Rolling windows (separes par virgule)",
            value="3,7",
            help="Ex: 3,7,14 pour les moyennes mobiles sur 3, 7 et 14 jours",
            key="features_windows",
        )
    with c3:
        date_col = st.selectbox(
            "Colonne de date",
            options=df.columns.tolist(),
            index=df.columns.tolist().index("date") if "date" in df.columns else 0,
            key="features_date_col",
        )
        value_col_opts = df.select_dtypes(include=["number"]).columns.tolist() or df.columns.tolist()
        value_col = st.selectbox(
            "Colonne de valeur",
            options=value_col_opts,
            index=value_col_opts.index("value") if "value" in value_col_opts else 0,
            key="features_value_col",
        )

    def parse_list(s: str):
        try:
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        except Exception:
            return None

    # Generated features preview
    st.markdown("**Features qui seront generees :**")
    lags_list = parse_list(lags_input) or [1, 7]
    windows_list = parse_list(windows_input) or [3, 7]

    preview_features = (
        ["year", "month", "day", "dayofweek", "is_weekend"]
        + [f"lag_{l}" for l in lags_list]
        + [f"roll_mean_{w}" for w in windows_list]
        + [f"roll_std_{w}" for w in windows_list]
    )
    st.markdown(", ".join([f"`{f}`" for f in preview_features]))

    # Launch feature generation
    if st.button("Generer les features", type="primary", use_container_width=True, key="features_run"):
        if build_feature_pipeline is None:
            st.error("`build_feature_pipeline` non disponible dans l'environnement.")
            return

        lags = parse_list(lags_input)
        windows = parse_list(windows_input)

        if not lags:
            st.error("Format de lags invalide. Utilisez des entiers separes par des virgules.")
            return
        if not windows:
            st.error("Format de windows invalide. Utilisez des entiers separes par des virgules.")
            return

        try:
            with st.spinner("Generation des features en cours..."):
                result = build_feature_pipeline(
                    df,
                    date_col=date_col,
                    value_col=value_col,
                    lags=lags,
                    windows=windows,
                )

            # build_feature_pipeline may return (df, encoders) or just df
            if isinstance(result, tuple) and len(result) == 2:
                feat, _ = result
            else:
                feat = result

            st.success(f"Features generees ! {len(df.columns)} -> {len(feat.columns)} colonnes ({len(feat)} lignes)")

            # Stats
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Lignes", f"{len(feat):,}")
            with c2:
                st.metric("Colonnes", f"{len(feat.columns)}")
            with c3:
                new_cols = len(feat.columns) - len(df.columns)
                st.metric("Nouvelles features", f"+{new_cols}")
            with c4:
                nan_rows = feat.isna().any(axis=1).sum()
                st.metric("Lignes avec NaN", f"{nan_rows}")

            # Preview
            st.subheader("Apercu des features generees")
            st.dataframe(feat.head(15), use_container_width=True, hide_index=True)

            # Feature analysis
            st.divider()
            st.subheader("Analyse des features")

            tab_corr, tab_dist, tab_stats = st.tabs(["Correlation", "Distributions", "Statistiques"])

            with tab_corr:
                num_feat = feat.select_dtypes(include=["number"])
                if len(num_feat.columns) > 1:
                    corr = num_feat.corr()
                    # Show only correlation with value column
                    if value_col in corr.columns:
                        val_corr = corr[value_col].drop(value_col).sort_values(ascending=True)
                        fig_corr = px.bar(
                            x=val_corr.values, y=val_corr.index,
                            orientation="h",
                            labels={"x": f"Correlation avec '{value_col}'", "y": "Feature"},
                            color=val_corr.values,
                            color_continuous_scale="RdBu_r",
                            color_continuous_midpoint=0,
                        )
                        fig_corr.update_layout(
                            height=max(300, len(val_corr) * 28),
                            coloraxis_showscale=False,
                            margin=dict(l=0, r=0, t=10, b=0),
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

            with tab_dist:
                # Show distributions of new features
                new_feature_cols = [c for c in feat.columns if c not in df.columns]
                if new_feature_cols:
                    selected_feat = st.selectbox(
                        "Feature a visualiser",
                        options=new_feature_cols,
                        key="feat_dist_col",
                    )
                    fig_feat_dist = px.histogram(
                        feat, x=selected_feat, nbins=40,
                        color_discrete_sequence=[PALETTE["primary"]],
                        marginal="box",
                    )
                    fig_feat_dist.update_layout(height=350, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_feat_dist, use_container_width=True)
                else:
                    st.info("Aucune nouvelle feature numerique generee.")

            with tab_stats:
                st.dataframe(
                    feat.describe().round(3).T.reset_index().rename(columns={"index": "Feature"}),
                    use_container_width=True,
                    hide_index=True,
                )

            # Feature importance preview (correlation-based)
            if value_col in num_feat.columns and len(num_feat.columns) > 2:
                st.divider()
                st.subheader("Importance estimee des features (basee sur la correlation)")
                val_corr_abs = num_feat.corr()[value_col].drop(value_col).abs().sort_values(ascending=False)
                fig_imp = px.bar(
                    x=val_corr_abs.values[:15], y=val_corr_abs.index[:15],
                    orientation="h",
                    labels={"x": "Correlation absolue", "y": "Feature"},
                    color=val_corr_abs.values[:15],
                    color_continuous_scale=[[0, "#EBF5FB"], [1, PALETTE["primary"]]],
                )
                fig_imp.update_layout(
                    height=max(300, min(len(val_corr_abs), 15) * 30),
                    coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_imp, use_container_width=True)

            # Save option
            st.divider()
            save = st.checkbox("Enregistrer les features dans `data/processed/`", value=True, key="feat_save")
            if save:
                outd = Path("data/processed")
                outd.mkdir(parents=True, exist_ok=True)
                out_name = outd / f"{Path(selected).stem.replace('_clean', '')}_features{Path(selected).suffix}"
                feat.to_csv(out_name, index=False)
                st.info(f"Features enregistrees : `{out_name}`")

            # Download
            csv = feat.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Telecharger les features (CSV)",
                data=csv,
                file_name="features.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Erreur lors de la generation des features : {e}")

    # Existing feature files
    st.divider()
    st.subheader("Fichiers de features existants")
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        feature_files = sorted(processed_dir.glob("*_features.csv"), reverse=True)
        if feature_files:
            st.markdown(f"**{len(feature_files)} fichier(s) de features dans `data/processed/`**")
            for f in feature_files:
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
            st.info("Aucun fichier `*_features.csv` trouve.")
    else:
        st.info("Dossier `data/processed/` introuvable.")
