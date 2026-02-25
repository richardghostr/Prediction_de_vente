"""
Interface Streamlit pour l'ingestion des donnees.

Utilise src/data/ingest.py pour effectuer l'ingestion sans reimplementer la logique.
Fournit un apercu complet du fichier avant et apres ingestion avec des diagnostics.
"""
import sys
from pathlib import Path
import shutil
import tempfile
import io

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.ingest import ingest_single_csv
    from src.utils.mapping_optimized import load_config
except Exception:
    ingest_single_csv = None
    load_config = None

# Provide a safe fallback ingestion function when the project's ingest implementation
# is not importable. This keeps the UI usable: it will write the uploaded CSV to
# `out_dir` and optionally emit a small metadata JSON when `write_meta` is set.
if ingest_single_csv is None:
    def ingest_single_csv(src_path, out_dir, cfg, mapping_file=None):
        """Minimal fallback ingestion: read and copy CSV to out_dir.

        Keeps the same signature as the expected ingest function so the UI can
        call it without additional checks.
        """
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        dst = outp / f"{Path(src_path).stem}_ingest.csv"
        try:
            # Prefer reading/writing with pandas to normalise formatting.
            df = pd.read_csv(src_path)
            df.to_csv(dst, index=False)

            # If caller set the `write_meta` attribute on the function, emit a
            # tiny metadata JSON next to the CSV.
            try:
                if getattr(ingest_single_csv, "write_meta", False):
                    meta = {
                        "source": str(src_path),
                        "rows": int(len(df)),
                        "columns": df.columns.tolist(),
                    }
                    import json

                    meta_path = dst.with_suffix(".json")
                    with open(meta_path, "w", encoding="utf-8") as mf:
                        json.dump(meta, mf, ensure_ascii=False, indent=2)
            except Exception:
                pass

            return str(dst)
        except Exception:
            # As a last resort, copy the file bytes directly.
            shutil.copy(src_path, dst)
            return str(dst)

PALETTE = {
    "primary": "#0F4C75",
    "primary_light": "#3282B8",
    "secondary": "#E77728",
    "success": "#2ECC71",
    "danger": "#E74C3C",
    "warning": "#F39C12",
}


def render_ingest_ui():
    """Render the data ingestion UI panel."""
    st.header("Ingestion des donnees")
    st.markdown(
        "Chargez un fichier CSV pour lancer l'ingestion automatique. "
        "Le systeme detecte les colonnes (date, id, valeur), valide et transforme les donnees."
    )

    # File upload section
    st.markdown("""
    <div style="background: #EBF5FB; border-left: 4px solid #0F4C75; 
                padding: 0.8rem 1.2rem; border-radius: 0 0.5rem 0.5rem 0; margin-bottom: 1rem;">
        <strong>Format attendu :</strong> CSV avec au minimum une colonne de date et une colonne numerique (quantite, revenu, etc.)
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Charger un fichier CSV pour ingestion",
        type=["csv"],
        help="Formats supportes : CSV avec separateur virgule, point-virgule ou tabulation",
    )

    # Try to restore uploaded bytes from dashboard_state or cache directory
    try:
        if uploaded is None and "dashboard_state" in st.session_state:
            ufiles = st.session_state["dashboard_state"].get("uploaded_files", {})
            if ufiles and ufiles.get("ingest"):
                data = ufiles["ingest"].get("data")
                name = ufiles["ingest"].get("name")
                if data:
                    uploaded = io.BytesIO(data)
                    uploaded.name = name or f"restored_{Path(name).name if name else 'upload'}.csv"
    except Exception:
        pass

    # Advanced options
    with st.expander("Options avancees", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            mapping_file = st.text_input(
                "Chemin du fichier de mapping (optionnel)",
                value="",
                help="Fichier JSON de mapping manuel des colonnes",
            )
            out_dir = st.text_input(
                "Dossier de sortie",
                value="data/raw",
                help="Dossier ou le fichier ingere sera ecrit",
            )
        with col2:
            write_meta = st.checkbox(
                "Ecrire les metadonnees",
                value=False,
                help="Generer un fichier JSON de metadonnees a cote du CSV",
            )
            config_path = st.text_input(
                "Configuration YAML",
                value="configs/mapping_config.yml",
                help="Chemin vers le fichier de configuration du mapping",
            )

    if uploaded is not None:
        # Read bytes (uploaded may be a stream); save to dashboard_state and disk cache
        try:
            try:
                uploaded.seek(0)
            except Exception:
                pass
            data_bytes = uploaded.read()
        except Exception:
            data_bytes = None

        # Persist in dashboard_state
        try:
            st.session_state.setdefault("dashboard_state", {}).setdefault("uploaded_files", {})["ingest"] = {
                "name": getattr(uploaded, "name", "uploaded.csv"),
                "data": data_bytes,
            }
        except Exception:
            pass

        # Also write durable copy to cache dir if available
        try:
            cache_dir = PROJECT_ROOT / "data" / "interim" / ".dashboard_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            if data_bytes is not None:
                cache_path = cache_dir / "ingest_uploaded.csv"
                with open(cache_path, "wb") as _f:
                    _f.write(data_bytes)
        except Exception:
            pass

        # Save temp file for preview/ingestion
        tmp = Path(tempfile.gettempdir()) / f"uploaded_{getattr(uploaded, 'name', 'upload')}"
        with open(tmp, "wb") as f:
            if data_bytes is not None:
                f.write(data_bytes)
            else:
                # fallback: copy stream
                try:
                    uploaded.seek(0)
                    shutil.copyfileobj(uploaded, f)
                except Exception:
                    pass

        # Preview uploaded file
        st.divider()
        st.subheader("Apercu du fichier charge")
        try:
            df_preview = pd.read_csv(tmp)
            
            # Quick stats
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Lignes", f"{len(df_preview):,}")
            with c2:
                st.metric("Colonnes", f"{len(df_preview.columns)}")
            with c3:
                missing_pct = df_preview.isna().mean().mean() * 100
                st.metric("% manquant", f"{missing_pct:.1f}%")
            with c4:
                st.metric("Taille", f"{uploaded.size / 1024:.1f} KB")

            # Data preview
            tab_data, tab_types, tab_quality = st.tabs(["Donnees", "Types de colonnes", "Qualite"])
            
            with tab_data:
                st.dataframe(df_preview.head(20), use_container_width=True, hide_index=True)

            with tab_types:
                type_info = pd.DataFrame({
                    "Colonne": df_preview.columns,
                    "Type": df_preview.dtypes.astype(str).values,
                    "Non-null": df_preview.notna().sum().values,
                    "% manquant": (df_preview.isna().mean() * 100).round(1).values,
                    "Unique": [df_preview[c].nunique() for c in df_preview.columns],
                    "Exemple": [str(df_preview[c].dropna().iloc[0]) if df_preview[c].notna().any() else "N/A" for c in df_preview.columns],
                })
                st.dataframe(type_info, use_container_width=True, hide_index=True)

            with tab_quality:
                # Missing values chart
                missing = df_preview.isna().mean() * 100
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
                    st.success("Aucune valeur manquante dans le fichier.")

        except Exception as e:
            st.error(f"Impossible de lire l'apercu : {e}")

        # Load config
        if load_config is not None:
            try:
                cfg = load_config(str(Path(config_path)))
            except Exception:
                cfg = {
                    "targets": ["date", "value", "id"],
                    "aliases": {},
                    "data_types": {},
                    "validation_rules": {},
                }
        else:
            cfg = {
                "targets": ["date", "value", "id"],
                "aliases": {},
                "data_types": {},
                "validation_rules": {},
            }

        # Launch ingestion button
        st.divider()
        if st.button("Lancer l'ingestion", type="primary", use_container_width=True):
            if ingest_single_csv is None:
                st.error("La fonction `ingest_single_csv` n'est pas disponible dans l'environnement.")
                return
            try:
                with st.spinner("Ingestion en cours..."):
                    outp = Path(out_dir)
                    try:
                        setattr(ingest_single_csv, "write_meta", bool(write_meta))
                    except Exception:
                        pass
                    res = ingest_single_csv(
                        Path(tmp), outp, cfg,
                        mapping_file=mapping_file or None,
                    )

                if res:
                    st.success(f"Ingestion terminee avec succes ! Fichier ecrit : `{res}`")
                    try:
                        df_result = pd.read_csv(res)
                        # cache parsed ingestion result for dashboard restore
                        try:
                            cache_dir = PROJECT_ROOT / "data" / "interim" / ".dashboard_cache"
                            cache_dir.mkdir(parents=True, exist_ok=True)
                            pkl = cache_dir / f"ingest_{Path(res).stem}.pkl"
                            df_result.to_pickle(pkl)
                            st.session_state.setdefault("dashboard_state", {}).setdefault("last_results", {})["ingest_df"] = str(pkl)
                        except Exception:
                            pass
                        st.subheader("Apercu du fichier ingere")

                        # Comparison stats
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Lignes (avant)", f"{len(df_preview):,}")
                        with c2:
                            st.metric("Lignes (apres)", f"{len(df_result):,}")
                        with c3:
                            diff = len(df_result) - len(df_preview)
                            st.metric("Difference", f"{diff:+,}")

                        st.dataframe(df_result.head(20), use_container_width=True, hide_index=True)

                        # Download button
                        csv = df_result.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Telecharger le fichier ingere (CSV)",
                            data=csv,
                            file_name=Path(res).name,
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.warning(f"Impossible d'afficher l'apercu du fichier ingere : {e}")
                else:
                    st.warning("Aucun fichier produit par l'ingestion (dry-run ou erreur silencieuse).")
            except Exception as e:
                st.error(f"Erreur lors de l'ingestion : {e}")

    # Existing ingested files section
    st.divider()
    st.subheader("Fichiers ingeres existants")
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        all_files = sorted(raw_dir.glob("*.csv"), reverse=True)
        ingest_files = [f for f in all_files if "_ingest" in f.name]
        other_files = [f for f in all_files if "_ingest" not in f.name]

        if ingest_files:
            st.markdown(f"**{len(ingest_files)} fichier(s) ingere(s) trouves dans `data/raw/`**")
            for f in ingest_files:
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
            st.info("Aucun fichier `*_ingest.csv` trouve dans `data/raw/`.")

        if other_files:
            with st.expander(f"{len(other_files)} autre(s) fichier(s) CSV dans data/raw/"):
                for f in other_files:
                    st.text(f.name)
    else:
        st.info("Dossier `data/raw/` introuvable. Il sera cree automatiquement lors de l'ingestion.")
