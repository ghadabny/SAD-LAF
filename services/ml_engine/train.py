# services/ml_engine/train.py
"""
Point d'entrée de l'entraînement du modèle SAD-LAF.

Architecture du pipeline d'entraînement :

    ┌─────────────────────────────────────────────────────────────────┐
    │  ÉTAPE 1 : GTFS (rapide, ~10s, pas de cache nécessaire)        │
    │  load_stop_times + load_trips + load_stops + load_routes        │
    │  → troncons_gtfs (296K tronçons TER)                           │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────────┐
    │  ÉTAPE 2 : LAF STATS (lent, ~30min sans cache)       [CACHE L1] │
    │  SC + CC + PV → build_unified_stats()                           │
    │  → fraud_score, nb_controles, nb_pv, pv_intensity, ...         │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────────┐
    │  ÉTAPE 3 : MERGE GTFS × LAF          [CACHE L2]                 │
    │  → df_merged (78K-100K tronçons avec historique brut)           │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────────┐
    │  ÉTAPE 4 : TRAIN/TEST SPLIT  ← DOIT PRÉCÉDER LE FIT DU PIPELINE│
    │  train_test_split(df_merged, test_size=0.2, random_state=42)    │
    │  → df_train_raw (80%) | df_test_raw (20%)                       │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────────┐
    │  ÉTAPE 5 : FEATURES — fit sur TRAIN uniquement                   │
    │  pipeline.fit_transform(df_train_raw)                           │
    │  pipeline.transform(df_test_raw)      ← jamais fit() sur test   │
    │  Sauvegarde : feature_pipeline.joblib                           │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
    ┌───────────────────────────▼─────────────────────────────────────┐
    │  ÉTAPE 6 : ENTRAÎNEMENT LIGHTGBM + ÉVALUATION                   │
    │  LGBMScorer.train(df_train_features)                            │
    │  scorer.evaluate(df_test_features)                              │
    │  Sauvegarde : lgbm_scorer.joblib + training_history.json        │
    └─────────────────────────────────────────────────────────────────┘

CORRECTION ANTI-LEAKAGE (v2) :
    Le split train/test se fait maintenant AVANT fit_transform().
    Avant (v1, BUGUÉ) :
        df_features = pipeline.fit_transform(df_COMPLET)   # ← fuite temporelle
        df_train, df_test = train_test_split(df_features)
    Après (v2, CORRIGÉ) :
        df_train_raw, df_test_raw = train_test_split(df_merged)  # ← split d'abord
        df_train = pipeline.fit_transform(df_train_raw)          # ← fit sur train seul
        df_test  = pipeline.transform(df_test_raw)               # ← transform seulement

Options CLI :
    --force-recompute-laf   : invalide le cache L1 (LAF stats)
    --force-recompute-all   : invalide les caches L1 + L2
    (sans option)           : utilise les caches existants si disponibles

Exemple :
    python services/ml_engine/train.py
    python services/ml_engine/train.py --force-recompute-laf
    python services/ml_engine/train.py --force-recompute-all
"""
import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

try:
    from sklearn.metrics import root_mean_squared_error as _rmse_fn
    def _rmse(y_true, y_pred):
        return _rmse_fn(y_true, y_pred)
except ImportError:
    from sklearn.metrics import mean_squared_error as _mse_fn
    def _rmse(y_true, y_pred):
        return _mse_fn(y_true, y_pred, squared=False)

from sklearn.model_selection import train_test_split

from shared.config import config
from services.ml_engine.data.cache import DataCache
from services.ml_engine.data.gtfs.loader import GTFSLoader
from services.ml_engine.data.gtfs.preprocessor import GTFSPreprocessor
from services.ml_engine.data.laf.loader import LAFLoader
from services.ml_engine.data.laf.preprocessor import LAFPreprocessor
from services.ml_engine.features.pipeline import FeaturePipeline
from services.ml_engine.features.temporal import TemporalFeatureTransformer
from services.ml_engine.features.historical import HistoricalFeatureTransformer
from services.ml_engine.models.lgbm_model import LGBMScorer


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 : GTFS
# ─────────────────────────────────────────────────────────────────────────────

def _load_gtfs() -> pd.DataFrame:
    """
    Charge et prépare les tronçons GTFS TER.
    Pas de cache nécessaire : charge en ~10s et reste stable.
    """
    print("\n📡 GTFS — Chargement des tronçons TER...")
    loader = GTFSLoader()
    troncons = GTFSPreprocessor().build_troncons(
        loader.load_stop_times(),
        loader.load_trips(),
        loader.load_stops(),
        loader.load_routes(),
        ter_only=True,
    )
    print(f"   → {len(troncons):,} tronçons GTFS TER disponibles.")
    return troncons


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 : LAF STATS [CACHE L1]
# ─────────────────────────────────────────────────────────────────────────────

def _build_laf_unified_stats(
    cache: DataCache,
    loader: LAFLoader,
    preprocessor: LAFPreprocessor,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Produit les statistiques unifiées SC + CC + PV par troncon_id.

    IMPORTANT : cette fonction calcule des statistiques sur la TOTALITÉ
    de la période historique (2022-2026). Ce n'est pas du leakage ici car
    ces statistiques représentent le "profil historique long terme" d'un
    tronçon — elles ne fuient pas d'une fenêtre temporelle test vers train.
    Le leakage était dans le FIT du transformer sur le dataset complet
    (corrigé dans _build_features_and_split).
    """
    laf_paths = loader.get_all_laf_paths()
    if not laf_paths:
        raise FileNotFoundError(
            f"❌ Aucun fichier LAF trouvé dans '{config.LAF_DIR}'. "
            f"Vérifie que les fichiers CC, SC et PV sont présents."
        )

    cache_key = cache.make_key("laf_unified_stats", laf_paths)

    if not force_recompute:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    print("\n🔄 LAF — Calcul des statistiques (non caché, peut prendre ~30min)...")
    print(f"   Fichiers source : {[p.name for p in laf_paths]}")

    # ── Traitement SC (par chunks pour économiser la RAM) ──────────────────
    print("\n   [SC] Traitement des titres scannés par chunks de 500K lignes...")
    sc_stats_list: list[pd.DataFrame] = []
    chunk_count = 0

    for chunk in loader.load_sc_chunked(chunksize=500_000):
        clean = preprocessor.clean_cc(chunk)
        if not clean.empty:
            sc_stats_list.append(preprocessor.build_troncon_stats(clean))
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"   [SC] {chunk_count} chunks traités ({chunk_count * 500_000:,} lignes)...")

    print(f"   [SC] Terminé : {chunk_count} chunks, {len(sc_stats_list)} chunks avec données.")

    # ── Traitement CC ──────────────────────────────────────────────────────
    print("\n   [CC] Traitement des contrôles comportés...")
    cc_clean = preprocessor.clean_cc(loader.load_cc())
    if not cc_clean.empty:
        sc_stats_list.append(preprocessor.build_troncon_stats(cc_clean))
        print(f"   [CC] {len(cc_clean):,} lignes valides ajoutées.")
    else:
        print("   [CC] Aucune ligne valide.")

    if not sc_stats_list:
        raise ValueError(
            "❌ Aucune donnée CC/SC valide après nettoyage. "
            "Vérifie que les fichiers SC trimestriels sont présents et lisibles."
        )

    # ── Agrégation finale CC/SC ────────────────────────────────────────────
    print("\n   [CC/SC] Agrégation finale sur tous les fichiers/chunks...")
    cc_sc_stats = (
        pd.concat(sc_stats_list, ignore_index=True)
        .groupby("troncon_id", as_index=False)
        .agg(
            nb_controles=(  "nb_controles",   "sum"),
            nb_irregularites=("nb_irregularites", "sum"),
        )
    )
    cc_sc_stats["taux_irregularite"] = (
        cc_sc_stats["nb_irregularites"] / cc_sc_stats["nb_controles"]
    ).round(4)
    print(
        f"   [CC/SC] {len(cc_sc_stats):,} tronçons uniques, "
        f"taux moyen = {cc_sc_stats['taux_irregularite'].mean():.3f}."
    )

    # ── Traitement PV ──────────────────────────────────────────────────────
    print("\n   [PV] Traitement des procès-verbaux...")
    pv_raw = loader.load_pv()
    if pv_raw.empty:
        print("   [PV] Fichier absent ou vide — fraud_score basé sur CC/SC uniquement.")
        pv_stats = pd.DataFrame()
    else:
        pv_clean = preprocessor.clean_pv(pv_raw)
        pv_stats = preprocessor.build_pv_stats(pv_clean)
        print(f"   [PV] {len(pv_stats):,} tronçons avec au moins 1 PV.")

    # ── Unification ────────────────────────────────────────────────────────
    print("\n   Unification CC/SC + PV → fraud_score unifié...")
    unified = preprocessor.build_unified_stats(cc_sc_stats, pv_stats)

    cache.set(cache_key, unified)
    return unified


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 : MERGE GTFS × LAF [CACHE L2]
# ─────────────────────────────────────────────────────────────────────────────

def _build_training_dataset(
    cache: DataCache,
    troncons_gtfs: pd.DataFrame,
    unified_laf_stats: pd.DataFrame,
    gtfs_paths: list[Path],
    laf_paths: list[Path],
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Joint les tronçons GTFS avec les stats LAF unifiées.

    Retourne df_merged : tronçons avec colonnes brutes (nb_controles,
    nb_pv, pv_intensity, fraud_score...) mais PAS encore les features
    hist_* — celles-ci seront calculées après le split.
    """
    all_paths = sorted(set(gtfs_paths + laf_paths))
    cache_key = cache.make_key("training_dataset", all_paths)

    if not force_recompute:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    print("\n🔗 MERGE — Jointure GTFS × LAF...")

    split = unified_laf_stats["troncon_id"].str.split("_", expand=True)
    if split.shape[1] < 3:
        raise ValueError(
            "❌ Impossible de parser troncon_id. "
            "Format attendu : 'origin_uic_dest_uic_hour' (3 parties séparées par '_')."
        )
    unified_laf_stats = unified_laf_stats.copy()
    unified_laf_stats["stop_id_dep"] = split[0]
    unified_laf_stats["stop_id_arr"] = split[1]

    numeric_sum_cols  = ["nb_controles", "nb_irregularites", "nb_pv",
                         "nb_pv_tariff", "nb_pv_non_tariff"]
    numeric_mean_cols = ["fraud_score", "pv_intensity", "pct_pv_tariff",
                         "montant_moyen_pv_cents"]

    sum_cols_present  = [c for c in numeric_sum_cols  if c in unified_laf_stats.columns]
    mean_cols_present = [c for c in numeric_mean_cols if c in unified_laf_stats.columns]

    agg_dict = {c: "sum"  for c in sum_cols_present}
    agg_dict.update({c: "mean" for c in mean_cols_present})

    stats_od = (
        unified_laf_stats
        .groupby(["stop_id_dep", "stop_id_arr"], as_index=False)
        .agg(agg_dict)
    )

    if "nb_controles" in stats_od.columns and "nb_pv" in stats_od.columns:
        denom = (stats_od["nb_controles"] + stats_od["nb_pv"]).replace(0, float("nan"))
        stats_od["fraud_score"] = (
            (stats_od.get("nb_irregularites", 0) + stats_od["nb_pv"]) / denom
        ).fillna(0.0).round(4)

    print(f"   {len(stats_od):,} paires O/D avec historique LAF.")

    df_merged = pd.merge(
        troncons_gtfs,
        stats_od,
        on=["stop_id_dep", "stop_id_arr"],
        how="inner",
    )

    if df_merged.empty:
        raise ValueError(
            "❌ Dataset vide après merge GTFS × LAF.\n"
            "Cause probable : les codes UIC dans les fichiers LAF ne correspondent "
            "pas aux stop_id nettoyés du GTFS (format attendu : 8 chiffres).\n"
            "Vérifie que GTFSPreprocessor.clean_stop_id() produit bien des UIC "
            "au même format que ticket_travelInformation_origin_uicCode dans les SC/CC."
        )

    coverage = len(df_merged["stop_id_dep"].unique())
    print(
        f"   ✅ {len(df_merged):,} tronçons avec historique LAF "
        f"({coverage:,} gares de départ couvertes)."
    )

    cache.set(cache_key, df_merged)
    return df_merged


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPES 4+5 : SPLIT D'ABORD, FEATURES ENSUITE
# ─────────────────────────────────────────────────────────────────────────────

def _split_then_build_features(
    df_merged: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Effectue le split train/test SUR LES DONNÉES BRUTES,
    puis construit les features en fittant le pipeline UNIQUEMENT sur le train.

    POURQUOI CET ORDRE EST CRITIQUE (anti-leakage temporel) :
        HistoricalFeatureTransformer.fit() calcule les valeurs de fallback
        et remplit le dictionnaire de lookup O/D sur le train uniquement.

    Retourne :
        (df_train_features, df_test_features)
    """
    print("\n✂️  SPLIT — Division train/test sur données brutes...")

    # ── Ajout de service_date AVANT le split ──────────────────────────────
    # TODO : remplacer par la vraie date de circulation par tronçon depuis
    # le GTFS (colonne service_date réelle). Actuellement tous les tronçons
    # reçoivent date.today(), ce qui neutralise les features temporelles
    # calendaires (is_vacances, is_jour_ferie, is_peak_hour) puisqu'elles
    # sont identiques pour tous les tronçons du dataset d'entraînement.
    df_merged = df_merged.copy()
    df_merged["service_date"] = date.today()

    df_train_raw, df_test_raw = train_test_split(
        df_merged, test_size=0.2, random_state=42
    )
    print(
        f"   Train brut : {len(df_train_raw):,} tronçons | "
        f"Test brut  : {len(df_test_raw):,} tronçons"
    )

    # ── Construction du pipeline et fit EXCLUSIVEMENT sur le train ────────
    print("\n⚙️  FEATURES — Fit du pipeline sur le train uniquement...")

    pipeline = FeaturePipeline([
        TemporalFeatureTransformer(),
        HistoricalFeatureTransformer(),
    ])

    df_train_features = pipeline.fit_transform(df_train_raw)
    df_test_features  = pipeline.transform(df_test_raw)

    saved_path = pipeline.save()
    print(f"   ✅ FeaturePipeline sauvegardé → {saved_path}")

    _verify_no_leakage_between_splits(df_train_raw, df_test_raw)

    print(
        f"\n   Train features : {len(df_train_features):,} lignes × "
        f"{len(df_train_features.columns)} colonnes"
    )
    print(
        f"   Test  features : {len(df_test_features):,} lignes × "
        f"{len(df_test_features.columns)} colonnes"
    )

    return df_train_features, df_test_features


def _verify_no_leakage_between_splits(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
) -> None:
    """
    Vérification sanité post-split.
    Loggue les paires O/D du test jamais vues dans le train → fallback attendu.
    Non bloquant.
    """
    if "stop_id_dep" not in df_train_raw.columns:
        return

    train_od = set(
        zip(df_train_raw["stop_id_dep"].astype(str),
            df_train_raw["stop_id_arr"].astype(str))
    )
    test_od = set(
        zip(df_test_raw["stop_id_dep"].astype(str),
            df_test_raw["stop_id_arr"].astype(str))
    )
    unseen  = test_od - train_od
    overlap = test_od & train_od

    print(
        f"\n   [Sanité split] Paires O/D train : {len(train_od):,} | "
        f"test : {len(test_od):,}"
    )
    print(
        f"   [Sanité split] Overlap train/test : {len(overlap):,} paires "
        f"({100*len(overlap)/len(test_od):.1f}%)"
    )
    if unseen:
        print(
            f"   [Sanité split] {len(unseen):,} paires O/D jamais vues en train "
            f"→ fallback appliqué (attendu en prod, OK)."
        )
    else:
        print("   [Sanité split] ✅ Toutes les paires O/D test sont couvertes par le train.")


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 : ENTRAÎNEMENT & ÉVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _train_and_evaluate(
    df_train_features: pd.DataFrame,
    df_test_features: pd.DataFrame,
) -> None:
    """
    Entraîne LightGBM, évalue les performances et sauvegarde le modèle.
    """
    print("\n🧠 ENTRAÎNEMENT — LightGBM + évaluation...")
    print(
        f"   Train : {len(df_train_features):,} tronçons | "
        f"Test  : {len(df_test_features):,} tronçons"
    )

    lgbm_params = config.lgbm_params
    scorer = LGBMScorer(**lgbm_params)
    scorer.train(df_train_features, target_col="fraud_score")

    metrics = scorer.evaluate(df_test_features, target_col="fraud_score")
    rmse = metrics["rmse"]

    # Prédire une seule fois pour les statistiques descriptives
    preds = scorer.predict(df_test_features)
    print(f"\n   📉 RMSE sur Test : {rmse:.4f}")
    print(
        f"   Prédictions — "
        f"min={preds.min():.3f}, "
        f"max={preds.max():.3f}, "
        f"moyenne={preds.mean():.3f}"
    )

    saved_model = scorer.save()
    print(f"   ✅ Modèle LightGBM sauvegardé → {saved_model}")

    _save_tracking_metrics(
        models_dir=saved_model.parent,
        params=lgbm_params,
        metrics=metrics,
        train_size=len(df_train_features),
        test_size=len(df_test_features),
    )


def _save_tracking_metrics(
    models_dir: Path,
    params: dict,
    metrics: dict,
    train_size: int,
    test_size: int,
) -> None:
    """Append les métriques du run courant dans training_history.json."""
    print("\n📊 TRACKING — Sauvegarde des métriques...")

    run_metrics = {
        "date":               datetime.now().isoformat(),
        "lgbm_params":        params,
        "rmse_test":          metrics["rmse"],
        "mae_test":           metrics.get("mae"),
        "r2_test":            metrics.get("r2"),
        "spearman_rho_test":  metrics.get("spearman_rho"),
        "nb_troncons_train":  train_size,
        "nb_troncons_test":   test_size,
    }

    metrics_file = models_dir / "training_history.json"
    history = []
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            print("   ⚠️  training_history.json corrompu — réinitialisé.")

    history.append(run_metrics)
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"   ✅ Tracking sauvegardé → {metrics_file} ({len(history)} runs)")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement du modèle SAD-LAF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python services/ml_engine/train.py
      → Utilise les caches existants (rapide si déjà calculé)

  python services/ml_engine/train.py --force-recompute-laf
      → Recalcule les stats LAF (nouvelle livraison de données)

  python services/ml_engine/train.py --force-recompute-all
      → Tout recalculer depuis zéro (~30min)
        """,
    )
    parser.add_argument(
        "--force-recompute-laf",
        action="store_true",
        help="Invalide le cache LAF (L1) et recalcule les stats SC+CC+PV.",
    )
    parser.add_argument(
        "--force-recompute-all",
        action="store_true",
        help="Invalide tous les caches (L1 + L2) et recalcule tout.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("\n" + "=" * 60)
    print("🚀  DÉMARRAGE — SAD-LAF Model Training v2 (anti-leakage)")
    print("=" * 60)

    force_laf = args.force_recompute_laf or args.force_recompute_all
    force_all = args.force_recompute_all

    if force_laf:
        print("\n⚠️  --force-recompute-laf : recalcul des stats LAF forcé.")
    if force_all:
        print("⚠️  --force-recompute-all  : recalcul complet forcé.")

    cache        = DataCache(config.PROCESSED_DIR)
    loader       = LAFLoader()
    preprocessor = LAFPreprocessor()

    try:
        # 1. GTFS
        troncons_gtfs = _load_gtfs()
        gtfs_dir   = config.GTFS_DIR
        gtfs_paths = sorted(gtfs_dir.glob("*.txt")) if gtfs_dir.exists() else []

        # 2. LAF stats [CACHE L1]
        unified_laf = _build_laf_unified_stats(
            cache, loader, preprocessor,
            force_recompute=force_laf,
        )

        # 3. Merge [CACHE L2]
        laf_paths = loader.get_all_laf_paths()
        df_merged = _build_training_dataset(
            cache, troncons_gtfs, unified_laf,
            gtfs_paths=gtfs_paths,
            laf_paths=laf_paths,
            force_recompute=force_all,
        )

        # 4+5. Split PUIS features (ordre anti-leakage)
        df_train_features, df_test_features = _split_then_build_features(df_merged)

        # 6. Entraînement
        _train_and_evaluate(df_train_features, df_test_features)

        print("\n" + "=" * 60)
        print("🎉  ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
        print("   Les fichiers .joblib sont prêts pour l'API.")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌  Erreur fatale : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()