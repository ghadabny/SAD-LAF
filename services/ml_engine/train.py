import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Imports Configuration
from shared.config import config

# Imports GTFS
from services.ml_engine.data.gtfs.loader import GTFSLoader
from services.ml_engine.data.gtfs.preprocessor import GTFSPreprocessor

# Imports LAF
from services.ml_engine.data.laf.loader import LAFLoader
from services.ml_engine.data.laf.preprocessor import LAFPreprocessor

# Imports Features & Models
from services.ml_engine.features.pipeline import FeaturePipeline
from services.ml_engine.features.temporal import TemporalFeatureTransformer
from services.ml_engine.features.historical import HistoricalFeatureTransformer
from services.ml_engine.models.lgbm_model import LGBMScorer


def load_and_prepare_data() -> pd.DataFrame:
    """Charge et fusionne les données GTFS et LAF."""
    print("📊 1/4 - Chargement et préparation des données (GTFS + LAF)...")

    # 1. Traitement GTFS
    gtfs_loader = GTFSLoader()
    stop_times = gtfs_loader.load_stop_times()
    trips = gtfs_loader.load_trips()
    stops = gtfs_loader.load_stops()
    routes = gtfs_loader.load_routes()

    gtfs_preprocessor = GTFSPreprocessor()
    troncons_gtfs = gtfs_preprocessor.build_troncons(
        stop_times, trips, stops, routes, ter_only=True
    )

    # 2. Traitement LAF
    print("[LAFLoader] Chargement des historiques (CC)...")
    laf_loader = LAFLoader()
    df_cc = laf_loader.load_cc()
    # Note: Pour le ML actuel, on n'a besoin que des CC pour le taux d'irrégularité

    print("[LAFPreprocessor] Nettoyage et calcul du taux de fraude...")
    laf_preprocessor = LAFPreprocessor()
    cc_clean = laf_preprocessor.clean_cc(df_cc)
    stats_laf = laf_preprocessor.build_troncon_stats(cc_clean)

    # 3. Adaptation pour l'entraînement ML
    # Renommer la variable cible pour qu'elle corresponde à ce qu'attend LightGBM
    stats_laf = stats_laf.rename(columns={'taux_irregularite': 'fraud_score'})

    if stats_laf.empty:
        raise ValueError("❌ Erreur : Les statistiques LAF sont vides. Vérifie tes fichiers CSV.")

    # Extraire stop_id_dep et stop_id_arr depuis ton 'troncon_id' (format origin_dest_hour)
    stats_laf[['stop_id_dep', 'stop_id_arr', 'dep_hour_laf']] = stats_laf['troncon_id'].str.split('_', expand=True)

    # Agréger le score moyen au niveau de la paire (Origine, Destination) pour
    # éviter de multiplier les lignes lors de la jointure avec le GTFS.
    stats_laf_agg = stats_laf.groupby(['stop_id_dep', 'stop_id_arr'], as_index=False)['fraud_score'].mean()

    # 4. Fusion (Jointure Interne)
    df_train = pd.merge(
        troncons_gtfs,
        stats_laf_agg[['stop_id_dep', 'stop_id_arr', 'fraud_score']],
        on=['stop_id_dep', 'stop_id_arr'],
        how='inner'
    )

    if df_train.empty:
        raise ValueError("❌ Erreur : Le dataset d'entraînement est vide après la fusion GTFS/LAF.")

    print(f"✅ Données prêtes : {len(df_train)} tronçons historiques croisés trouvés.")
    return df_train


def extract_features(df_train: pd.DataFrame) -> pd.DataFrame:
    """Passe les données dans le pipeline de features et le sauvegarde."""
    print("⚙️ 2/4 - Création et apprentissage des Features...")

    # ATTENTION: Ici, pas de crochets parasites, c'est du Python pur !
    pipeline = FeaturePipeline([
        TemporalFeatureTransformer(),
        HistoricalFeatureTransformer()
    ])

    # Apprentissage et transformation
    df_features = pipeline.fit_transform(df_train)

    # Sauvegarde du pipeline pour l'API
    saved_pipeline_path = pipeline.save()
    print(f"✅ FeaturePipeline sauvegardé sous : {saved_pipeline_path}")

    return df_features


def train_and_evaluate(df_features: pd.DataFrame):
    """Entraîne LightGBM, évalue ses performances et sauvegarde le modèle + métriques."""
    print("🧠 3/4 - Séparation Train/Test, Entraînement et Évaluation...")

    # Séparation des données (80% apprentissage, 20% test)
    df_train_split, df_test_split = train_test_split(df_features, test_size=0.2, random_state=42)

    # Récupération des hyperparamètres depuis le YAML via config
    lgbm_params = config.lgbm_params

    # Instanciation et entraînement du modèle
    scorer = LGBMScorer(**lgbm_params)
    scorer.train(df_train_split, target_col='fraud_score')

    # Évaluation sur le jeu de test
    predictions_test = scorer.predict(df_test_split)
    vraies_valeurs = df_test_split['fraud_score']

    rmse = mean_squared_error(vraies_valeurs, predictions_test, squared=False)
    print(f"📉 Performance du modèle (RMSE sur Test) : {rmse:.4f}")

    # Sauvegarde du modèle
    saved_model_path = scorer.save()
    print(f"✅ Modèle LightGBM sauvegardé sous : {saved_model_path}")

    # Tracking minimaliste (POC)
    save_tracking_metrics(saved_model_path.parent, lgbm_params, rmse, len(df_train_split))


def save_tracking_metrics(models_dir: Path, params: dict, rmse: float, train_size: int):
    """Gère l'historique des entraînements dans un fichier JSON."""
    print("📊 4/4 - Sauvegarde des métriques de tracking...")

    run_metrics = {
        "date": datetime.now().isoformat(),
        "lgbm_params": params,
        "rmse_test": rmse,
        "nb_troncons_train": train_size
    }

    metrics_file = models_dir / "training_history.json"

    history = []
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                pass  # Fichier corrompu ou vide, on repart de zéro

    history.append(run_metrics)

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"✅ Tracking sauvegardé dans {metrics_file}")


def main():
    """Point d'entrée principal du script."""
    print("\n" + "=" * 50)
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT DU MODÈLE SAD-LAF")
    print("=" * 50 + "\n")

    try:
        # 1. Préparation
        df_train = load_and_prepare_data()

        # 2. Features
        df_features = extract_features(df_train)

        # 3. Entraînement & Évaluation & Tracking
        train_and_evaluate(df_features)

        print("\n🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
        print("👉 Les fichiers .joblib sont prêts à être consommés par l'API.")

    except Exception as e:
        print(f"\n❌ Erreur fatale lors de l'entraînement : {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()