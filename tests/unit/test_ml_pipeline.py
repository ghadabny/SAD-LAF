"""
tests/integration/test_ml_pipeline.py — Chaîne ML complète : Train → Save → Load → Predict.

Ces tests d'intégration vérifient que les briques ML fonctionnent ENSEMBLE :
    1. FeaturePipeline (TemporalFeatureTransformer + HistoricalFeatureTransformer)
    2. LGBMScorer (entraînement, sauvegarde, rechargement, prédiction)

Ils couvrent le scénario de déploiement réel :
    Entraîner sur données LAF → sauvegarder .joblib → déployer l'API → prédire

Pourquoi des tests d'intégration et pas uniquement unitaires ?
    Un test unitaire de pipeline.save() ne valide pas que scorer.predict()
    utilise correctement les features produites par ce pipeline.
    L'intégration entre les deux modules est critique — c'est le but ici.

Isolation :
    tmp_path + monkeypatch → chaque test a son propre dossier MODELS_DIR.
    Les fichiers .joblib sont supprimés automatiquement après le test.
    Aucune dépendance à data/models/ ou aux données LAF réelles.

Performance :
    LGBMScorer est instancié avec n_estimators=10 pour être rapide en tests.
    (< 1 seconde vs ~30 secondes pour un entraînement complet)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from services.ml_engine.features.pipeline import FeaturePipeline
from services.ml_engine.features.temporal import TemporalFeatureTransformer
from services.ml_engine.features.historical import HistoricalFeatureTransformer
from services.ml_engine.models.lgbm_model import LGBMScorer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Gares du réseau alsacien (codes UIC 8 chiffres)
GARES = [
    ("87212027", "87214007"),  # Strasbourg → Sélestat
    ("87214007", "87214080"),  # Sélestat → Colmar
    ("87214080", "87182063"),  # Colmar → Mulhouse
    ("87182063", "87191007"),  # Mulhouse → Obernai
]


@pytest.fixture
def synthetic_training_data() -> pd.DataFrame:
    """
    Dataset synthétique avec signal de fraude pour entraîner LightGBM.

    Caractéristiques :
        - 120 tronçons (suffisant pour LightGBM, acceptable en CI)
        - 4 paires O/D avec historique — HistoricalFeatureTransformer peut fit
        - Signal de fraude : tronçons en heure de pointe + Strasbourg → score plus élevé
        - Couverture de weekends et vacances (utile pour TemporalFeatureTransformer)

    Le signal doit être présent pour que LightGBM apprenne quelque chose
    et que Spearman > 0 sur le test set.
    """
    np.random.seed(42)
    n = 120

    records = []
    base_date = date(2024, 9, 2)  # lundi

    for i in range(n):
        od_idx  = i % len(GARES)
        stop_dep, stop_arr = GARES[od_idx]

        # Horaires variés : matin, midi, soir, heure de pointe
        dep_min = np.random.choice([
            8 * 60 + 23,   # pointe matin
            10 * 60,       # hors pointe
            12 * 60 + 30,  # déjeuner
            17 * 60 + 30,  # pointe soir
            19 * 60,       # hors pointe soir
        ])

        # Date de service : cycle de 4 semaines
        service_date = base_date + timedelta(days=i % 28)

        # Signal de fraude :
        #   - Heures de pointe → fraude plus élevée
        #   - Strasbourg → Sélestat → tronçon historiquement chaud
        #   + bruit gaussien
        is_peak = (7 * 60 <= dep_min < 9 * 60) or (17 * 60 <= dep_min < 19 * 60)
        is_hot_od = (stop_dep == "87212027")  # Strasbourg départ = plus de fraude

        base_score = 0.35 if is_peak else 0.15
        if is_hot_od:
            base_score += 0.15
        noise = np.random.normal(0, 0.05)
        fraud = float(np.clip(base_score + noise, 0.0, 1.0))

        # Colonnes LAF agrégées — requises par HistoricalFeatureTransformer.fit()
        # _REQUIRED_SOURCE_COLS = ['nb_controles', 'pct_pv_tariff']
        # Ces valeurs simulent le résultat de build_unified_stats() sur un vrai
        # historique LAF. Le signal est volontairement cohérent avec fraud_score
        # pour que LightGBM puisse apprendre quelque chose sur le jeu de test.
        nb_controles = np.random.randint(10, 150)
        nb_pv        = int(nb_controles * fraud * np.random.uniform(0.8, 1.2))
        nb_pv        = max(0, min(nb_pv, nb_controles))
        nb_pv_tariff = int(nb_pv * np.random.uniform(0.4, 0.9))

        records.append({
            "trip_id":         f"TRIP_{i:03d}",
            "train_number":    f"1177{i % 100:02d}",
            "service_id":      "000001",
            "stop_sequence":   i % 5,
            "stop_id_dep":     stop_dep,
            "stop_name_dep":   "Gare A",
            "dep_minutes":     int(dep_min),
            "stop_id_arr":     stop_arr,
            "stop_name_arr":   "Gare B",
            "arr_minutes":     int(dep_min) + 30,
            "duration_min":    30,
            "service_date":    service_date,
            # ── Colonnes LAF ──────────────────────────────────────────────────
            "nb_controles":    nb_controles,
            "nb_pv":           nb_pv,
            "nb_pv_tariff":    nb_pv_tariff,
            "nb_pv_non_tariff": nb_pv - nb_pv_tariff,
            "pv_intensity":    round(nb_pv / max(nb_controles, 1), 4),
            "pct_pv_tariff":   round(nb_pv_tariff / max(nb_pv, 1), 4),
            "montant_moyen_pv_cents": int(np.random.uniform(50, 300)),
            # ── Cible ─────────────────────────────────────────────────────────
            "fraud_score":     round(fraud, 4),
        })

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_inference_data() -> pd.DataFrame:
    """
    Dataset synthétique pour l'inférence (sans la colonne fraud_score).
    Simule des tronçons GTFS du jour à scorer après déploiement.
    """
    return pd.DataFrame([
        {
            "trip_id": "TRIP_INF_001", "train_number": "117756",
            "service_id": "000001", "stop_sequence": 0,
            "stop_id_dep": "87212027", "stop_name_dep": "Strasbourg",
            "dep_minutes": 8 * 60 + 23,  # pointe matin
            "stop_id_arr": "87214007", "stop_name_arr": "Sélestat",
            "arr_minutes": 8 * 60 + 52, "duration_min": 29,
            "service_date": date(2024, 9, 2),  # lundi
        },
        {
            "trip_id": "TRIP_INF_002", "train_number": "117758",
            "service_id": "000001", "stop_sequence": 0,
            "stop_id_dep": "87214007", "stop_name_dep": "Sélestat",
            "dep_minutes": 10 * 60,  # hors pointe
            "stop_id_arr": "87214080", "stop_name_arr": "Colmar",
            "arr_minutes": 10 * 60 + 23, "duration_min": 23,
            "service_date": date(2024, 9, 7),  # samedi
        },
        {
            "trip_id": "TRIP_INF_003", "train_number": "117760",
            "service_id": "000001", "stop_sequence": 0,
            "stop_id_dep": "87214080", "stop_name_dep": "Colmar",
            "dep_minutes": 17 * 60 + 30,  # pointe soir
            "stop_id_arr": "87182063", "stop_name_arr": "Mulhouse",
            "arr_minutes": 18 * 60 + 23, "duration_min": 53,
            "service_date": date(2024, 10, 21),  # vacances Toussaint zone B
        },
    ])


@pytest.fixture
def fitted_pipeline_and_scorer(synthetic_training_data, tmp_path, monkeypatch):
    """
    Fixture utilitaire : pipeline fitté + scorer entraîné, prêts à l'emploi.
    Utilisé par plusieurs tests pour éviter la duplication d'entraînement.
    """
    from shared import config as cfg
    monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

    pipeline = FeaturePipeline([
        TemporalFeatureTransformer(),
        HistoricalFeatureTransformer(),
    ])
    df_train = pipeline.fit_transform(synthetic_training_data)
    pipeline.save("feature_pipeline.joblib")

    scorer = LGBMScorer(n_estimators=10, num_leaves=8)  # rapide pour CI
    scorer.train(df_train, target_col="fraud_score")
    scorer.save("lgbm_scorer.joblib")

    return pipeline, scorer


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeaturePipelineFit:
    """Tests du FeaturePipeline seul."""

    def test_fit_transform_ajoute_features_temporelles(
        self, synthetic_training_data, tmp_path, monkeypatch
    ):
        """
        Après fit_transform(), le DataFrame doit contenir toutes les
        features temporelles produites par TemporalFeatureTransformer.
        """
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        pipeline = FeaturePipeline([TemporalFeatureTransformer()])
        result   = pipeline.fit_transform(synthetic_training_data)

        features_attendues = ["dep_hour", "is_weekend", "is_vacances",
                              "is_peak_hour", "day_of_week"]
        for feature in features_attendues:
            assert feature in result.columns, f"Feature manquante : {feature}"

    def test_fit_transform_preserve_nb_lignes(
        self, synthetic_training_data, tmp_path, monkeypatch
    ):
        """fit_transform() ne doit pas supprimer de lignes."""
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        pipeline = FeaturePipeline([TemporalFeatureTransformer()])
        result   = pipeline.fit_transform(synthetic_training_data)

        assert len(result) == len(synthetic_training_data)

    def test_transform_sans_fit_leve_erreur(self, synthetic_inference_data):
        """transform() sans fit() préalable doit lever RuntimeError."""
        pipeline = FeaturePipeline([TemporalFeatureTransformer()])
        with pytest.raises(RuntimeError, match="not.*fitted|n'a pas"):
            pipeline.transform(synthetic_inference_data)


class TestMLPipelineChaineTrain:
    """
    Tests de la chaîne complète Train → Save → Load → Predict.
    C'est le test critique de la phase d'intégration.
    """

    def test_chaine_complete_train_save_load_predict(
        self,
        synthetic_training_data,
        synthetic_inference_data,
        tmp_path,
        monkeypatch,
    ):
        """
        Chaîne complète :
            Entraînement → sauvegarde .joblib → rechargement → prédiction.

        Ce test valide que le déploiement post-entraînement fonctionne :
        les artefacts sauvegardés par train.py peuvent être utilisés par l'API.
        """
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        # ── Phase ENTRAÎNEMENT ────────────────────────────────────────────────
        pipeline = FeaturePipeline([
            TemporalFeatureTransformer(),
            HistoricalFeatureTransformer(),
        ])
        df_train_features = pipeline.fit_transform(synthetic_training_data)
        pipeline.save("feature_pipeline.joblib")

        scorer = LGBMScorer(n_estimators=10, num_leaves=8)
        scorer.train(df_train_features, target_col="fraud_score")
        scorer.save("lgbm_scorer.joblib")

        # Vérification de la présence des fichiers .joblib
        assert (tmp_path / "feature_pipeline.joblib").exists(), \
            "feature_pipeline.joblib doit exister après save()"
        assert (tmp_path / "lgbm_scorer.joblib").exists(), \
            "lgbm_scorer.joblib doit exister après save()"

        # ── Phase RECHARGEMENT (simule le démarrage de l'API) ─────────────────
        pipeline_loaded = FeaturePipeline.load("feature_pipeline.joblib")
        scorer_loaded   = LGBMScorer.load("lgbm_scorer.joblib")

        assert pipeline_loaded is not None, "Pipeline rechargé ne doit pas être None"
        assert scorer_loaded is not None,   "Scorer rechargé ne doit pas être None"

        # ── Phase PRÉDICTION ──────────────────────────────────────────────────
        df_inference_features = pipeline_loaded.transform(synthetic_inference_data)
        predictions           = scorer_loaded.predict(df_inference_features)

        # Assertions de sanité
        assert len(predictions) == len(synthetic_inference_data), \
            "1 prédiction attendue par tronçon d'entrée"
        assert not predictions.isna().any(), \
            "Aucun NaN dans les prédictions"
        assert isinstance(predictions, pd.Series), \
            "scorer.predict() doit retourner une pd.Series"

    def test_predictions_dans_range_raisonnable(
        self, fitted_pipeline_and_scorer, synthetic_inference_data, tmp_path, monkeypatch
    ):
        """
        Les prédictions LightGBM peuvent légèrement dépasser [0, 1]
        (régression non bornée). Le predict.py les clamp — on vérifie
        ici que les valeurs brutes restent dans un range raisonnable.
        """
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        pipeline, scorer = fitted_pipeline_and_scorer
        df_inf_feat = pipeline.transform(synthetic_inference_data)
        predictions = scorer.predict(df_inf_feat)

        # Les prédictions brutes ne doivent pas être aberrantes
        assert predictions.min() > -0.5, "Prédictions trop négatives"
        assert predictions.max() < 1.5,  "Prédictions trop élevées"

    def test_predict_od_jamais_vue_retourne_score(
        self, fitted_pipeline_and_scorer, tmp_path, monkeypatch
    ):
        """
        Une paire O/D inconnue à l'entraînement doit utiliser le fallback
        de HistoricalFeatureTransformer (valeur moyenne globale).
        Ne doit PAS lever d'exception.
        """
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        pipeline, scorer = fitted_pipeline_and_scorer

        df_inconnu = pd.DataFrame([{
            "trip_id": "TRIP_INCONNU", "train_number": "999",
            "service_id": "000001", "stop_sequence": 0,
            "stop_id_dep": "99999999",  # jamais vue
            "stop_name_dep": "Nulle Part",
            "dep_minutes": 480,
            "stop_id_arr": "88888888",  # jamais vue
            "stop_name_arr": "Ailleurs",
            "arr_minutes": 510, "duration_min": 30,
            "service_date": date(2024, 9, 2),
            "fraud_score": 0.0,  # colonne cible présente (sera ignorée)
        }])

        df_transformed  = pipeline.transform(df_inconnu)
        predictions     = scorer.predict(df_transformed)

        assert len(predictions) == 1
        assert not predictions.isna().any(), \
            "Paire O/D inconnue ne doit pas produire NaN"

    def test_scorer_sans_entrainement_leve_erreur(self, tmp_path, monkeypatch):
        """LGBMScorer.save() sans train() doit lever RuntimeError."""
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        scorer = LGBMScorer()
        with pytest.raises(RuntimeError):
            scorer.save()

    def test_scorer_fichier_absent_leve_file_not_found(self, tmp_path, monkeypatch):
        """LGBMScorer.load() sur un fichier absent doit lever FileNotFoundError."""
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        with pytest.raises(FileNotFoundError):
            LGBMScorer.load("inexistant.joblib")

    def test_pipeline_fichier_absent_leve_file_not_found(self, tmp_path, monkeypatch):
        """FeaturePipeline.load() sur un fichier absent doit lever FileNotFoundError."""
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="Fichier introuvable"):
            FeaturePipeline.load("inexistant.joblib")

    def test_pipeline_rechargé_produit_meme_features(
        self,
        synthetic_training_data,
        synthetic_inference_data,
        tmp_path,
        monkeypatch,
    ):
        """
        Un pipeline rechargé doit produire exactement les mêmes colonnes
        que le pipeline original (reproductibilité des features).
        """
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "MODELS_DIR", tmp_path)

        pipeline = FeaturePipeline([
            TemporalFeatureTransformer(),
            HistoricalFeatureTransformer(),
        ])
        pipeline.fit_transform(synthetic_training_data)
        pipeline.save()

        pipeline_loaded = FeaturePipeline.load()

        result_original = pipeline.transform(synthetic_inference_data)
        result_loaded   = pipeline_loaded.transform(synthetic_inference_data)

        assert set(result_original.columns) == set(result_loaded.columns), \
            "Pipeline rechargé doit produire les mêmes colonnes"
        assert len(result_original) == len(result_loaded)