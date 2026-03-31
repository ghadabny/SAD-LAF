"""
services/api/routers/predict.py — Endpoints de scoring ML.

Deux endpoints exposés :

    GET  /predict?service_date=YYYY-MM-DD
        Charge les tronçons TER depuis le GTFS, les score via LightGBM,
        retourne la liste triée par score décroissant.
        Utilisé par Power Automate / dashboard.

    POST /predict/batch
        Reçoit une liste de tronçons bruts (TronconInput), les score en batch,
        retourne un PredictScoreItem par tronçon.
        Utilisé par l'OR Engine (via httpx) avant l'optimisation.

Pattern Singleton via lru_cache :
    _load_scorer()   → LGBMScorer chargé une seule fois (~50 MB .joblib)
    _load_pipeline() → FeaturePipeline chargé une seule fois

    lru_cache(maxsize=1) garantit que le fichier n'est lu qu'une fois,
    même si l'endpoint est appelé des milliers de fois.
    L'API redémarre après un ré-entraînement pour purger le cache.

Fallback POC :
    Si lgbm_scorer.joblib ou feature_pipeline.joblib sont absents,
    predict_fraud_scores() retourne des scores déterministes basés
    sur le hash des stop_ids — cohérents entre appels mais aléatoires.
    Permet de tester l'API sans données LAF réelles.
"""

import hashlib
import logging
import random
from datetime import date, datetime
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, HTTPException

from services.ml_engine.data.gtfs.loader import GTFSLoader
from services.ml_engine.data.gtfs.preprocessor import GTFSPreprocessor
from services.ml_engine.features.pipeline import FeaturePipeline
from services.ml_engine.models.lgbm_model import LGBMScorer
from shared.constants import MIN_BOARD_DURATION_MINUTES
from shared.schemas import (
    PredictRequest,
    PredictResponse,
    PredictResponseSchema,
    PredictScoreItem,
    TronconScoreSchema,
)

router = APIRouter(prefix="/predict", tags=["Predict"])
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des modèles — Singleton via lru_cache
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_scorer() -> LGBMScorer | None:
    """
    Charge le scorer LightGBM depuis le disque, une seule fois au démarrage.

    lru_cache(maxsize=1) sert de Singleton :
        - Première invocation → lit lgbm_scorer.joblib (~50 MB, ~1-2s)
        - Invocations suivantes → retourne l'objet en mémoire (~0ms)

    Retourne None si le fichier est absent (mode POC sans entraînement).
    Dans ce cas, predict_fraud_scores() utilisera le fallback déterministe.

    Pourquoi ne pas lever d'exception ici ?
        L'API doit démarrer même sans modèle (mode POC).
        L'absence de modèle est loggée comme warning, pas comme erreur fatale.
    """
    try:
        scorer = LGBMScorer.load()
        logger.info("[predict] ✅ LGBMScorer chargé : %d features.", len(scorer.feature_cols))
        return scorer
    except FileNotFoundError:
        logger.warning(
            "[predict] ⚠️  lgbm_scorer.joblib introuvable. "
            "Lance `python -m services.ml_engine.train` pour entraîner le modèle. "
            "Fallback déterministe activé."
        )
        return None
    except Exception as e:
        logger.error("[predict] ❌ Erreur lors du chargement du scorer : %s", e)
        return None


@lru_cache(maxsize=1)
def _load_pipeline() -> FeaturePipeline | None:
    """
    Charge le FeaturePipeline depuis le disque, une seule fois au démarrage.

    Même logique que _load_scorer() — Singleton via lru_cache.

    Le pipeline encode les statistiques historiques calculées lors
    de l'entraînement (HistoricalFeatureTransformer) et les paramètres
    de normalisation temporelle.
    """
    try:
        pipeline = FeaturePipeline.load()
        logger.info(
            "[predict] ✅ FeaturePipeline chargé : [%s].",
            " → ".join(pipeline.feature_names),
        )
        return pipeline
    except FileNotFoundError:
        logger.warning(
            "[predict] ⚠️  feature_pipeline.joblib introuvable. "
            "Fallback déterministe activé."
        )
        return None
    except Exception as e:
        logger.error("[predict] ❌ Erreur lors du chargement du pipeline : %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Logique métier de scoring
# ─────────────────────────────────────────────────────────────────────────────

def predict_fraud_scores(df: pd.DataFrame) -> pd.Series:
    """
    Orchestre la chaîne de prédiction ML : pipeline.transform() → scorer.predict().

    Séquence ML (modèles disponibles) :
        1. FeaturePipeline.transform(df)
           → Calcule les features temporelles (dep_hour, is_weekend…)
             et historiques (taux_fraude_od moyen) à partir des données brutes.
        2. LGBMScorer.predict(df_transformed)
           → Prédit le fraud_score en utilisant uniquement les colonnes
             feature_cols apprises lors de l'entraînement (ignore le reste).
        3. Clamp dans [0.0, 1.0] — LightGBM régression peut dépasser les bornes.

    Fallback déterministe (modèles absents) :
        Score basé sur le hash MD5 de (stop_id_dep + stop_id_arr).
        Déterministe → mêmes inputs = mêmes scores entre appels.
        Reproductible → scores cohérents pour le débogage.

    Paramètres :
        df : DataFrame avec colonnes minimales :
             dep_minutes (int), service_date (date),
             stop_id_dep (str), stop_id_arr (str)

    Retourne :
        pd.Series de float, un score ∈ [0.0, 1.0] par ligne.
    """
    scorer   = _load_scorer()
    pipeline = _load_pipeline()

    if scorer is None or pipeline is None:
        # ── Fallback déterministe ──────────────────────────────────────────────
        # Utilise le hash des stop_ids pour générer des scores cohérents.
        # L'objectif est de permettre à l'API de fonctionner sans données LAF,
        # tout en produisant des scores stables (pas de random.random() brut).
        scores = []
        for _, row in df.iterrows():
            key  = f"{row.get('stop_id_dep', '')}_{row.get('stop_id_arr', '')}"
            seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)
            rng  = random.Random(seed)
            scores.append(round(rng.uniform(0.0, 1.0), 4))
        return pd.Series(scores, index=df.index, dtype=float)

    # ── Chemin ML nominal ─────────────────────────────────────────────────────
    # Le pipeline transforme les données brutes en features exploitables par LightGBM.
    # Le scorer utilise uniquement les colonnes apprises lors de l'entraînement
    # (scorer.feature_cols) — les colonnes supplémentaires (trip_id, etc.) sont ignorées.
    df_transformed = pipeline.transform(df)
    raw_scores     = scorer.predict(df_transformed)

    # Clamp dans [0, 1] et arrondi à 4 décimales
    return raw_scores.clip(0.0, 1.0).round(4)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 1 : POST /predict/batch — scoring batch pour l'OR Engine
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/batch",
    response_model=PredictResponse,
    summary="Scorer un batch de tronçons via LightGBM [OR Engine]",
    description=(
        "Endpoint interne utilisé par l'OR Engine. "
        "Reçoit une liste de tronçons avec leurs attributs GTFS + service_date, "
        "et retourne un score de fraude ML par tronçon. "
        "Si le modèle est absent (POC), utilise un fallback déterministe."
    ),
)
def predict_batch(request: PredictRequest) -> PredictResponse:
    """
    Score un batch de tronçons via la chaîne ML complète.

    **Corps de la requête :**
    ```json
    {
        "troncons": [
            {
                "trip_id": "TRIP_001",
                "train_number": "117756",
                "service_id": "000001",
                "stop_sequence": 0,
                "stop_id_dep": "87212027",
                "dep_minutes": 503,
                "stop_id_arr": "87214007",
                "arr_minutes": 532,
                "duration_min": 29,
                "service_date": "2024-09-02"
            }
        ]
    }
    ```

    **Note sur service_date :**
    Chaque TronconInput doit inclure sa date de service réelle.
    Cette date est propagée depuis le GTFS (calendar_dates × service_id)
    dans le solver avant l'appel à cet endpoint.
    Elle est indispensable à TemporalFeatureTransformer pour calculer
    is_weekend, is_vacances, is_peak_hour, etc.
    """
    # Conversion de la liste de tronçons en DataFrame
    # model_dump() sérialise les types Pydantic (date → str) en dict Python natif.
    # pd.DataFrame() reconstruit les types corrects automatiquement.
    df = pd.DataFrame([t.model_dump() for t in request.troncons])

    try:
        scores_series = predict_fraud_scores(df)
    except Exception as e:
        logger.exception("[predict] Erreur lors du scoring ML du batch.")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne lors du scoring ML : {e}",
        )

    # Construction des PredictScoreItem avec la clé de jointure (trip_id, stop_id_dep, dep_minutes)
    # Cette clé permet à inject_scores() dans solver.py de retrouver l'arc correspondant.
    score_items = [
        PredictScoreItem(
            trip_id=str(row["trip_id"]),
            stop_sequence=int(row["stop_sequence"]),
            stop_id_dep=str(row["stop_id_dep"]),
            dep_minutes=int(row["dep_minutes"]),
            fraud_score=float(scores_series.iloc[i]),
        )
        for i, (_, row) in enumerate(df.iterrows())
    ]

    return PredictResponse(
        scores=score_items,
        nb_troncons=len(score_items),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint 2 : GET /predict — scoring GTFS pour Power Automate / dashboard
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=PredictResponseSchema,
    summary="Scorer les tronçons TER d'une date donnée [Dashboard / Power Automate]",
)
def predict(service_date: date) -> PredictResponseSchema:
    """
    Pour une date de service, charge les tronçons TER depuis le GTFS,
    les score via LightGBM et retourne la liste triée par score décroissant.

    Utilisé par Power Automate et le dashboard opérationnel.
    L'OR Engine utilise plutôt POST /predict/batch.

    **Paramètre :**
    - `service_date` : date au format YYYY-MM-DD (ex: 2024-09-02)

    **Retourne :**
    Liste de tronçons TER enrichis avec leur score de fraude, triés par score décroissant.
    Les tronçons de durée < MIN_BOARD_DURATION_MINUTES sont filtrés (contrôle impossible).
    """
    # ── Chargement GTFS ───────────────────────────────────────────────────────
    try:
        loader         = GTFSLoader()
        stop_times     = loader.load_stop_times()
        trips          = loader.load_trips()
        stops          = loader.load_stops()
        routes         = loader.load_routes()
        calendar_dates = loader.load_calendar_dates()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Données GTFS non disponibles : {e}. "
                "Lance d'abord GTFSDownloader().download()."
            ),
        )

    # ── Filtrage par date de service ──────────────────────────────────────────
    date_int = int(service_date.strftime("%Y%m%d"))
    services_du_jour = calendar_dates[
        (calendar_dates["date"] == date_int) &
        (calendar_dates["exception_type"] == 1)
    ]["service_id"].tolist()

    if not services_du_jour:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Aucun service trouvé pour la date {service_date}. "
                "Vérifiez que la date est dans la plage des 151 jours du GTFS."
            ),
        )

    trips_du_jour = trips[trips["service_id"].isin(services_du_jour)]

    # ── Construction des tronçons ─────────────────────────────────────────────
    preprocessor = GTFSPreprocessor()
    troncons_df  = preprocessor.build_troncons(
        stop_times, trips_du_jour, stops, routes, ter_only=True
    )

    if troncons_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun tronçon TER trouvé pour la date {service_date}.",
        )

    # Filtre les tronçons trop courts pour un contrôle physiquement possible
    troncons_df = troncons_df[
        troncons_df["duration_min"] >= MIN_BOARD_DURATION_MINUTES
    ].reset_index(drop=True)

    # ── Propagation de service_date ───────────────────────────────────────────
    # Indispensable pour TemporalFeatureTransformer (is_weekend, is_vacances…).
    # La date n'est pas dans le DataFrame GTFS brut — elle est injectée ici
    # depuis le contexte de la requête.
    # Voir : conseil de gestion service_date dans la documentation projet.
    troncons_df["service_date"] = service_date

    # ── Scoring ML ────────────────────────────────────────────────────────────
    try:
        scores = predict_fraud_scores(troncons_df)
    except Exception as e:
        logger.exception("[predict] Erreur lors du scoring ML GTFS.")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du scoring ML : {e}",
        )

    # ── Assemblage de la réponse ──────────────────────────────────────────────
    troncons_scores = [
        TronconScoreSchema(
            trip_id=str(row["trip_id"]),
            train_number=str(row["train_number"]),
            service_id=str(row["service_id"]),
            stop_sequence=int(row["stop_sequence"]),
            stop_id_dep=str(row["stop_id_dep"]),
            stop_name_dep=str(row["stop_name_dep"]),
            dep_minutes=int(row["dep_minutes"]),
            stop_id_arr=str(row["stop_id_arr"]),
            stop_name_arr=str(row["stop_name_arr"]),
            arr_minutes=int(row["arr_minutes"]),
            duration_min=int(row["duration_min"]),
            fraud_score=float(scores.iloc[i]),
        )
        for i, (_, row) in enumerate(troncons_df.iterrows())
    ]

    # Tri décroissant : les tronçons les plus risqués en premier
    troncons_scores.sort(key=lambda t: t.fraud_score, reverse=True)

    return PredictResponseSchema(
        troncons=troncons_scores,
        service_date=service_date,
        scored_at=datetime.now(),
        nb_troncons=len(troncons_scores),
    )