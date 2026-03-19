import random
from datetime import date, datetime

from fastapi import APIRouter, HTTPException

from services.ml_engine.data.gtfs.loader import GTFSLoader
from services.ml_engine.data.gtfs.preprocessor import GTFSPreprocessor
from shared.schemas import PredictResponseSchema, TronconScoreSchema
from shared.constants import MIN_BOARD_DURATION_MINUTES

router = APIRouter(prefix="/predict", tags=["Predict"])


@router.post(
    "",
    response_model=PredictResponseSchema,
    summary="Scorer les tronçons d'une date donnée",
)
def predict(service_date: date) -> PredictResponseSchema:
    """
    Pour une date de service donnée, retourne tous les tronçons TER
    enrichis avec un score de fraude.

    **État actuel (POC sans données LAF) :**
    Les scores sont générés aléatoirement entre 0.0 et 1.0.
    Quand le modèle LightGBM sera entraîné sur les données LAF,
    cette fonction appellera LGBMModel.predict() à la place.

    **Paramètre :**
    - `service_date` : date au format YYYY-MM-DD (ex: 2024-09-02)

    **Retourne :**
    Liste de tronçons avec leur score de fraude, triés par score décroissant.
    """
    try:
        loader = GTFSLoader()
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

    # Filtrer les trips qui circulent à la date demandée
    date_int = int(service_date.strftime("%Y%m%d"))
    services_du_jour = calendar_dates[
        (calendar_dates["date"] == date_int) &
        (calendar_dates["exception_type"] == 1)
    ]["service_id"].tolist()

    if not services_du_jour:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun service trouvé pour la date {service_date}. "
                   "Vérifiez que la date est dans la plage des 151 jours du GTFS.",
        )

    # Filtrer les trips du jour
    trips_du_jour = trips[trips["service_id"].isin(services_du_jour)]

    # Construire les tronçons
    preprocessor = GTFSPreprocessor()
    troncons_df  = preprocessor.build_troncons(
        stop_times, trips_du_jour, stops, routes, ter_only=True
    )

    if troncons_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun tronçon TER trouvé pour la date {service_date}.",
        )
    troncons_df = troncons_df[
        troncons_df["duration_min"] >= MIN_BOARD_DURATION_MINUTES
        ].reset_index(drop=True)

    # ── Scoring ───────────────────────────────────────────────────────────────
    # TODO : remplacer par LGBMModel.predict(pipeline.transform(troncons_df))
    #        quand les données LAF seront disponibles.
    random.seed(int(service_date.strftime("%Y%m%d")))  # seed reproductible

    troncons_scores = []
    for _, row in troncons_df.iterrows():
        troncons_scores.append(
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
                fraud_score=round(random.uniform(0.0, 1.0), 4),
            )
        )

    # Trier par score décroissant — les plus risqués en premier
    troncons_scores.sort(key=lambda t: t.fraud_score, reverse=True)

    return PredictResponseSchema(
        troncons=troncons_scores,
        service_date=service_date,
        scored_at=datetime.now(),
        nb_troncons=len(troncons_scores),
    )