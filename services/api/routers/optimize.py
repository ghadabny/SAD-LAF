import random
from datetime import date, datetime

from fastapi import APIRouter, HTTPException

from services.ml_engine.data.gtfs.loader import GTFSLoader
from services.ml_engine.data.gtfs.preprocessor import GTFSPreprocessor
from services.or_engine.graph.builder import TimeExpandedGraphBuilder
from services.or_engine.graph.transition import ArcType
from shared.schemas import (
    ArcSchema, ArcTypeSchema, NodeSchema,
    OptimizeResponseSchema, TourneeRequestSchema, TourneeSchema,
)

router = APIRouter(prefix="/optimize", tags=["Optimize"])


@router.post(
    "",
    response_model=OptimizeResponseSchema,
    summary="Générer une tournée d'inspection optimisée",
)
def optimize(request: TourneeRequestSchema) -> OptimizeResponseSchema:
    """
    Génère la meilleure tournée possible pour une mission d'inspection LAF.

    La tournée maximise la somme des scores de fraude sous contrainte
    de durée maximale de mission.

    **État actuel (POC sans données LAF) :**
    Les scores sont générés aléatoirement. L'optimiseur sélectionne
    les arcs de façon gloutonne (greedy) — prend toujours l'arc TRAIN
    avec le meilleur score disponible depuis la position courante.
    Le solveur MILP PuLP remplacera cette logique quand il sera implémenté.

    **Corps de la requête :**
    ```json
    {
        "gare_depart_id": "87212027",
        "heure_depart_min": 503,
        "duree_max_minutes": 360,
        "service_date": "2024-09-02"
    }
    ```
    """
    # ── Chargement des données GTFS ───────────────────────────────────────────
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
            detail=f"Données GTFS non disponibles : {e}.",
        )

    # ── Filtrer les trips du jour ─────────────────────────────────────────────
    date_int = int(request.service_date.strftime("%Y%m%d"))
    services_du_jour = calendar_dates[
        (calendar_dates["date"] == date_int) &
        (calendar_dates["exception_type"] == 1)
    ]["service_id"].tolist()

    if not services_du_jour:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun service trouvé pour la date {request.service_date}.",
        )

    trips_du_jour = trips[trips["service_id"].isin(services_du_jour)]

    # ── Construire les tronçons et le graphe ──────────────────────────────────
    preprocessor = GTFSPreprocessor()
    troncons_df  = preprocessor.build_troncons(
        stop_times, trips_du_jour, stops, routes, ter_only=True
    )

    if troncons_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Aucun tronçon TER trouvé pour la date {request.service_date}.",
        )

    builder = TimeExpandedGraphBuilder()
    graph   = builder.build(troncons_df, request.service_date)

    # ── Injecter les scores (mock aléatoires pour le POC) ────────────────────
    # TODO : remplacer par LGBMModel.predict() quand les données LAF arrivent
    random.seed(int(request.service_date.strftime("%Y%m%d")))
    for node, arcs in graph.items():
        for arc in arcs:
            if arc.arc_type == ArcType.TRAIN:
                arc.fraud_score = round(random.uniform(0.0, 1.0), 4)

    # ── Optimisation greedy ───────────────────────────────────────────────────
    # TODO : remplacer par OrienteeringOptimizer (MILP PuLP)
    #        quand orienteering.py sera implémenté.
    #
    # Logique greedy :
    #   Depuis la position courante, prend toujours l'arc TRAIN
    #   avec le meilleur score disponible dans la limite de temps restant.
    #   Si une correspondance est nécessaire, on la prend automatiquement.

    # Trouver le nœud de départ
    noeuds_depart = [
        n for n in graph
        if n.stop_id == request.gare_depart_id
        and n.time_minutes >= request.heure_depart_min
    ]

    if not noeuds_depart:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Aucun train au départ de la gare {request.gare_depart_id} "
                f"après {request.heure_depart_min // 60:02d}:{request.heure_depart_min % 60:02d}. "
                "Vérifiez le code UIC de la gare et l'heure de départ."
            ),
        )

    # Nœud de départ = le plus tôt après l'heure demandée
    noeud_courant = min(noeuds_depart, key=lambda n: n.time_minutes)
    temps_ecoule  = 0
    arcs_tournee  = []
    score_total   = 0.0

    while temps_ecoule < request.duree_max_minutes:
        arcs_disponibles = builder.get_reachable_arcs(graph, noeud_courant)

        if not arcs_disponibles:
            break

        # Cherche d'abord un arc TRAIN avec le meilleur score
        arcs_train = [
            a for a in arcs_disponibles
            if a.arc_type == ArcType.TRAIN
            and temps_ecoule + a.duration_min <= request.duree_max_minutes
        ]

        if arcs_train:
            # Prend le TRAIN avec le meilleur score
            meilleur_arc = max(arcs_train, key=lambda a: a.fraud_score)
            arcs_tournee.append(meilleur_arc)
            score_total  += meilleur_arc.fraud_score
            temps_ecoule += meilleur_arc.duration_min
            noeud_courant = meilleur_arc.destination
        else:
            # Pas de TRAIN disponible → cherche une correspondance
            arcs_corr = [
                a for a in arcs_disponibles
                if a.arc_type == ArcType.CORRESPONDANCE
                and temps_ecoule + a.duration_min <= request.duree_max_minutes
            ]
            if arcs_corr:
                # Prend la correspondance la plus courte
                corr = min(arcs_corr, key=lambda a: a.duration_min)
                arcs_tournee.append(corr)
                temps_ecoule += corr.duration_min
                noeud_courant = corr.destination
            else:
                break

    if not arcs_tournee:
        raise HTTPException(
            status_code=404,
            detail=(
                "Impossible de construire une tournée depuis cette gare "
                "et cette heure. Essayez une heure de départ plus tôt."
            ),
        )

    # ── Construire la réponse ─────────────────────────────────────────────────
    gare_depart  = _node_to_schema(arcs_tournee[0].source)
    gare_arrivee = _node_to_schema(arcs_tournee[-1].destination)
    nb_trains    = sum(1 for a in arcs_tournee if a.arc_type == ArcType.TRAIN)

    tournee = TourneeSchema(
        arcs=[_arc_to_schema(a) for a in arcs_tournee],
        score_total=round(score_total, 4),
        duree_totale_minutes=temps_ecoule,
        nb_trains=max(nb_trains, 1),
        gare_depart=gare_depart,
        gare_arrivee=gare_arrivee,
        service_date=request.service_date,
    )

    return OptimizeResponseSchema(
        tournee=tournee,
        request=request,
        optimized_at=datetime.now(),
    )


# ── Helpers de conversion ──────────────────────────────────────────────────────
# Convertissent les objets du domaine (Node, Arc) en schémas Pydantic
# pour la sérialisation JSON de l'API.

def _node_to_schema(node) -> NodeSchema:
    return NodeSchema(
        stop_id=node.stop_id,
        stop_name=node.stop_name,
        time_minutes=node.time_minutes,
        service_date=node.service_date,
    )


def _arc_to_schema(arc) -> ArcSchema:
    return ArcSchema(
        source=_node_to_schema(arc.source),
        destination=_node_to_schema(arc.destination),
        arc_type=ArcTypeSchema(arc.arc_type.value),
        duration_min=arc.duration_min,
        trip_id=arc.trip_id,
        train_number=arc.train_number,
        fraud_score=arc.fraud_score,
    )