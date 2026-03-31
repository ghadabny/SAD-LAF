# services/api/routers/optimize.py
"""
Router FastAPI pour la génération de tournées d'inspection LAF.

SOLID — principe D (refactoring) :
    Avant ce refactoring, optimize() dupliquait intégralement la logique
    de solver.run() : GTFSLoader + GTFSPreprocessor + TimeExpandedGraphBuilder
    + greedy. Deux implémentations du même algorithme qui pouvaient diverger.

    Après refactoring, optimize() est un adaptateur HTTP pur :
        TourneeRequestSchema → solver.run() → OptimizeResponseSchema

    solver.run() est l'unique point d'entrée de l'algorithme.
    optimize.py traduit uniquement les types HTTP ↔ domaine.
"""
from datetime import datetime

from fastapi import APIRouter, HTTPException

from services.or_engine.solver import run as solver_run
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

    Délègue entièrement à solver.run() — ce router est un adaptateur HTTP pur.
    Toute la logique métier (GTFS, graphe, scoring ML, optimisation) vit dans
    services/or_engine/solver.py.

    **Corps de la requête :**
    ```json
    {
        "gare_depart_id": "87212027",
        "heure_depart_min": 480,
        "duree_max_minutes": 360,
        "service_date": "2024-09-02"
    }
    ```

    **Codes d'erreur :**
    - 503 : GTFS non disponible (lancer GTFSDownloader().download())
    - 404 : date hors calendrier, gare inconnue, ou tournée impossible
    - 500 : erreur interne inattendue
    """
    try:
        result = solver_run(
            service_date=request.service_date,
            gare_depart_id=request.gare_depart_id,
            heure_depart_min=request.heure_depart_min,
            duree_max_minutes=request.duree_max_minutes,
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Données GTFS non disponibles : {e}. "
                   "Lance d'abord GTFSDownloader().download().",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne lors de la génération de la tournée : {e}",
        )

    # ── Conversion domaine → schémas Pydantic ─────────────────────────────────
    # solver.run() retourne des objets Arc/Node du domaine.
    # L'API expose des schémas Pydantic sérialisables en JSON.
    arcs_tournee = result["arcs"]

    if not arcs_tournee:
        raise HTTPException(
            status_code=404,
            detail="Impossible de construire une tournée depuis cette gare "
                   "et cette heure. Essayez une heure de départ plus tôt.",
        )

    gare_depart  = _node_to_schema(arcs_tournee[0].source)
    gare_arrivee = _node_to_schema(arcs_tournee[-1].destination)
    nb_trains    = result["nb_trains"]

    tournee = TourneeSchema(
        arcs=[_arc_to_schema(a) for a in arcs_tournee],
        score_total=result["score_total"],
        duree_totale_minutes=result["duree_minutes"],
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


# ── Helpers de conversion domaine → schémas ───────────────────────────────────
# Responsabilité unique : traduire les objets du domaine en types Pydantic.
# Ces helpers n'ont aucune logique métier.

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