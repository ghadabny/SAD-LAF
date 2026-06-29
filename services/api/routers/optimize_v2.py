# services/api/routers/optimize_v2.py
"""
Router FastAPI — POST /optimize/v2

CORRECTIONS v3.1 :
    Bug 1 — tournee_id incohérent entre JSON et CSV :
        Cause : TourneeFormatter.format() était appelé DEUX fois (une pour extraire l'ID,
                une pour l'export), chaque appel générant un UUID distinct via uuid4().
        Fix   : L'ID est généré UNE SEULE fois via generate_tournee_id() au niveau du router,
                puis injecté dans tous les objets downstream. Le formatter ne génère plus d'ID.

    Bug 2 — deux lignes dans le CSV :
        Cause : export() appelait format() en interne, et le router appelait format() aussi.
        Fix   : export() reçoit directement le DataFrame déjà formaté (export_from_df()).

    Bug 3 — tournées introuvables / pending vide après validation :
        Cause : Le singleton get_booking_store() peut être réinitialisé entre requêtes
                en mode uvicorn --reload ou si deux workers tournent.
        Fix   : Le store est initialisé au niveau module (import time) pour garantir
                une instance unique par processus. La persistance JSON assure la survie
                entre redémarrages.

    Bug 6 — double instanciation de OptimizeResponseV2Schema (FIX v3.2) :
        Cause : L'objet response_obj était créé une première fois pour être passé au
                formatter, puis un DEUXIÈME objet quasi-identique était retourné.
                C'était un reliquat du refactoring Bug 1 — l'objet intermédiaire
                n'avait plus de raison d'être distinct du retour final.
        Fix   : Un seul objet est construit. csv_path est mis à jour directement via
                model_copy() (Pydantic v2, immuable) ou via affectation directe si le
                modèle n'est pas frozen.
                Si OptimizeResponseV2Schema est frozen → on utilise model_copy().
                Sinon → affectation directe response_obj.csv_path = ...
"""
from __future__ import annotations

import logging
import uuid
from datetime import date as date_type
from datetime import datetime

from fastapi import APIRouter, HTTPException

from services.api.booking_store import (
    STATUT_EN_ATTENTE,
    STATUT_VALIDEE,
    get_booking_store,
)
from services.exporter.export import TourneeExporter
from services.exporter.formatter import TourneeFormatter
from services.or_engine.graph.transition import ArcType
from services.or_engine.solver import run as solver_run
from shared.schemas import (
    ArcSchema,
    ArcTypeSchema,
    CancelTourneeRequest,
    NodeSchema,
    OptimizeResponseV2Schema,
    RefuseTourneeRequest,
    TourneeRecordSchema,
    TourneeRequestV2Schema,
    TourneeSchema,
    ValidateTourneeRequest,
    ValidateTourneeResponse,
)

router = APIRouter(prefix="/optimize", tags=["Optimize"])
logger = logging.getLogger(__name__)

SEUIL_PERTE_WARNING_PCT: float = 15.0


# ─────────────────────────────────────────────────────────────────────────────
# Générateur d'ID centralisé
# ─────────────────────────────────────────────────────────────────────────────

def _generate_tournee_id(agent_id: str | None, service_date: date_type, optimized_at: datetime) -> str:
    """
    Génère l'identifiant unique d'une tournée.

    Appelé UNE SEULE FOIS par requête au niveau du router.
    Format : TRN_{AGENT_ID}_{YYYYMMDD}_{UID8HEX}

    Exemple : TRN_AGENT_001_20240902_3A7F1B2C
    """
    agent = (agent_id or "INCONNU").upper().replace(" ", "_")
    date_str = service_date.strftime("%Y%m%d")
    uid = str(uuid.uuid4()).split("-")[0].upper()
    return f"TRN_{agent}_{date_str}_{uid}"


# ─────────────────────────────────────────────────────────────────────────────
# POST /optimize/v2 — Génération
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/v2",
    response_model=OptimizeResponseV2Schema,
    summary="Générer une tournée (en attente de validation N+1)",
)
def optimize_v2(request: TourneeRequestV2Schema) -> OptimizeResponseV2Schema:
    """
    Génère la meilleure tournée possible et l'enregistre EN_ATTENTE_VALIDATION.

    Le tournee_id est généré UNE SEULE FOIS ici et propagé à :
      - la réponse JSON
      - le fichier CSV exporté
      - le TripBookingStore

    Codes d'erreur :
    - 422 : paramètres invalides (Pydantic)
    - 503 : GTFS non disponible (fichiers manquants)
    - 404 : tournée impossible (aucun train dans la fenêtre)
    - 500 : erreur interne inattendue
    """
    logger.info(
        "[optimize_v2] Requête — agent=%s | départ=%s | PS=%dh%02d | FS=%dh%02d | mode=%s",
        request.agent_id or "?",
        request.gare_depart_id,
        request.heure_ps_min // 60, request.heure_ps_min % 60,
        request.heure_fs_min // 60, request.heure_fs_min % 60,
        request.mode,
    )

    warning_messages: list[str] = []
    gare_arrivee = request.gare_arrivee_effective

    # Anti-doublons : récupère les trips déjà utilisés par d'autres agents ce jour
    excluded_trip_ids: set | None = None
    if request.agent_id:
        booked = get_booking_store().get_all_for_date(request.service_date)
        excluded_trip_ids = {tid for tid, ag in booked.items() if ag != request.agent_id} or None

    score_libre = None

    # ── Résolution de référence (calcul score_perte_pct) ─────────────────────
    # On tourne d'abord SANS contrainte de retour pour obtenir le score maximal
    # théorique. Cela permet de mesurer la "perte" imposée par la contrainte.
    if gare_arrivee is not None:
        try:
            result_libre = _run_solver(request, gare_arrivee_id=None, excluded_trip_ids=excluded_trip_ids)
            score_libre  = result_libre["score_total"]
        except (ValueError, FileNotFoundError):
            # Si la résolution libre échoue, on continue sans score de référence.
            # Le score_perte_pct sera 0.0 — ce n'est pas bloquant.
            pass

    # ── Résolution avec contrainte de retour ─────────────────────────────────
    try:
        result = _run_solver(request, gare_arrivee_id=gare_arrivee, excluded_trip_ids=excluded_trip_ids)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"GTFS non disponibles : {e}")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("[optimize_v2] Erreur inattendue : %s", e)
        raise HTTPException(status_code=500, detail=f"Erreur interne : {e}")

    nb_trains_reels = sum(1 for a in result["arcs"] if a.arc_type == ArcType.TRAIN)
    if nb_trains_reels == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                "Aucune tournée aller-retour possible depuis cette gare dans cette fenêtre. "
                "Élargissez la plage PS/FS ou utilisez le mode 'decouche'."
            )
        )

    # ── Calcul score_perte_pct ────────────────────────────────────────────────
    score_perte_pct = 0.0
    if score_libre is not None and score_libre > 0:
        score_perte_pct = max(
            0.0,
            (score_libre - result["score_total"]) / score_libre * 100,
        )
        if score_perte_pct >= SEUIL_PERTE_WARNING_PCT:
            logger.info(
                "[optimize_v2] Score réduit de %.1f%% par la contrainte de retour "
                "(usage interne — non exposé à l'agent).",
                score_perte_pct,
            )

    # ── Horodatage et ID — générés UNE SEULE FOIS ────────────────────────────
    now        = datetime.now()
    tournee_id = _generate_tournee_id(request.agent_id, request.service_date, now)

    # ── Construction des objets domaine ──────────────────────────────────────
    arcs_tournee        = result["arcs"]
    gare_depart_schema  = _node_to_schema(arcs_tournee[0].source)
    gare_arrivee_schema = _node_to_schema(arcs_tournee[-1].destination)

    tournee = TourneeSchema(
        arcs=[_arc_to_schema(a) for a in arcs_tournee],
        score_total=result["score_total"],
        duree_totale_minutes=result["duree_minutes"],
        nb_trains=max(result["nb_trains"], 1),
        gare_depart=gare_depart_schema,
        gare_arrivee=gare_arrivee_schema,
        service_date=request.service_date,
        generated_at=now,
    )

    # ── FIX B6 — Construction UNIQUE de la réponse ────────────────────────────
    # AVANT : deux instanciations distinctes de OptimizeResponseV2Schema
    #   (une pour le formatter, une pour le return) → reliquat du Bug 1.
    # APRÈS : un seul objet, csv_path mis à jour après l'export.
    #
    # Note Pydantic v2 :
    #   Si OptimizeResponseV2Schema est configuré avec model_config frozen=True,
    #   l'affectation directe est interdite → utiliser model_copy(update={...}).
    #   Si le modèle n'est PAS frozen (cas par défaut), l'affectation directe
    #   response_obj.csv_path = ... fonctionne.
    #   Les deux cas sont gérés ci-dessous via le bloc try/except AttributeError.
    response_obj = OptimizeResponseV2Schema(
        tournee=tournee,
        request=request,
        tournee_id=tournee_id,
        statut=STATUT_VALIDEE,
        score_perte_pct=round(score_perte_pct, 2),
        trains_en_conflit=[],
        csv_path=None,                      # sera mis à jour après l'export
        warning_messages=warning_messages,
        optimized_at=now,
    )

    # ── Export CSV ────────────────────────────────────────────────────────────
    try:
        exporter  = TourneeExporter()
        formatter = TourneeFormatter()
        # format_with_id reçoit l'objet ET l'ID déjà fixé → pas de double UUID
        df        = formatter.format_with_id(response_obj, tournee_id)
        csv_path  = exporter.export_from_df(df, response_obj, tournee_id)
        exporter.export_json_flat(df, response_obj, tournee_id)

        # Mise à jour de csv_path sur l'objet unique
        try:
            # Cas 1 : modèle non frozen → affectation directe
            response_obj.csv_path = csv_path.name
        except (AttributeError, TypeError):
            # Cas 2 : modèle frozen (Pydantic v2 model_config frozen=True)
            response_obj = response_obj.model_copy(update={"csv_path": csv_path.name})

        logger.info("[optimize_v2] CSV exporté : %s", csv_path.name)

    except Exception as e:
        # L'export CSV est non-bloquant : on log un warning et on continue.
        # L'agent reçoit sa tournée même si le fichier n'a pas pu être écrit.
        warning_messages.append(f"Export CSV indisponible : {e}")
        logger.warning("[optimize_v2] Export CSV échoué : %s", e)

    # ── Enregistrement EN_ATTENTE dans le store ───────────────────────────────
    # Seulement si un agent_id est fourni (tournée nominative).
    # Sans agent_id : tournée anonyme, pas de tracking dans le BookingStore.
    trip_ids = _extract_trip_ids(result["arcs"])
    if request.agent_id:
        store = get_booking_store()
        store.register_tournee(
            tournee_id=tournee_id,
            agent_id=request.agent_id,
            service_date=request.service_date,
            trip_ids=trip_ids,
            auto_validate=True,
        )
        logger.info(
            "[optimize_v2] Tournée %s VALIDEE automatiquement pour %s (%d trains).",
            tournee_id, request.agent_id, len(trip_ids),
        )

    return response_obj


# ─────────────────────────────────────────────────────────────────────────────
# POST /optimize/v2/validate
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/v2/validate",
    response_model=ValidateTourneeResponse,
    summary="Valider une tournée (N+1) — booking effectif des trains",
)
def validate_tournee(request: ValidateTourneeRequest) -> ValidateTourneeResponse:
    """
    Valide une tournée EN_ATTENTE_VALIDATION et réserve les trains dans le store.

    Codes d'erreur :
    - 404 : tournee_id inconnu dans le store
    - 409 : conflit — un ou plusieurs trains déjà réservés par un autre agent
    - 422 : tournée dans un statut incompatible (déjà validée, refusée, annulée)
    """
    store = get_booking_store()

    try:
        record = store.validate_tournee(
            tournee_id=request.tournee_id,
            validated_by=request.validated_by,
            max_agents_per_train=request.max_agents_per_train,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        msg = str(e)
        # Conflit de réservation → 409 Conflict (sémantique REST correcte)
        # Statut incompatible   → 422 Unprocessable Entity
        if "Conflit" in msg:
            raise HTTPException(status_code=409, detail=msg)
        raise HTTPException(status_code=422, detail=msg)

    logger.info(
        "[validate_tournee] ✅ %s validée par %s",
        request.tournee_id, request.validated_by,
    )

    return ValidateTourneeResponse(
        tournee_id=record["tournee_id"],
        statut=record["statut"],
        message=(
            f"Tournée {record['tournee_id']} validée par {request.validated_by}. "
            f"{len(record['trip_ids'])} train(s) réservé(s) pour {record['agent_id']}."
        ),
        record=TourneeRecordSchema(**record),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /optimize/v2/refuse
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/v2/refuse",
    response_model=ValidateTourneeResponse,
    summary="Refuser une tournée (N+1)",
)
def refuse_tournee(request: RefuseTourneeRequest) -> ValidateTourneeResponse:
    """
    Refuse une tournée EN_ATTENTE_VALIDATION.

    Les trains restent libres — l'agent doit régénérer une nouvelle tournée.

    Codes d'erreur :
    - 404 : tournee_id inconnu
    - 422 : tournée dans un statut incompatible
    """
    store = get_booking_store()

    try:
        record = store.refuse_tournee(
            tournee_id=request.tournee_id,
            refused_by=request.refused_by,
            motif=request.motif,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    logger.info(
        "[refuse_tournee] ❌ %s refusée par %s",
        request.tournee_id, request.refused_by,
    )

    return ValidateTourneeResponse(
        tournee_id=record["tournee_id"],
        statut=record["statut"],
        message=(
            f"Tournée {record['tournee_id']} refusée par {request.refused_by}. "
            f"Les trains restent libres. L'agent doit régénérer une tournée."
        ),
        record=TourneeRecordSchema(**record),
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /optimize/v2/cancel
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/v2/cancel",
    response_model=ValidateTourneeResponse,
    summary="Annuler une tournée validée",
)
def cancel_tournee(request: CancelTourneeRequest) -> ValidateTourneeResponse:
    """
    Annule une tournée VALIDEE et libère les trains dans le store.

    Seules les tournées en statut VALIDEE peuvent être annulées.
    (On ne peut pas annuler une tournée EN_ATTENTE — il faut la refuser.)

    Codes d'erreur :
    - 404 : tournee_id inconnu
    - 422 : tournée dans un statut incompatible (ex: déjà annulée ou EN_ATTENTE)
    """
    store = get_booking_store()

    try:
        record = store.cancel_tournee(
            tournee_id=request.tournee_id,
            cancelled_by=request.cancelled_by,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    logger.info(
        "[cancel_tournee] 🔄 %s annulée par %s",
        request.tournee_id, request.cancelled_by,
    )

    return ValidateTourneeResponse(
        tournee_id=record["tournee_id"],
        statut=record["statut"],
        message=(
            f"Tournée {record['tournee_id']} annulée. "
            f"{len(record['trip_ids'])} train(s) libéré(s)."
        ),
        record=TourneeRecordSchema(**record),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /optimize/v2/pending
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/v2/pending",
    response_model=list[TourneeRecordSchema],
    summary="Lister les tournées en attente (dashboard N+1)",
)
def get_pending_tournees(service_date: str | None = None) -> list[TourneeRecordSchema]:
    """
    Retourne toutes les tournées EN_ATTENTE_VALIDATION.

    Paramètre optionnel :
        service_date : filtre sur la date de service (format YYYY-MM-DD).
                       Si absent, retourne toutes les tournées en attente.

    Utilisé par le dashboard N+1 (manager) pour visualiser les tournées
    générées par les agents et en attente de validation.
    """
    store = get_booking_store()

    parsed_date = None
    if service_date:
        try:
            parsed_date = date_type.fromisoformat(service_date)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Format de date invalide : '{service_date}'. "
                    f"Attendu : YYYY-MM-DD (ex: 2024-09-02)."
                ),
            )

    records = store.get_tournees_en_attente(service_date=parsed_date)
    return [TourneeRecordSchema(**r) for r in records]


# ─────────────────────────────────────────────────────────────────────────────
# GET /optimize/v2/tournee/{tournee_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/v2/tournee/{tournee_id}",
    response_model=TourneeRecordSchema,
    summary="Consulter une tournée par son identifiant",
)
def get_tournee(tournee_id: str) -> TourneeRecordSchema:
    """
    Retourne le détail d'une tournée par son identifiant unique.

    Codes d'erreur :
    - 404 : tournee_id inconnu dans le store
    """
    store  = get_booking_store()
    record = store.get_tournee(tournee_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tournée '{tournee_id}' introuvable.",
        )
    return TourneeRecordSchema(**record)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers privés
# ─────────────────────────────────────────────────────────────────────────────

def _run_solver(
    request: TourneeRequestV2Schema,
    gare_arrivee_id: str | None,
    excluded_trip_ids: set | None = None,
) -> dict:
    """
    Lance le solver OR avec les paramètres de la requête.

    Délègue entièrement à services.or_engine.solver.run().
    Séparé ici pour pouvoir être mocké facilement dans les tests.
    """
    return solver_run(
        service_date=request.service_date,
        gare_depart_id=request.gare_depart_id,
        heure_depart_min=request.heure_depart_min,
        duree_max_minutes=request.duree_max_minutes,
        gare_arrivee_id=gare_arrivee_id,
        excluded_trip_ids=excluded_trip_ids,
    )


def _extract_trip_ids(arcs) -> list[str]:
    """Extrait les trip_id des arcs TRAIN (les correspondances n'ont pas de trip_id)."""
    return [a.trip_id for a in arcs if a.arc_type == ArcType.TRAIN and a.trip_id]


def _node_to_schema(node) -> NodeSchema:
    """Convertit un Node domaine en NodeSchema Pydantic."""
    return NodeSchema(
        stop_id=node.stop_id,
        stop_name=node.stop_name,
        time_minutes=node.time_minutes,
        service_date=node.service_date,
    )


def _arc_to_schema(arc) -> ArcSchema:
    """Convertit un Arc domaine en ArcSchema Pydantic."""
    return ArcSchema(
        source=_node_to_schema(arc.source),
        destination=_node_to_schema(arc.destination),
        arc_type=ArcTypeSchema(arc.arc_type.value),
        duration_min=arc.duration_min,
        trip_id=arc.trip_id,
        train_number=arc.train_number,
        fraud_score=arc.fraud_score,
    )