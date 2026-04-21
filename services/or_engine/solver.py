"""
services/or_engine/solver.py — Orchestration de la génération de tournées.

Séquence complète :
    ┌─────────────────────────────────────────────────────────────────────┐
    │  GTFS  →  Fusion RT  →  Graphe temps-étendu  →  Scores ML  →  OR   │
    └─────────────────────────────────────────────────────────────────────┘

Changement v2 :
    run() accepte un paramètre optionnel gare_arrivee_id.
    Il est propagé directement à OrienteeringOptimizer.solve().
    Toute la logique de contrainte de retour vit dans l'optimiseur.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import httpx
import pandas as pd

from services.ml_engine.data.gtfs.loader import GTFSLoader
from services.ml_engine.data.gtfs.preprocessor import GTFSPreprocessor
from services.ml_engine.data.gtfs_rt.cache import GTFSRTCache
from services.ml_engine.data.gtfs_rt.merger import GTFSRTMerger
from services.or_engine.graph.builder import TimeExpandedGraphBuilder
from services.or_engine.graph.transition import Arc, ArcType, Node
from services.or_engine.optimizer.orienteering import OrienteeringOptimizer
from shared.constants import CONTROL_SATURATION_MINUTES
from shared.schemas import (
    PredictRequest,
    PredictResponse,
    PredictScoreItem,
    TronconInput,
)
from shared.config import config

logger = logging.getLogger(__name__)

DEFAULT_PREDICT_API_URL = config.PREDICT_API_URL

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 : Injection des scores dans le graphe
# ─────────────────────────────────────────────────────────────────────────────

def inject_scores(
    graph: dict[Node, list[Arc]],
    scores: list[PredictScoreItem],
) -> None:
    """
    Injecte les scores ML pondérés dans les arcs TRAIN du graphe (mutation in-place).

    Pondération par efficacité terrain :
        score_effectif = score_ml × min(1.0, duration_min / CONTROL_SATURATION_MINUTES)
    """
    if not scores:
        logger.debug("[inject_scores] Liste vide — graphe non modifié.")
        return

    score_lookup = _build_score_lookup(scores)
    n_injected, n_missed = _apply_scores_to_graph(graph, score_lookup)
    _log_injection_report(n_injected, n_missed)


def _build_score_lookup(scores: list[PredictScoreItem]) -> dict[tuple, float]:
    return {
        (s.trip_id, s.stop_id_dep, s.dep_minutes): s.fraud_score
        for s in scores
    }


def _apply_scores_to_graph(
    graph: dict[Node, list[Arc]],
    score_lookup: dict[tuple, float],
) -> tuple[int, int]:
    n_injected = 0
    n_missed   = 0
    for arcs in graph.values():
        for arc in arcs:
            if arc.arc_type != ArcType.TRAIN:
                continue
            key = (arc.trip_id or "", arc.source.stop_id, arc.source.time_minutes)
            if key in score_lookup:
                arc.fraud_score = _weighted_score(score_lookup[key], arc.duration_min)
                n_injected += 1
            else:
                n_missed += 1
    return n_injected, n_missed


def _weighted_score(raw_score: float, duration_min: int) -> float:
    efficacite = min(1.0, duration_min / CONTROL_SATURATION_MINUTES)
    return raw_score * efficacite


def _log_injection_report(n_injected: int, n_missed: int) -> None:
    if n_missed > 0:
        logger.warning(
            "[inject_scores] %d arc(s) sans score ML. %d scorés avec succès.",
            n_missed, n_injected,
        )
    else:
        logger.info("[inject_scores] ✅ %d arc(s) scorés.", n_injected)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 : Appel API ML
# ─────────────────────────────────────────────────────────────────────────────

def fetch_scores_from_api(
    troncons_df: pd.DataFrame,
    service_date: date,
    predict_url: str = DEFAULT_PREDICT_API_URL,
    timeout: float = 30.0,
) -> list[PredictScoreItem]:
    """
    Appelle POST /predict/batch et retourne les scores ML.
    Retourne une liste vide en cas d'erreur (degraded mode, pas de crash).
    """
    payload = _build_predict_request(troncons_df, service_date).model_dump(mode="json")
    logger.info("[solver] predict/batch : %d tronçons → %s", len(troncons_df), predict_url)

    try:
        response = httpx.post(predict_url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = PredictResponse.model_validate(response.json())
        logger.info("[solver] ✅ %d scores reçus.", result.nb_troncons)
        return result.scores

    except httpx.TimeoutException:
        logger.error("[solver] ⚠️ Timeout sur %s — fraud_score=0.0.", predict_url)
    except httpx.HTTPStatusError as e:
        logger.error("[solver] ⚠️ HTTP %d depuis %s.", e.response.status_code, predict_url)
    except httpx.ConnectError:
        logger.error("[solver] ⚠️ Connexion refusée sur %s.", predict_url)
    except Exception as e:
        logger.exception("[solver] ❌ Erreur inattendue : %s", e)

    return []


def _build_predict_request(troncons_df: pd.DataFrame, service_date: date) -> PredictRequest:
    return PredictRequest(troncons=[
        _row_to_troncon_input(row, service_date)
        for _, row in troncons_df.iterrows()
    ])


def _row_to_troncon_input(row: pd.Series, service_date: date) -> TronconInput:
    return TronconInput(
        trip_id=str(row["trip_id"]),
        train_number=str(row.get("train_number", "")),
        service_id=str(row.get("service_id", "")),
        stop_sequence=int(row.get("stop_sequence", 0)),
        stop_id_dep=str(row["stop_id_dep"]),
        stop_name_dep=str(row.get("stop_name_dep", "")),
        dep_minutes=int(row["dep_minutes"]),
        stop_id_arr=str(row["stop_id_arr"]),
        stop_name_arr=str(row.get("stop_name_arr", "")),
        arr_minutes=int(row.get("arr_minutes", 0)),
        duration_min=int(row["duration_min"]),
        service_date=service_date,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 : Chargement GTFS statique
# ─────────────────────────────────────────────────────────────────────────────

def _load_gtfs_troncons(service_date: date) -> pd.DataFrame:
    loader         = GTFSLoader()
    stop_times     = loader.load_stop_times()
    trips          = loader.load_trips()
    stops          = loader.load_stops()
    routes         = loader.load_routes()
    calendar_dates = loader.load_calendar_dates()

    date_int = int(service_date.strftime("%Y%m%d"))
    services_du_jour = calendar_dates[
        (calendar_dates["date"] == date_int) &
        (calendar_dates["exception_type"] == 1)
    ]["service_id"].tolist()

    if not services_du_jour:
        raise ValueError(f"[solver] Aucun service GTFS pour la date {service_date}.")

    trips_du_jour = trips[trips["service_id"].isin(services_du_jour)]
    preprocessor  = GTFSPreprocessor()
    troncons_df   = preprocessor.build_troncons(
        stop_times, trips_du_jour, stops, routes, ter_only=True
    )
    troncons_df["service_date"] = service_date
    return troncons_df


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 : Enrichissement GTFS-RT
# ─────────────────────────────────────────────────────────────────────────────

def _apply_gtfs_rt(troncons_df: pd.DataFrame) -> pd.DataFrame:
    try:
        cache = GTFSRTCache()
        stop_updates, cancelled_trips = cache.load()

        if stop_updates.empty and cancelled_trips.empty:
            logger.info("[solver] Cache RT vide — tronçons statiques utilisés.")
            return troncons_df

        merger      = GTFSRTMerger()
        troncons_rt = merger.merge(troncons_df, stop_updates, cancelled_trips)
        troncons_rt, nb_annules = _filter_cancelled(troncons_rt)
        nb_retards = _count_delayed(troncons_rt)

        logger.info(
            "[solver] GTFS-RT : %d tronçons annulés retirés | %d avec retard.",
            nb_annules, nb_retards,
        )
        return troncons_rt

    except Exception as e:
        logger.warning("[solver] ⚠️ GTFS-RT indisponible (%s) — tronçons statiques utilisés.", e)
        return troncons_df


def _filter_cancelled(troncons_rt: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "is_cancelled" not in troncons_rt.columns:
        return troncons_rt, 0
    nb_avant   = len(troncons_rt)
    filtre     = troncons_rt[troncons_rt["is_cancelled"] == False].copy()
    nb_annules = nb_avant - len(filtre)
    return filtre, nb_annules


def _count_delayed(troncons_rt: pd.DataFrame) -> int:
    if "delay_dep_sec" not in troncons_rt.columns:
        return 0
    return int((troncons_rt["delay_dep_sec"] > 0).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration principale
# ─────────────────────────────────────────────────────────────────────────────

def run(
    service_date: date,
    gare_depart_id: str,
    heure_depart_min: int,
    duree_max_minutes: int = 360,
    predict_url: str = DEFAULT_PREDICT_API_URL,
    gare_arrivee_id: Optional[str] = None,   # ← NOUVEAU v2
) -> dict:
    """
    Orchestre la génération complète d'une tournée optimisée.

    Séquence :
        1. GTFS statique
        2. Fusion GTFS-RT (retire les annulés, ajoute les retards)
        3. Graphe temps-étendu
        4. Scores ML
        5. OrienteeringOptimizer (avec contrainte de retour si gare_arrivee_id)

    Paramètres :
        gare_arrivee_id : code UIC 8 chiffres de la gare d'arrivée.
                          None = tournée libre (comportement v1).
                          Fourni = contrainte stricte de terminaison.

    Lève :
        FileNotFoundError : GTFS non disponible
        ValueError        : date hors calendrier, gare inconnue,
                            tournée impossible, contrainte retour infaisable
    """
    logger.info(
        "[solver] ══ Démarrage ══ date=%s | départ=%s | %02dh%02d | max=%dmin | retour=%s",
        service_date, gare_depart_id,
        heure_depart_min // 60, heure_depart_min % 60,
        duree_max_minutes,
        gare_arrivee_id or "libre",
    )

    troncons_df = _load_gtfs_troncons(service_date)
    logger.info("[solver] %d tronçons TER statiques chargés.", len(troncons_df))

    troncons_df = _apply_gtfs_rt(troncons_df)
    logger.info("[solver] %d tronçons après fusion RT.", len(troncons_df))

    builder = TimeExpandedGraphBuilder()
    graph   = builder.build(troncons_df, service_date)

    scores = fetch_scores_from_api(troncons_df, service_date, predict_url)
    inject_scores(graph, scores)

    optimizer = OrienteeringOptimizer(time_limit_seconds=30)
    tournee   = optimizer.solve(
        graph=graph,
        gare_depart_id=gare_depart_id,
        heure_depart_min=heure_depart_min,
        duree_max_minutes=duree_max_minutes,
        gare_arrivee_id=gare_arrivee_id,   # ← propagé
    )

    logger.info(
        "[solver] ══ Tournée ══ %d trains | score=%.4f | durée=%dmin",
        tournee["nb_trains"], tournee["score_total"], tournee["duree_minutes"],
    )
    return tournee