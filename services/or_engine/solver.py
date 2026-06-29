# services/or_engine/solver.py
"""
SOLID — DIP (v2) :
    Introduction de TourneeOrchestrator — classe qui reçoit toutes ses
    dépendances par injection plutôt que de les instancier elle-même.

Avant :
    def run(...):
        loader       = GTFSLoader()           # concret — impossible à mocker proprement
        preprocessor = GTFSPreprocessor()     # concret
        cache        = GTFSRTCache()          # concret
        merger       = GTFSRTMerger()         # concret
        builder      = TimeExpandedGraphBuilder()   # concret
        optimizer    = OrienteeringOptimizer(...)   # concret

Après :
    class TourneeOrchestrator:
        def __init__(self, loader=None, preprocessor=None, ...):
            self._loader = loader or GTFSLoader()   # injectable

    # Compatibilité backward — optimize_v2.py n'a pas à changer
    def run(...) -> dict:
        return TourneeOrchestrator().run(...)

Usage test :
    orchestrator = TourneeOrchestrator(
        loader=FakeLoader(),
        preprocessor=FakePreprocessor(),
        rt_cache=FakeCache(),
    )
    result = orchestrator.run(service_date=..., ...)
    → aucun fichier GTFS requis, aucun appel API réseau
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
    timeout: float = 120.0,
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
# TourneeOrchestrator — classe principale avec injection de dépendances (DIP)
# ─────────────────────────────────────────────────────────────────────────────

class TourneeOrchestrator:
    """
    Orchestre la génération complète d'une tournée optimisée.

    SOLID — DIP :
        Toutes les dépendances sont injectables via le constructeur.
        Les valeurs par défaut instancient les classes de production.
        En test, on injecte des doublures sans fichiers GTFS ni appels réseau.

    Usage production :
        orchestrator = TourneeOrchestrator()
        tournee = orchestrator.run(service_date=..., ...)

    Usage test :
        orchestrator = TourneeOrchestrator(
            loader=FakeGTFSLoader(),
            rt_cache=FakeRTCache(),
        )
        tournee = orchestrator.run(...)
    """

    def __init__(
        self,
        loader:        GTFSLoader              | None = None,
        preprocessor:  GTFSPreprocessor        | None = None,
        rt_cache:      GTFSRTCache             | None = None,
        rt_merger:     GTFSRTMerger            | None = None,
        graph_builder: TimeExpandedGraphBuilder | None = None,
        optimizer:     OrienteeringOptimizer    | None = None,
    ) -> None:
        """
        Paramètres (tous optionnels) :
            loader        : lit les fichiers GTFS .txt depuis le disque
            preprocessor  : construit les tronçons depuis les tables GTFS
            rt_cache      : lit le cache Parquet GTFS-RT
            rt_merger     : joint les tronçons statiques avec les données RT
            graph_builder : construit le graphe temps-étendu
            optimizer     : résout le problème d'orienteering (MILP + fallback greedy)
        """
        self._loader        = loader        or GTFSLoader()
        self._preprocessor  = preprocessor  or GTFSPreprocessor()
        self._rt_cache      = rt_cache      or GTFSRTCache()
        self._rt_merger     = rt_merger     or GTFSRTMerger()
        self._graph_builder = graph_builder or TimeExpandedGraphBuilder()
        self._optimizer     = optimizer     or OrienteeringOptimizer(time_limit_seconds=30)

    # ── Interface publique ────────────────────────────────────────────────────

    def run(
        self,
        service_date:     date,
        gare_depart_id:   str,
        heure_depart_min: int,
        duree_max_minutes: int = 360,
        predict_url:      str = DEFAULT_PREDICT_API_URL,
        gare_arrivee_id:  Optional[str] = None,
        excluded_trip_ids: Optional[set] = None,
    ) -> dict:
        """
        Orchestre la génération complète d'une tournée optimisée.

        Séquence :
            1. GTFS statique     → tronçons du jour
            2. GTFS-RT           → retire les annulés, ajoute les retards
            3. Graphe            → temps-étendu (Node/Arc)
            4. Scores ML         → API predict/batch
            5. Optimisation OR   → OrienteeringOptimizer (MILP + greedy fallback)

        Paramètres :
            gare_arrivee_id : code UIC 8 chiffres de la gare d'arrivée.
                              None = tournée libre.
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

        troncons_df = self._load_gtfs_troncons(service_date)
        logger.info("[solver] %d tronçons TER statiques chargés.", len(troncons_df))

        troncons_df = self._apply_gtfs_rt(troncons_df)
        logger.info("[solver] %d tronçons après fusion RT.", len(troncons_df))

        graph  = self._graph_builder.build(troncons_df, service_date)
        scores = fetch_scores_from_api(troncons_df, service_date, predict_url)
        inject_scores(graph, scores)

        tournee = self._optimizer.solve(
            graph=graph,
            gare_depart_id=gare_depart_id,
            heure_depart_min=heure_depart_min,
            duree_max_minutes=duree_max_minutes,
            gare_arrivee_id=gare_arrivee_id,
            excluded_trip_ids=excluded_trip_ids,
        )

        logger.info(
            "[solver] ══ Tournée ══ %d trains | score=%.4f | durée=%dmin",
            tournee["nb_trains"], tournee["score_total"], tournee["duree_minutes"],
        )
        return tournee

    # ── Méthodes privées ─────────────────────────────────────────────────────

    def _load_gtfs_troncons(self, service_date: date) -> pd.DataFrame:
        """
        Charge les tronçons GTFS statiques filtrés sur la date de service.

        Utilise self._loader et self._preprocessor (injectables).
        """
        stop_times     = self._loader.load_stop_times()
        trips          = self._loader.load_trips()
        stops          = self._loader.load_stops()
        routes         = self._loader.load_routes()
        calendar_dates = self._loader.load_calendar_dates()

        date_int = int(service_date.strftime("%Y%m%d"))
        services_du_jour = calendar_dates[
            (calendar_dates["date"] == date_int) &
            (calendar_dates["exception_type"] == 1)
        ]["service_id"].tolist()

        if not services_du_jour:
            raise ValueError(f"[solver] Aucun service GTFS pour la date {service_date}.")

        trips_du_jour = trips[trips["service_id"].isin(services_du_jour)]
        troncons_df   = self._preprocessor.build_troncons(
            stop_times, trips_du_jour, stops, routes, ter_only=True
        )
        troncons_df["service_date"] = service_date
        return troncons_df

    def _apply_gtfs_rt(self, troncons_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrichit les tronçons avec les données temps réel (retards, annulations).

        Utilise self._rt_cache et self._rt_merger (injectables).
        Fallback silencieux si le cache est vide ou corrompu.
        """
        try:
            stop_updates, cancelled_trips = self._rt_cache.load()

            if stop_updates.empty and cancelled_trips.empty:
                logger.info("[solver] Cache RT vide — tronçons statiques utilisés.")
                return troncons_df

            troncons_rt = self._rt_merger.merge(troncons_df, stop_updates, cancelled_trips)
            troncons_rt, nb_annules = _filter_cancelled(troncons_rt)
            nb_retards = _count_delayed(troncons_rt)

            logger.info(
                "[solver] GTFS-RT : %d tronçons annulés retirés | %d avec retard.",
                nb_annules, nb_retards,
            )
            return troncons_rt

        except Exception as e:
            logger.warning(
                "[solver] ⚠️ GTFS-RT indisponible (%s) — tronçons statiques utilisés.", e
            )
            return troncons_df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers module-level (inchangés)
# ─────────────────────────────────────────────────────────────────────────────

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
# Shim de compatibilité backward
# ─────────────────────────────────────────────────────────────────────────────
# optimize_v2.py appelle solver.run(...) directement.
# Ce shim garantit qu'aucun fichier appelant ne change.
# Il crée un TourneeOrchestrator avec les dépendances par défaut.
# ─────────────────────────────────────────────────────────────────────────────

def run(
    service_date:      date,
    gare_depart_id:    str,
    heure_depart_min:  int,
    duree_max_minutes: int = 360,
    predict_url:       str = DEFAULT_PREDICT_API_URL,
    gare_arrivee_id:   Optional[str] = None,
    excluded_trip_ids: Optional[set] = None,
) -> dict:
    """
    Shim de compatibilité — délègue à TourneeOrchestrator().run().

    Tous les appelants existants (optimize_v2.py, tests, etc.) continuent
    d'appeler solver.run() sans modification.
    Pour injecter des dépendances (tests unitaires), utiliser directement :
        TourneeOrchestrator(loader=..., ...).run(...)
    """
    return TourneeOrchestrator().run(
        service_date=service_date,
        gare_depart_id=gare_depart_id,
        heure_depart_min=heure_depart_min,
        duree_max_minutes=duree_max_minutes,
        predict_url=predict_url,
        gare_arrivee_id=gare_arrivee_id,
        excluded_trip_ids=excluded_trip_ids,
    )