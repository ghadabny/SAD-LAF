"""
services/or_engine/solver.py — Orchestration de la génération de tournées.

Séquence complète :
    ┌─────────────────────────────────────────────────────────────────────┐
    │  GTFS                                                               │
    │  (GTFSLoader + GTFSPreprocessor)                                    │
    │       ↓                                                             │
    │  Graphe temps-étendu                                                │
    │  (TimeExpandedGraphBuilder)                                         │
    │       ↓                                                             │
    │  Scores ML                                                          │
    │  (POST /predict/batch via httpx)                                    │
    │       ↓                                                             │
    │  Injection dans le graphe                                           │
    │  (inject_scores — mutation in-place des Arc.fraud_score)            │
    │       ↓                                                             │
    │  Optimisation                                                       │
    │  (OrienteeringOptimizer MILP — greedy en attendant PuLP)            │
    └─────────────────────────────────────────────────────────────────────┘

Principes SOLID :
    S — chaque fonction a une responsabilité unique et documentée
    O — inject_scores est ouvert à d'autres scorers (rule-based, etc.)
    L — le solver ne dépend pas du type concret d'optimiseur
    I — interfaces séparées pour scoring et optimisation
    D — dépendances via injection (predict_url configurable pour les tests)

Gestion service_date :
    La date n'est pas dans les tronçons bruts GTFS.
    Elle est propagée du contexte (requête d'optimisation) vers les TronconInput
    dans _build_predict_request(). Voir docstring de cette fonction.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING

import httpx
import pandas as pd

from services.ml_engine.data.gtfs.loader import GTFSLoader
from services.ml_engine.data.gtfs.preprocessor import GTFSPreprocessor
from services.or_engine.graph.builder import TimeExpandedGraphBuilder
from services.or_engine.graph.transition import Arc, ArcType, Node
from shared.schemas import (
    PredictRequest,
    PredictResponse,
    PredictScoreItem,
    TronconInput,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PREDICT_API_URL = "http://api:8000/predict/batch"
"""
URL de l'API predict dans l'environnement Docker Compose.
Le service 'api' est le nom du container FastAPI dans docker-compose.yml.
Configurable via le paramètre predict_url des fonctions pour les tests.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 : Injection des scores dans le graphe
# ─────────────────────────────────────────────────────────────────────────────

def inject_scores(
    graph: dict[Node, list[Arc]],
    scores: list[PredictScoreItem],
) -> None:
    """
    Injecte les scores ML dans les arcs TRAIN du graphe (mutation in-place).

    Seuls les arcs ArcType.TRAIN sont scorés :
        - Un arc TRAIN représente un trajet en train où un agent peut contrôler.
        - Un arc CORRESPONDANCE représente une attente en gare, sans voyageurs
          à contrôler → fraud_score non pertinent, reste à 0.0.

    Clé de correspondance arc ↔ score :
        (trip_id, stop_id_dep, dep_minutes)
            trip_id       → identifie le train
            stop_id_dep   → identifie le tronçon dans le trip (gare de départ)
            dep_minutes   → disambiguïsation si un même trip passe deux fois
                            en gare (rare mais possible sur certains TER)

        Pour un arc : (arc.trip_id, arc.source.stop_id, arc.source.time_minutes)
        Pour un score : (score.trip_id, score.stop_id_dep, score.dep_minutes)

    Complexité :
        O(|scores|) pour construire le lookup dict
        O(|arcs|)   pour itérer sur le graphe
        Total : O(|scores| + |arcs|) — linéaire, pas quadratique

    Paramètres :
        graph  : graphe temps-étendu retourné par TimeExpandedGraphBuilder.build()
                 → muté in-place (les Arc.fraud_score sont modifiés directement)
        scores : liste de PredictScoreItem retournés par POST /predict/batch

    Retourne :
        None — la fonction est une procédure (mutation in-place, pas de copie)

    Effets de bord :
        Arc.fraud_score mis à jour pour les arcs TRAIN matchés.
        Les arcs non matchés conservent fraud_score=0.0 (valeur par défaut).
    """
    if not scores:
        logger.debug("[inject_scores] Liste de scores vide — graphe non modifié.")
        return

    # ── Pré-construction du lookup en O(|scores|) ─────────────────────────────
    # Clé : (trip_id, stop_id_dep, dep_minutes) → fraud_score
    # En cas de doublons (ne devrait pas arriver), la dernière valeur gagne.
    score_lookup: dict[tuple[str, str, int], float] = {
        (s.trip_id, s.stop_id_dep, s.dep_minutes): s.fraud_score
        for s in scores
    }

    n_injected = 0
    n_missed   = 0

    # ── Itération sur le graphe en O(|arcs|) ─────────────────────────────────
    for node, arcs in graph.items():
        for arc in arcs:

            # Seuls les arcs TRAIN sont scorés
            if arc.arc_type != ArcType.TRAIN:
                continue

            # Construction de la clé de correspondance depuis l'arc
            key = (
                arc.trip_id or "",        # trip_id de l'arc (peut être None pour CORR)
                arc.source.stop_id,       # code UIC de la gare de départ
                arc.source.time_minutes,  # heure de départ en minutes
            )

            if key in score_lookup:
                arc.fraud_score = score_lookup[key]
                n_injected += 1
            else:
                # Arc sans score ML correspondant
                # Peut arriver si le GTFS a été filtré différemment entre
                # le builder et l'appel API (ex: tronçons trop courts filtrés
                # par MIN_BOARD_DURATION_MINUTES dans l'un mais pas l'autre).
                n_missed += 1
                logger.debug(
                    "[inject_scores] Arc TRAIN sans score ML : "
                    "trip_id=%s, gare=%s, dep_min=%d → fraud_score reste 0.0",
                    arc.trip_id, arc.source.stop_id, arc.source.time_minutes,
                )

    # ── Rapport d'injection ───────────────────────────────────────────────────
    if n_missed > 0:
        logger.warning(
            "[inject_scores] %d arc(s) TRAIN sans score ML (fraud_score = 0.0). "
            "%d arc(s) scorés avec succès. "
            "Vérifiez la cohérence entre les tronçons GTFS et les scores API.",
            n_missed, n_injected,
        )
    else:
        logger.info(
            "[inject_scores] ✅ %d arc(s) TRAIN scorés avec succès.",
            n_injected,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 : Conversion DataFrame → PredictRequest (propagation service_date)
# ─────────────────────────────────────────────────────────────────────────────

def _build_predict_request(
    troncons_df: pd.DataFrame,
    service_date: date,
) -> PredictRequest:
    """
    Convertit un DataFrame de tronçons GTFS en PredictRequest pour l'API ML.

    Rôle de service_date :
        Le GTFS ne stocke pas la date dans les tronçons — un tronçon est
        défini par (trip_id, stop_sequence) indépendamment de la date.
        La date de circulation vient de calendar_dates via service_id.

        Lors de la construction des tronçons dans run(), on filtre déjà
        les trips qui circulent à service_date. Ici on propage cette date
        explicitement dans chaque TronconInput pour que TemporalFeatureTransformer
        puisse calculer is_weekend, is_vacances, is_peak_hour, etc.

        Ce pattern (injection depuis le contexte plutôt que modification du
        GTFSPreprocessor) respecte la Single Responsibility Principle :
            GTFSPreprocessor → structure des tronçons (agnostique à la date)
            solver           → contexte métier (date de la tournée)

    Paramètres :
        troncons_df  : DataFrame de tronçons (output de GTFSPreprocessor.build_troncons())
                       Colonnes requises : trip_id, train_number, service_id,
                       stop_sequence, stop_id_dep, dep_minutes, stop_id_arr,
                       arr_minutes, duration_min
        service_date : date de la tournée à propager dans chaque TronconInput

    Retourne :
        PredictRequest prêt à être sérialisé en JSON et envoyé à POST /predict/batch
    """
    troncons_input = []

    for _, row in troncons_df.iterrows():
        troncons_input.append(
            TronconInput(
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
                service_date=service_date,  # ← propagation clé
            )
        )

    return PredictRequest(troncons=troncons_input)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 : Appel API ML via httpx
# ─────────────────────────────────────────────────────────────────────────────

def fetch_scores_from_api(
    troncons_df: pd.DataFrame,
    service_date: date,
    predict_url: str = DEFAULT_PREDICT_API_URL,
    timeout: float = 30.0,
) -> list[PredictScoreItem]:
    """
    Appelle POST /predict/batch et retourne les scores ML.

    Utilise httpx synchrone :
        Le solver est un processus batch (pas async) — httpx synchrone
        est plus simple et suffisant pour ce cas d'usage.

    Résilience aux pannes :
        En cas d'erreur (timeout, HTTP 5xx, connexion refusée), la fonction
        retourne une liste vide SANS lever d'exception.

        → Le solver continue avec des fraud_score = 0.0 sur les arcs TRAIN.
        → Les tournées générées sont moins ciblées mais toujours valides
          d'un point de vue opérationnel (contraintes temporelles respectées).

        Ce choix de design évite que la panne du service ML bloque
        complètement la génération de tournées — degraded mode plutôt que crash.

    Paramètres :
        troncons_df  : DataFrame de tronçons (output de GTFSPreprocessor)
        service_date : date de service à propager dans les TronconInput
        predict_url  : URL de l'endpoint POST /predict/batch
                       (configurable pour les tests avec un mock httpx)
        timeout      : délai max en secondes avant abandon

    Retourne :
        list[PredictScoreItem] — vide si l'API est indisponible ou en erreur
    """
    predict_request = _build_predict_request(troncons_df, service_date)
    payload         = predict_request.model_dump(mode="json")

    logger.info(
        "[solver] Appel API predict/batch : %d tronçons → %s",
        len(troncons_df), predict_url,
    )

    try:
        response = httpx.post(predict_url, json=payload, timeout=timeout)
        response.raise_for_status()

        predict_response = PredictResponse.model_validate(response.json())

        logger.info(
            "[solver] ✅ API predict/batch : %d scores reçus.",
            predict_response.nb_troncons,
        )
        return predict_response.scores

    except httpx.TimeoutException:
        logger.error(
            "[solver] ⚠️  Timeout (>%.0fs) sur %s. "
            "Scores ML non disponibles — tournée avec fraud_score=0.0.",
            timeout, predict_url,
        )
        return []

    except httpx.HTTPStatusError as e:
        logger.error(
            "[solver] ⚠️  Erreur HTTP %d depuis %s : %s. "
            "Scores ML non disponibles — tournée avec fraud_score=0.0.",
            e.response.status_code,
            predict_url,
            e.response.text[:300],
        )
        return []

    except httpx.ConnectError:
        logger.error(
            "[solver] ⚠️  Connexion refusée sur %s. "
            "L'API predict est-elle démarrée ? (docker-compose up api) "
            "Tournée avec fraud_score=0.0.",
            predict_url,
        )
        return []

    except Exception as e:
        logger.exception(
            "[solver] ❌ Erreur inattendue lors de l'appel API predict : %s", e
        )
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 : Optimisation greedy (placeholder MILP)
# ─────────────────────────────────────────────────────────────────────────────

def _greedy_optimize(
    graph: dict[Node, list[Arc]],
    builder: TimeExpandedGraphBuilder,
    gare_depart_id: str,
    heure_depart_min: int,
    duree_max_minutes: int,
) -> dict:
    """
    Optimisation greedy : sélectionne toujours l'arc TRAIN avec le meilleur
    fraud_score disponible depuis la position courante.

    Logique :
        1. Trouver le nœud de départ (gare + premier train après heure_depart)
        2. À chaque étape :
            a. Si un arc TRAIN disponible dans le budget → prendre le meilleur score
            b. Sinon, si une correspondance disponible → prendre la plus courte
            c. Sinon → fin de la tournée
        3. S'arrêter quand budget épuisé ou plus d'arcs disponibles

    Pourquoi greedy et pas MILP ?
        Le solveur MILP PuLP (OrienteeringOptimizer) est prévu pour la v2.
        Le greedy garantit une solution faisable rapidement pour le POC.
        Il n'est PAS optimal — il peut rater un arc de score 0.9 accessible
        via une correspondance longue au profit d'un arc de score 0.7 direct.

    TODO : remplacer par OrienteeringOptimizer (PuLP/CBC) dans orienteering.py
           Le MILP formule ce problème comme un Orienteering Problem :
           maximiser sum(fraud_score × x_arc) sous contrainte de durée.

    Paramètres :
        graph             : graphe avec fraud_scores déjà injectés
        builder           : pour accéder à get_reachable_arcs()
        gare_depart_id    : code UIC de la gare de départ
        heure_depart_min  : heure de départ en minutes depuis minuit
        duree_max_minutes : budget temps total

    Retourne :
        dict avec :
            arcs            : liste ordonnée des Arc empruntés
            score_total     : somme des fraud_score sur les arcs TRAIN
            duree_minutes   : durée totale de la tournée
            nb_trains       : nombre d'arcs TRAIN (trains contrôlés)
    """
    # ── Trouver le nœud de départ ─────────────────────────────────────────────
    noeuds_depart = [
        n for n in graph
        if n.stop_id == gare_depart_id
        and n.time_minutes >= heure_depart_min
    ]

    if not noeuds_depart:
        raise ValueError(
            f"[solver] Aucun train depuis la gare {gare_depart_id} "
            f"après {heure_depart_min // 60:02d}:{heure_depart_min % 60:02d}. "
            "Vérifiez le code UIC et l'heure de départ."
        )

    # Premier train disponible après l'heure demandée
    noeud_courant = min(noeuds_depart, key=lambda n: n.time_minutes)
    temps_ecoule  = 0
    arcs_tournee: list[Arc] = []
    score_total   = 0.0

    # ── Boucle greedy ─────────────────────────────────────────────────────────
    while temps_ecoule < duree_max_minutes:
        arcs_disponibles = builder.get_reachable_arcs(graph, noeud_courant)

        if not arcs_disponibles:
            break

        # Priorité 1 : arc TRAIN dans le budget, meilleur fraud_score
        arcs_train = [
            a for a in arcs_disponibles
            if a.arc_type == ArcType.TRAIN
            and temps_ecoule + a.duration_min <= duree_max_minutes
        ]

        if arcs_train:
            meilleur_arc  = max(arcs_train, key=lambda a: a.fraud_score)
            arcs_tournee.append(meilleur_arc)
            score_total  += meilleur_arc.fraud_score
            temps_ecoule += meilleur_arc.duration_min
            noeud_courant = meilleur_arc.destination
            continue

        # Priorité 2 : correspondance dans le budget (la plus courte)
        arcs_corr = [
            a for a in arcs_disponibles
            if a.arc_type == ArcType.CORRESPONDANCE
            and temps_ecoule + a.duration_min <= duree_max_minutes
        ]

        if arcs_corr:
            corr = min(arcs_corr, key=lambda a: a.duration_min)
            arcs_tournee.append(corr)
            temps_ecoule  += corr.duration_min
            noeud_courant  = corr.destination
        else:
            break

    nb_trains = sum(1 for a in arcs_tournee if a.arc_type == ArcType.TRAIN)

    return {
        "arcs":          arcs_tournee,
        "score_total":   round(score_total, 4),
        "duree_minutes": temps_ecoule,
        "nb_trains":     nb_trains,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration principale
# ─────────────────────────────────────────────────────────────────────────────

def run(
    service_date: date,
    gare_depart_id: str,
    heure_depart_min: int,
    duree_max_minutes: int = 360,
    predict_url: str = DEFAULT_PREDICT_API_URL,
) -> dict:
    """
    Orchestre la génération complète d'une tournée optimisée.

    Séquence :
        GTFS → GraphBuilder → API Predict → inject_scores → _greedy_optimize

    Paramètres :
        service_date      : date de la tournée
        gare_depart_id    : code UIC 8 chiffres de la gare de départ
        heure_depart_min  : heure de départ en minutes depuis minuit (ex: 8h → 480)
        duree_max_minutes : durée maximale de la tournée (défaut : 360 min = 6h)
        predict_url       : URL de l'API ML (configurable pour les tests)

    Retourne :
        dict avec arcs, score_total, duree_minutes, nb_trains

    Lève :
        FileNotFoundError : si le GTFS n'est pas disponible
        ValueError        : si aucun service GTFS pour la date, ou aucun train
                            depuis la gare demandée
    """
    logger.info(
        "[solver] ══════ Démarrage ══════ "
        "date=%s | gare=%s | départ=%02dh%02d | durée_max=%dmin",
        service_date, gare_depart_id,
        heure_depart_min // 60, heure_depart_min % 60,
        duree_max_minutes,
    )

    # ── Étape 1 : Chargement GTFS ─────────────────────────────────────────────
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
        raise ValueError(
            f"[solver] Aucun service GTFS pour la date {service_date}. "
            "Vérifiez que le GTFS couvre cette date."
        )

    trips_du_jour = trips[trips["service_id"].isin(services_du_jour)]

    preprocessor = GTFSPreprocessor()
    troncons_df  = preprocessor.build_troncons(
        stop_times, trips_du_jour, stops, routes, ter_only=True
    )

    # Propagation de service_date (voir docstring de _build_predict_request)
    troncons_df["service_date"] = service_date

    logger.info(
        "[solver] GTFS chargé : %d tronçons TER pour le %s.",
        len(troncons_df), service_date,
    )

    # ── Étape 2 : Construction du graphe temps-étendu ─────────────────────────
    builder = TimeExpandedGraphBuilder()
    graph   = builder.build(troncons_df, service_date)

    # ── Étape 3 : Appel API ML pour les scores ────────────────────────────────
    scores = fetch_scores_from_api(troncons_df, service_date, predict_url)

    # ── Étape 4 : Injection des scores dans le graphe ─────────────────────────
    inject_scores(graph, scores)

    # ── Étape 5 : Optimisation ────────────────────────────────────────────────
    # TODO : remplacer _greedy_optimize par OrienteeringOptimizer (PuLP/CBC)
    tournee = _greedy_optimize(
        graph=graph,
        builder=builder,
        gare_depart_id=gare_depart_id,
        heure_depart_min=heure_depart_min,
        duree_max_minutes=duree_max_minutes,
    )

    logger.info(
        "[solver] ══════ Tournée générée ══════ "
        "%d trains | score=%.4f | durée=%dmin",
        tournee["nb_trains"], tournee["score_total"], tournee["duree_minutes"],
    )

    return tournee


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from datetime import date as dt

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 60)
    print("🚀  SAD-LAF OR Engine Solver — Mode CLI")
    print("=" * 60)

    try:
        result = run(
            service_date=dt.today(),
            gare_depart_id="87212027",   # Strasbourg
            heure_depart_min=8 * 60,     # 08:00
            duree_max_minutes=360,
        )

        print(f"\n✅  Tournée générée avec succès :")
        print(f"    Trains contrôlés : {result['nb_trains']}")
        print(f"    Score total      : {result['score_total']:.4f}")
        print(f"    Durée totale     : {result['duree_minutes']} min")

        print("\n📋  Séquence d'arcs :")
        for i, arc in enumerate(result["arcs"], 1):
            emoji = "🚆" if arc.arc_type == ArcType.TRAIN else "🔄"
            print(
                f"    {i:2d}. {emoji}  {arc.source.label!s:20s} → "
                f"{arc.destination.label!s:20s} "
                f"| {arc.duration_min:3d} min "
                f"| score={arc.fraud_score:.4f}"
            )

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌  Erreur : {e}")
        sys.exit(1)