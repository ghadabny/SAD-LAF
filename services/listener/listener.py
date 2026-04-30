# services/listener/listener.py
"""
services/listener/listener.py — Listener event-driven basé sur des fichiers JSON.

Conformité SOLID :
    S — FileListener : orchestration filesystem uniquement.
        TourneeGenerationService : logique métier de génération isolée.
        Helpers _archiver_* : responsabilité archivage isolée.

    O — _DECISION_HANDLERS : dispatch par dictionnaire.
        Ajouter une action = ajouter une entrée dans le dict + un handler.
        La classe FileListener ne change pas.

    L — Pas d'héritage → non applicable.

    I — FileListener dépend de TourneeGenerationService (interface implicite).
        Les dépendances (exporter, formatter, store) sont injectées
        dans TourneeGenerationService via le constructeur.

    D — TourneeExporter et TourneeFormatter sont injectés dans
        TourneeGenerationService, pas instanciés en dur dans les méthodes.
        solver_run est injectable via le paramètre du constructeur.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from pydantic import ValidationError

from services.api.booking_store import STATUT_EN_ATTENTE, get_booking_store
from services.exporter.export import TourneeExporter
from services.exporter.formatter import TourneeFormatter
from services.listener.schemas import DecisionFileSchema, TourneeRequestFileSchema
from services.or_engine.graph.transition import ArcType as ArcTypeDomaine
from services.or_engine.solver import run as solver_run
from shared.config import config
from shared.schemas import (
    ArcSchema,
    ArcTypeSchema,
    NodeSchema,
    OptimizeResponseV2Schema,
    TourneeRequestV2Schema,
    TourneeSchema,
)

logger = logging.getLogger(__name__)

_EXTENSION_PROCESSING = ".processing"
_EXTENSION_ERROR      = ".error.json"


# ─────────────────────────────────────────────────────────────────────────────
# Service de génération — SRP + DIP
# ─────────────────────────────────────────────────────────────────────────────

class TourneeGenerationService:
    """
    Encapsule toute la logique métier de génération d'une tournée.

    SOLID — S : une seule responsabilité → générer et enregistrer une tournée.
    SOLID — D : TourneeExporter, TourneeFormatter et solver sont injectés
                via le constructeur → mockables sans patch dans les tests.

    Usage production :
        service = TourneeGenerationService()

    Usage test :
        service = TourneeGenerationService(
            solver=fake_solver,
            exporter=FakeTourneeExporter(),
            formatter=FakeTourneeFormatter(),
        )
    """

    def __init__(
        self,
        solver:    Callable  = solver_run,
        exporter:  TourneeExporter  | None = None,
        formatter: TourneeFormatter | None = None,
    ) -> None:
        self._solver    = solver
        self._exporter  = exporter  or TourneeExporter()
        self._formatter = formatter or TourneeFormatter()

    def generer(self, request: TourneeRequestV2Schema) -> str:
        """
        Génère une tournée complète et l'enregistre EN_ATTENTE dans le store.

        Retourne le tournee_id généré.

        Lève :
            FileNotFoundError : GTFS non disponibles
            ValueError        : aucune tournée possible
        """
        gare_arrivee    = request.gare_arrivee_effective
        score_libre     = self._score_libre(request, gare_arrivee)
        result          = self._resoudre(request, gare_arrivee)
        score_perte_pct = self._calculer_perte(score_libre, result["score_total"])

        now        = datetime.now()
        tournee_id = self._generer_id(request, now)

        response_obj = self._construire_reponse(request, result, tournee_id, score_perte_pct, now)
        self._exporter_csv(response_obj, tournee_id)
        self._enregistrer_store(request, result, tournee_id)

        return tournee_id

    # ── Étapes privées ────────────────────────────────────────────────────────

    def _score_libre(
        self,
        request: TourneeRequestV2Schema,
        gare_arrivee: Optional[str],
    ) -> Optional[float]:
        """Calcule le score sans contrainte de retour (pour mesurer la perte)."""
        if gare_arrivee is None:
            return None
        try:
            result = self._solver(
                service_date      = request.service_date,
                gare_depart_id    = request.gare_depart_id,
                heure_depart_min  = request.heure_depart_min,
                duree_max_minutes = request.duree_max_minutes,
                gare_arrivee_id   = None,
            )
            return result["score_total"]
        except (ValueError, FileNotFoundError):
            return None

    def _resoudre(
        self,
        request: TourneeRequestV2Schema,
        gare_arrivee: Optional[str],
    ) -> dict:
        """Lance le solver avec la contrainte de retour."""
        result = self._solver(
            service_date      = request.service_date,
            gare_depart_id    = request.gare_depart_id,
            heure_depart_min  = request.heure_depart_min,
            duree_max_minutes = request.duree_max_minutes,
            gare_arrivee_id   = gare_arrivee,
        )
        if not result["arcs"]:
            raise ValueError(
                "Aucune tournée possible depuis cette gare dans cette fenêtre. "
                "Essayez le mode 'decouche' ou élargissez la plage PS/FS."
            )
        return result

    @staticmethod
    def _calculer_perte(score_libre: Optional[float], score_total: float) -> float:
        if score_libre and score_libre > 0:
            return max(0.0, (score_libre - score_total) / score_libre * 100)
        return 0.0

    @staticmethod
    def _generer_id(request: TourneeRequestV2Schema, now: datetime) -> str:
        agent    = (request.agent_id or "INCONNU").upper().replace(" ", "_")
        date_str = request.service_date.strftime("%Y%m%d")
        uid      = str(uuid.uuid4()).split("-")[0].upper()
        return f"TRN_{agent}_{date_str}_{uid}"

    @staticmethod
    def _construire_reponse(
        request: TourneeRequestV2Schema,
        result: dict,
        tournee_id: str,
        score_perte_pct: float,
        now: datetime,
    ) -> OptimizeResponseV2Schema:
        arcs_tournee = result["arcs"]
        tournee = TourneeSchema(
            arcs                 = [_arc_to_schema(a) for a in arcs_tournee],
            score_total          = result["score_total"],
            duree_totale_minutes = result["duree_minutes"],
            nb_trains            = max(result["nb_trains"], 1),
            gare_depart          = _node_to_schema(arcs_tournee[0].source),
            gare_arrivee         = _node_to_schema(arcs_tournee[-1].destination),
            service_date         = request.service_date,
            generated_at         = now,
        )
        return OptimizeResponseV2Schema(
            tournee           = tournee,
            request           = request,
            tournee_id        = tournee_id,
            statut            = STATUT_EN_ATTENTE,
            score_perte_pct   = round(score_perte_pct, 2),
            trains_en_conflit = [],
            csv_path          = None,
            warning_messages  = [],
            optimized_at      = now,
        )

    def _exporter_csv(self, response_obj: OptimizeResponseV2Schema, tournee_id: str) -> None:
        df       = self._formatter.format_with_id(response_obj, tournee_id)
        csv_path = self._exporter.export_from_df(df, response_obj, tournee_id)
        self._exporter.export_json_flat(df, response_obj, tournee_id)
        logger.info("[GenerationService] CSV exporté : %s", csv_path.name)

    @staticmethod
    def _enregistrer_store(
        request: TourneeRequestV2Schema,
        result: dict,
        tournee_id: str,
    ) -> None:
        if not request.agent_id:
            return
        trip_ids = [
            a.trip_id
            for a in result["arcs"]
            if a.arc_type == ArcTypeDomaine.TRAIN and a.trip_id
        ]
        store = get_booking_store()
        store.register_tournee(
            tournee_id   = tournee_id,
            agent_id     = request.agent_id,
            service_date = request.service_date,
            trip_ids     = trip_ids,
        )
        logger.info(
            "[GenerationService] Tournée %s EN_ATTENTE pour %s (%d trains).",
            tournee_id, request.agent_id, len(trip_ids),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Handlers de décision — OCP
# Ajouter une action = ajouter une fonction + une entrée dans _DECISION_HANDLERS.
# FileListener ne change jamais.
# ─────────────────────────────────────────────────────────────────────────────

def _handle_validate(decision: DecisionFileSchema) -> None:
    """Valide une tournée EN_ATTENTE dans le store."""
    get_booking_store().validate_tournee(
        tournee_id           = decision.tournee_id,
        validated_by         = decision.acteur_id,
        max_agents_per_train = 1,
    )
    logger.info("[Listener] Tournée %s validée par %s.", decision.tournee_id, decision.acteur_id)


def _handle_refuse(decision: DecisionFileSchema) -> None:
    """Refuse une tournée EN_ATTENTE dans le store."""
    get_booking_store().refuse_tournee(
        tournee_id = decision.tournee_id,
        refused_by = decision.acteur_id,
        motif      = decision.motif,
    )
    logger.info(
        "[Listener] Tournée %s refusée par %s (motif: %s).",
        decision.tournee_id, decision.acteur_id, decision.motif or "—",
    )


def _handle_cancel(decision: DecisionFileSchema) -> None:
    """Annule une tournée VALIDEE dans le store."""
    get_booking_store().cancel_tournee(
        tournee_id   = decision.tournee_id,
        cancelled_by = decision.acteur_id,
    )
    logger.info("[Listener] Tournée %s annulée par %s.", decision.tournee_id, decision.acteur_id)


# Dispatch OCP — ajouter ici sans toucher à FileListener
_DECISION_HANDLERS: dict[str, Callable[[DecisionFileSchema], None]] = {
    "VALIDATE": _handle_validate,
    "REFUSE":   _handle_refuse,
    "CANCEL":   _handle_cancel,
}


# ─────────────────────────────────────────────────────────────────────────────
# FileListener — orchestration filesystem uniquement (SRP)
# ─────────────────────────────────────────────────────────────────────────────

class FileListener:
    """
    Surveille les dossiers data/requests/ et data/decisions/.

    SOLID — S : responsabilité unique = orchestration filesystem.
               La logique métier est déléguée à TourneeGenerationService
               et aux handlers _DECISION_HANDLERS.
    SOLID — D : TourneeGenerationService est injecté → testable sans I/O réel.

    Usage production :
        listener = FileListener()

    Usage test :
        listener = FileListener(generation_service=FakeGenerationService())
    """

    def __init__(
        self,
        generation_service: TourneeGenerationService | None = None,
    ) -> None:
        self._generation_service = generation_service or TourneeGenerationService()

        self._requests_dir:        Path = config.REQUESTS_DIR
        self._decisions_dir:       Path = config.DECISIONS_DIR
        self._requests_processed:  Path = self._requests_dir  / "processed"
        self._requests_errors:     Path = self._requests_dir  / "errors"
        self._decisions_processed: Path = self._decisions_dir / "processed"
        self._decisions_errors:    Path = self._decisions_dir / "errors"

        for d in (
            self._requests_dir,   self._requests_processed,  self._requests_errors,
            self._decisions_dir,  self._decisions_processed, self._decisions_errors,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ── Interface publique ────────────────────────────────────────────────────

    def process_pending_requests(self) -> None:
        """Traite tous les fichiers .json dans data/requests/ (premier arrivé, premier servi)."""
        for fichier in self._lister_json(self._requests_dir):
            self._traiter_requete(fichier)

    def process_pending_decisions(self) -> None:
        """Traite tous les fichiers .json dans data/decisions/."""
        for fichier in self._lister_json(self._decisions_dir):
            self._traiter_decision(fichier)

    # ── Traitement atomique ───────────────────────────────────────────────────

    def _traiter_requete(self, fichier: Path) -> None:
        fichier_processing = _renommer_en_processing(fichier)
        if fichier_processing is None:
            return

        logger.info("[Listener] ▶ Requête : %s", fichier.name)
        try:
            donnees        = _lire_json(fichier_processing)
            schema_fichier = TourneeRequestFileSchema.model_validate(donnees)
            request        = TourneeRequestV2Schema(
                gare_depart_id       = schema_fichier.gare_depart_id,
                heure_ps_min         = schema_fichier.heure_ps_min,
                heure_fs_min         = schema_fichier.heure_fs_min,
                service_date         = schema_fichier.service_date,
                mode                 = schema_fichier.mode,
                show_scores          = schema_fichier.show_scores,
                max_agents_per_train = schema_fichier.max_agents_per_train,
                agent_id             = schema_fichier.agent_id,
                gare_arrivee_id      = schema_fichier.gare_arrivee_id,
            )
            self._generation_service.generer(request)
            _archiver_succes(fichier_processing, self._requests_processed)
            logger.info("[Listener] ✅ Requête traitée : %s", fichier.name)

        except (ValidationError, ValueError, KeyError, FileNotFoundError) as exc:
            logger.warning("[Listener] ⚠ Erreur requête '%s' : %s", fichier.name, exc)
            _archiver_erreur(fichier_processing, self._requests_errors, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Listener] ✖ Erreur inattendue '%s' : %s", fichier.name, exc)
            _archiver_erreur(fichier_processing, self._requests_errors, str(exc))

    def _traiter_decision(self, fichier: Path) -> None:
        fichier_processing = _renommer_en_processing(fichier)
        if fichier_processing is None:
            return

        logger.info("[Listener] ▶ Décision : %s", fichier.name)
        try:
            donnees  = _lire_json(fichier_processing)
            decision = DecisionFileSchema.model_validate(donnees)

            # SOLID-O : dispatch par dictionnaire, pas de if/elif
            handler = _DECISION_HANDLERS.get(decision.action)
            if handler is None:
                raise ValueError(f"Action non gérée : '{decision.action}'.")
            handler(decision)

            _archiver_succes(fichier_processing, self._decisions_processed)
            logger.info("[Listener] ✅ Décision traitée : %s", fichier.name)

        except (ValidationError, ValueError, KeyError) as exc:
            logger.warning("[Listener] ⚠ Erreur décision '%s' : %s", fichier.name, exc)
            _archiver_erreur(fichier_processing, self._decisions_errors, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Listener] ✖ Erreur inattendue '%s' : %s", fichier.name, exc)
            _archiver_erreur(fichier_processing, self._decisions_errors, str(exc))

    # ── Utilitaire ────────────────────────────────────────────────────────────

    @staticmethod
    def _lister_json(dossier: Path) -> list[Path]:
        """Fichiers .json racine uniquement, triés par date de création."""
        return sorted(
            (f for f in dossier.iterdir() if f.is_file() and f.suffix == ".json"),
            key=lambda f: f.stat().st_ctime,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires filesystem (module-level, pas de state)
# ─────────────────────────────────────────────────────────────────────────────

def _renommer_en_processing(fichier: Path) -> Optional[Path]:
    """Rename atomique → .processing. Retourne None si déjà pris."""
    cible = fichier.with_suffix(_EXTENSION_PROCESSING)
    try:
        fichier.rename(cible)
        return cible
    except FileNotFoundError:
        return None


def _lire_json(fichier: Path) -> dict:
    with fichier.open(encoding="utf-8") as f:
        return json.load(f)


def _archiver_succes(fichier_processing: Path, dossier_dest: Path) -> None:
    destination = dossier_dest / f"{fichier_processing.stem}.json"
    fichier_processing.rename(destination)


def _archiver_erreur(
    fichier_processing: Path,
    dossier_errors: Path,
    message_erreur: str,
) -> None:
    nom          = fichier_processing.stem
    dest_fichier = dossier_errors / f"{nom}.json"
    dest_erreur  = dossier_errors / f"{nom}{_EXTENSION_ERROR}"
    try:
        fichier_processing.rename(dest_fichier)
    except Exception as exc:  # noqa: BLE001
        logger.error("[Listener] Impossible d'archiver '%s' : %s", fichier_processing.name, exc)
        return
    with dest_erreur.open("w", encoding="utf-8") as f:
        json.dump(
            {"fichier": f"{nom}.json", "erreur": message_erreur, "timestamp": datetime.now().isoformat()},
            f, ensure_ascii=False, indent=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de conversion domaine
# ─────────────────────────────────────────────────────────────────────────────

def _node_to_schema(node) -> NodeSchema:
    return NodeSchema(
        stop_id=node.stop_id, stop_name=node.stop_name,
        time_minutes=node.time_minutes, service_date=node.service_date,
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