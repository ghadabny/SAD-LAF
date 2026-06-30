"""
services/exporter/formatter.py — Mise en forme des tournées pour export CSV.

CORRECTIONS v3.1 :
    Bug 1 & 2 — double UUID et double ligne CSV :
        Cause : _generate_tournee_id() était appelé dans format() via uuid4().
                Le router appelait format() une première fois pour extraire l'ID,
                puis export() appelait format() une deuxième fois → 2 UUIDs, 2 lignes.
        Fix   : Ajout de format_with_id(response, tournee_id) qui reçoit l'ID
                déjà généré par le router. format() existant est conservé pour
                compatibilité (tests unitaires) mais l'endpoint v2 utilise
                format_with_id().
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from shared.schemas import ArcSchema, ArcTypeSchema

if TYPE_CHECKING:
    from shared.schemas import OptimizeResponseV2Schema

# ── Seuils de priorité ────────────────────────────────────────────────────────
SEUIL_HAUTE:   float = 0.65
SEUIL_MOYENNE: float = 0.35


class TourneeFormatter:
    """
    Formate une OptimizeResponseV2Schema en lignes CSV exportables.

    Deux méthodes publiques :
        format(response)                → génère un nouvel UUID (compat. tests)
        format_with_id(response, id)    → utilise l'ID fourni (endpoint v2)
    """

    def format(self, response: "OptimizeResponseV2Schema") -> pd.DataFrame:
        """
        Formate la tournée en générant un nouveau tournee_id.
        Conservé pour compatibilité avec les tests unitaires.
        """
        tournee_id = self._generate_tournee_id(response)
        return self._build_dataframe(response, tournee_id)

    def format_with_id(
        self,
        response: "OptimizeResponseV2Schema",
        tournee_id: str,
    ) -> pd.DataFrame:
        """
        Formate la tournée en utilisant le tournee_id fourni par le router.

        FIX Bug 1 & 2 : cette méthode n'appelle PAS uuid4(), garantissant
        que le CSV et la réponse JSON partagent exactement le même ID,
        et qu'il n'y a qu'une seule ligne par arc TRAIN.
        """
        return self._build_dataframe(response, tournee_id)

    # ── Logique interne ───────────────────────────────────────────────────────

    def _build_dataframe(
        self,
        response: "OptimizeResponseV2Schema",
        tournee_id: str,
    ) -> pd.DataFrame:
        """
        Construit le DataFrame, une ligne par arc TRAIN.
        Les arcs CORRESPONDANCE sont exclus — seule la durée d'attente
        entre deux trains est reportée sur la ligne du train précédent.
        """
        agent_id    = response.request.agent_id or "INCONNU"
        show_scores = response.request.show_scores

        arcs_train = [
            a for a in response.tournee.arcs
            if a.arc_type == ArcTypeSchema.TRAIN
        ]

        if not arcs_train:
            return pd.DataFrame()

        gare_finale = arcs_train[-1].destination.stop_name
        heure_fin   = _minutes_to_hhmm(arcs_train[-1].destination.time_minutes)

        rows = []
        for idx, arc in enumerate(arcs_train):
            attente_min = self._attente_apres(response.tournee.arcs, arc)

            row: dict = {
                "tournee_id":           tournee_id,
                "agent_id":             agent_id,
                "service_date":         response.tournee.service_date.isoformat(),
                "generated_at":         response.optimized_at.strftime("%Y-%m-%dT%H:%M:%S"),
                "ordre":                idx + 1,
                "numero_train":         arc.train_number or "",
                "gare_dep":             arc.source.stop_name,
                "heure_dep":            _minutes_to_hhmm(arc.source.time_minutes),
                "gare_arr":             arc.destination.stop_name,
                "heure_arr":            _minutes_to_hhmm(arc.destination.time_minutes),
                "duree_trajet_min":     arc.duration_min,
                "attente_suivant_min":  attente_min,
                "priorite":             _score_to_priorite(arc.fraud_score),
                "score_total_tournee":  round(response.tournee.score_total, 4),
                "nb_trains":            response.tournee.nb_trains,
                "duree_totale_min":     response.tournee.duree_totale_minutes,
                "gare_finale":          gare_finale,
                "heure_fin":            heure_fin,
                "score_perte_pct":      round(response.score_perte_pct, 1),
                "warnings":             " | ".join(response.warning_messages) if response.warning_messages else "",
            }

            if show_scores:
                row["fraud_score"] = round(arc.fraud_score, 4)

            rows.append(row)

        return pd.DataFrame(rows)

    def _generate_tournee_id(self, response: "OptimizeResponseV2Schema") -> str:
        """
        Génère un ID unique — utilisé UNIQUEMENT par format() pour la compat. tests.
        L'endpoint optimize_v2 utilise format_with_id() et génère son propre ID.
        """
        agent_id = (response.request.agent_id or "UNKNOWN").upper().replace(" ", "_")
        date_str = response.tournee.service_date.strftime("%Y%m%d")
        uid      = str(uuid.uuid4()).split("-")[0].upper()
        return f"TRN_{agent_id}_{date_str}_{uid}"

    def _attente_apres(
        self,
        all_arcs: list[ArcSchema],
        arc_train: ArcSchema,
    ) -> int:
        """Durée d'attente (correspondance) immédiatement après un arc TRAIN."""
        found = False
        for a in all_arcs:
            if found and a.arc_type == ArcTypeSchema.CORRESPONDANCE:
                return a.duration_min
            if found and a.arc_type == ArcTypeSchema.TRAIN:
                return 0
            if a is arc_train:
                found = True
        return 0


# ── Fonctions utilitaires module-level ────────────────────────────────────────

def _minutes_to_hhmm(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _score_to_priorite(score: float) -> str:
    if score >= SEUIL_HAUTE:
        return "HAUTE"
    if score >= SEUIL_MOYENNE:
        return "MOYENNE"
    return "BASSE"