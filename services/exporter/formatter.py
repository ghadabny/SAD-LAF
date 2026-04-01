# services/exporter/formatter.py
"""
services/exporter/formatter.py — Mise en forme des tournées pour Dataverse.

Responsabilité unique : transformer un objet TourneeSchema (domaine Python)
en lignes CSV prêtes à être consommées par Power Automate / Dataverse.

Principes SOLID appliqués :
    S — formatter.py ne sait pas d'où vient la tournée ni où va le CSV.
        Il reçoit une TourneeSchema, retourne un DataFrame pandas.
    O — Pour ajouter une nouvelle colonne Dataverse, on ajoute une entrée
        dans _arc_to_row() sans toucher à la logique orchestratrice.
    D — Dépend de TourneeSchema (abstraction Pydantic), pas du JSON brut.

Schéma Dataverse cible (une ligne = un arc TRAIN contrôlé) :
    ┌────────────────────────────┬──────────────────────────────────────────┐
    │ Colonne Dataverse          │ Description                              │
    ├────────────────────────────┼──────────────────────────────────────────┤
    │ tournee_id                 │ UUID unique de la tournée                │
    │ service_date               │ Date de circulation (YYYY-MM-DD)         │
    │ generated_at               │ Horodatage de génération                 │
    │ gare_depart_id             │ Code UIC gare de départ tournée          │
    │ gare_depart_nom            │ Nom gare de départ tournée               │
    │ gare_arrivee_id            │ Code UIC gare d'arrivée tournée          │
    │ gare_arrivee_nom           │ Nom gare d'arrivée tournée               │
    │ duree_totale_minutes       │ Durée totale de la tournée               │
    │ score_total                │ Somme des scores de fraude               │
    │ nb_trains                  │ Nombre de trains contrôlés               │
    │ arc_ordre                  │ Position de l'arc dans la séquence       │
    │ train_numero               │ Numéro commercial du train               │
    │ trip_id                    │ Identifiant technique GTFS               │
    │ gare_dep_id                │ Code UIC départ du tronçon               │
    │ gare_dep_nom               │ Nom gare départ du tronçon               │
    │ heure_dep                  │ Heure départ (HH:MM)                     │
    │ gare_arr_id                │ Code UIC arrivée du tronçon              │
    │ gare_arr_nom               │ Nom gare arrivée du tronçon              │
    │ heure_arr                  │ Heure arrivée (HH:MM)                    │
    │ duree_troncon_minutes      │ Durée du tronçon en minutes              │
    │ fraud_score                │ Score de fraude prédit (0.0 à 1.0)       │
    │ priorite                   │ HAUTE / MOYENNE / BASSE (selon score)    │
    └────────────────────────────┴──────────────────────────────────────────┘

Une tournée de N arcs TRAIN → N lignes dans le CSV.
Les arcs CORRESPONDANCE sont exclus (pas d'action de contrôle possible).
"""
from __future__ import annotations

import uuid
from datetime import datetime

import pandas as pd

from shared.schemas import ArcSchema, ArcTypeSchema, TourneeSchema


# ─────────────────────────────────────────────────────────────────────────────
# Seuils de priorité (métier LAF)
# ─────────────────────────────────────────────────────────────────────────────

SEUIL_HAUTE:  float = 0.65   # fraud_score ≥ 0.65 → HAUTE
SEUIL_MOYENNE: float = 0.35  # fraud_score ≥ 0.35 → MOYENNE
                              # fraud_score <  0.35 → BASSE


class TourneeFormatter:
    """
    Convertit une TourneeSchema en DataFrame pandas formaté pour Dataverse.

    Usage :
        formatter = TourneeFormatter()
        df = formatter.format(tournee)
        # → DataFrame prêt pour to_csv()
    """

    def format(self, tournee: TourneeSchema) -> pd.DataFrame:
        """
        Transforme une tournée en DataFrame — une ligne par arc TRAIN.

        Paramètres :
            tournee : TourneeSchema validée par Pydantic (sortie de l'API /optimize)

        Retourne :
            pd.DataFrame avec les colonnes Dataverse.
            DataFrame vide si la tournée ne contient aucun arc TRAIN.

        Ne lève jamais d'exception — retourne un DataFrame vide en cas de problème.
        """
        # Identifiant unique de la tournée (tracabilité Dataverse)
        tournee_id = str(uuid.uuid4())

        # Métadonnées communes à toutes les lignes de cette tournée
        meta = self._build_tournee_meta(tournee, tournee_id)

        # Filtrer uniquement les arcs TRAIN (pas CORRESPONDANCE)
        arcs_train = [
            (i, arc) for i, arc in enumerate(tournee.arcs)
            if arc.arc_type == ArcTypeSchema.TRAIN
        ]

        if not arcs_train:
            print(
                f"[TourneeFormatter] ⚠️  Tournée {tournee_id} : "
                "aucun arc TRAIN — DataFrame vide retourné."
            )
            return pd.DataFrame()

        # Construire une ligne par arc TRAIN
        rows = [
            {**meta, **self._arc_to_row(arc, ordre=i + 1)}
            for i, (_, arc) in enumerate(arcs_train)
        ]

        df = pd.DataFrame(rows)

        # Ordonner les colonnes dans l'ordre défini par Dataverse
        df = df[self._column_order()]

        print(
            f"[TourneeFormatter] ✅ Tournée {tournee_id} : "
            f"{len(df)} arc(s) TRAIN formatés | "
            f"score={tournee.score_total:.4f} | "
            f"date={tournee.service_date}"
        )
        return df

    def format_batch(self, tournees: list[TourneeSchema]) -> pd.DataFrame:
        """
        Formate plusieurs tournées en un seul DataFrame.

        Usage typique : export en fin de journée de toutes les tournées générées.

        Paramètres :
            tournees : liste de TourneeSchema

        Retourne :
            pd.DataFrame concaténé — une ligne par arc TRAIN, toutes tournées confondues.
        """
        if not tournees:
            print("[TourneeFormatter] Aucune tournée à formater.")
            return pd.DataFrame()

        dfs = [self.format(t) for t in tournees]
        dfs_non_vides = [df for df in dfs if not df.empty]

        if not dfs_non_vides:
            return pd.DataFrame()

        result = pd.concat(dfs_non_vides, ignore_index=True)
        print(
            f"[TourneeFormatter] Batch : {len(tournees)} tournée(s) → "
            f"{len(result)} ligne(s) au total."
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers privés
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tournee_meta(
        self,
        tournee: TourneeSchema,
        tournee_id: str,
    ) -> dict:
        """
        Construit le dictionnaire des métadonnées communes à tous les arcs
        d'une même tournée.
        """
        return {
            "tournee_id":           tournee_id,
            "service_date":         tournee.service_date.strftime("%Y-%m-%d"),
            "generated_at":         tournee.generated_at.strftime("%Y-%m-%dT%H:%M:%S"),
            "gare_depart_id":       tournee.gare_depart.stop_id,
            "gare_depart_nom":      tournee.gare_depart.stop_name,
            "gare_arrivee_id":      tournee.gare_arrivee.stop_id,
            "gare_arrivee_nom":     tournee.gare_arrivee.stop_name,
            "duree_totale_minutes": tournee.duree_totale_minutes,
            "score_total":          round(tournee.score_total, 4),
            "nb_trains":            tournee.nb_trains,
        }

    def _arc_to_row(self, arc: ArcSchema, ordre: int) -> dict:
        """
        Convertit un arc TRAIN en dictionnaire de colonnes Dataverse.

        Paramètres :
            arc   : ArcSchema de type TRAIN
            ordre : position dans la séquence (1-indexé)
        """
        heure_dep = _minutes_to_hhmm(arc.source.time_minutes)
        heure_arr = _minutes_to_hhmm(arc.destination.time_minutes)

        return {
            "arc_ordre":           ordre,
            "train_numero":        arc.train_number or "",
            "trip_id":             arc.trip_id or "",
            "gare_dep_id":         arc.source.stop_id,
            "gare_dep_nom":        arc.source.stop_name,
            "heure_dep":           heure_dep,
            "gare_arr_id":         arc.destination.stop_id,
            "gare_arr_nom":        arc.destination.stop_name,
            "heure_arr":           heure_arr,
            "duree_troncon_minutes": arc.duration_min,
            "fraud_score":         round(arc.fraud_score, 4),
            "priorite":            _score_to_priorite(arc.fraud_score),
        }

    def _column_order(self) -> list[str]:
        """Ordre canonique des colonnes dans le CSV final."""
        return [
            # Métadonnées de la tournée
            "tournee_id",
            "service_date",
            "generated_at",
            "gare_depart_id",
            "gare_depart_nom",
            "gare_arrivee_id",
            "gare_arrivee_nom",
            "duree_totale_minutes",
            "score_total",
            "nb_trains",
            # Détail de l'arc TRAIN
            "arc_ordre",
            "train_numero",
            "trip_id",
            "gare_dep_id",
            "gare_dep_nom",
            "heure_dep",
            "gare_arr_id",
            "gare_arr_nom",
            "heure_arr",
            "duree_troncon_minutes",
            "fraud_score",
            "priorite",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions utilitaires module-level
# ─────────────────────────────────────────────────────────────────────────────

def _minutes_to_hhmm(minutes: int) -> str:
    """
    Convertit des minutes depuis minuit en chaîne HH:MM.

    Exemples :
        503 → "08:23"
        0   → "00:00"
        1439 → "23:59"
    """
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _score_to_priorite(score: float) -> str:
    """
    Convertit un fraud_score en niveau de priorité métier LAF.

    Seuils définis par la Direction de Lignes Alsace :
        ≥ 0.65 → HAUTE   (contrôle prioritaire)
        ≥ 0.35 → MOYENNE (contrôle recommandé)
        <  0.35 → BASSE  (contrôle opportuniste)
    """
    if score >= SEUIL_HAUTE:
        return "HAUTE"
    elif score >= SEUIL_MOYENNE:
        return "MOYENNE"
    return "BASSE"