"""
services/or_engine/constraints/operational.py
Contraintes opérationnelles LAF — validation post-optimisation.

SOLID — principe S :
    Chaque fonction a une responsabilité unique et est testable indépendamment.
    is_tournee_operationnellement_valide() orchestre ; elle ne calcule rien elle-même.

SOLID — principe O :
    Nouvelles règles métier = nouvelles fonctions ajoutées ici,
    sans modifier is_tournee_operationnellement_valide() si elle délègue
    à des collecteurs de règles.
"""
from services.or_engine.graph.transition import Arc, ArcType
from shared.constants import MAX_TRANSFER_MINUTES, MIN_FRAUD_SCORE_THRESHOLD


# ── Collecteurs d'anomalies ───────────────────────────────────────────────────

def get_correspondances_excessives(arcs: list[Arc]) -> list[Arc]:
    """
    Retourne les arcs de correspondance dont la durée dépasse MAX_TRANSFER_MINUTES.

    Si le builder fonctionne correctement (borne bisect = MAX_TRANSFER_MINUTES),
    cette liste doit toujours être vide. Sert de filet de sécurité post-optimisation.
    """
    return [
        a for a in arcs
        if a.arc_type == ArcType.CORRESPONDANCE
        and a.duration_min > MAX_TRANSFER_MINUTES
    ]


def get_trains_sous_seuil(arcs: list[Arc]) -> list[Arc]:
    """
    Retourne les arcs TRAIN dont le score effectif est inférieur au seuil.

    Les arcs à fraud_score=0.0 (non scorés par le ML) sont exclus :
    leur 0.0 signifie "absence de données", pas "absence de fraude".
    """
    return [
        a for a in arcs
        if a.arc_type == ArcType.TRAIN
        and 0.0 < a.fraud_score < MIN_FRAUD_SCORE_THRESHOLD
    ]


# ── Métriques ─────────────────────────────────────────────────────────────────

def ratio_temps_actif(arcs: list[Arc]) -> float:
    """
    Ratio temps en train / temps total de la tournée.
    Un ratio < 0.40 indique une tournée dominée par les attentes.
    Retourne 0.0 si la liste est vide.
    """
    temps_train = sum(a.duration_min for a in arcs if a.arc_type == ArcType.TRAIN)
    temps_total = sum(a.duration_min for a in arcs)
    return temps_train / temps_total if temps_total > 0 else 0.0


# ── Validation complète ───────────────────────────────────────────────────────

def is_tournee_operationnellement_valide(arcs: list[Arc]) -> tuple[bool, list[str]]:
    """
    Validation complète d'une tournée générée.

    Vérifie trois critères :
        1. Absence d'attentes excessives (> MAX_TRANSFER_MINUTES)
        2. Absence de trains sous le seuil de score (intérêt LAF insuffisant)
        3. Ratio temps actif >= 40%

    Retourne :
        (True, [])               si la tournée est valide
        (False, [anomalie, ...]) avec la liste des anomalies détectées
    """
    anomalies: list[str] = []
    anomalies.extend(_check_correspondances_excessives(arcs))
    anomalies.extend(_check_trains_sous_seuil(arcs))
    anomalies.extend(_check_ratio_temps_actif(arcs))
    return len(anomalies) == 0, anomalies


def _check_correspondances_excessives(arcs: list[Arc]) -> list[str]:
    """Retourne les messages d'anomalie pour les attentes excessives."""
    return [
        f"Attente excessive : {a.duration_min} min à {a.source.stop_name} "
        f"(max : {MAX_TRANSFER_MINUTES} min)"
        for a in get_correspondances_excessives(arcs)
    ]


def _check_trains_sous_seuil(arcs: list[Arc]) -> list[str]:
    """Retourne les messages d'anomalie pour les trains sous le seuil de score."""
    return [
        f"Train sous-seuil : {a.train_number} score={a.fraud_score:.3f} "
        f"(seuil : {MIN_FRAUD_SCORE_THRESHOLD})"
        for a in get_trains_sous_seuil(arcs)
    ]


def _check_ratio_temps_actif(arcs: list[Arc]) -> list[str]:
    """Retourne un message d'anomalie si le ratio temps actif est insuffisant."""
    ratio = ratio_temps_actif(arcs)
    if ratio < 0.40:
        return [f"Ratio temps actif insuffisant : {ratio:.0%} (seuil : 40%)"]
    return []