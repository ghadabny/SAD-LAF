from datetime import date

# ─────────────────────────────────────────────────────────────────────────────
# Contraintes métier LAF
# Source : cahier des charges Direction de Lignes Alsace / UO Bord
# ─────────────────────────────────────────────────────────────────────────────

# Durée minimale d'embarquement/débarquement pour qu'un contrôle soit possible
MIN_BOARD_DURATION_MINUTES: int = 6

# Durée maximale d'une mission agent (contrainte RH)
MAX_MISSION_DURATION_HOURS: int = 6

# Temps minimum de correspondance global (fallback si gare non connue)
MIN_TRANSFER_MINUTES: int = 5

# Temps d'attente maximum acceptable entre deux trains (qualité tournée agent)
# Au-delà, l'arc de correspondance n'est pas créé dans le graphe.
MAX_TRANSFER_MINUTES: int = 60

# ── Délai de prise de service ─────────────────────────────────────────────────
# L'agent monte à bord 10 minutes après sa prise de service (PS).
# Exemple : PS 04h30 → premier train possible à 04h40.
MISSION_START_OFFSET_MINUTES: int = 10

# ── Temps de correspondance minimum par gare ──────────────────────────────────
# Source : données terrain agents LAF — Direction Lignes Alsace
# Calibrés sur la géographie des quais et le temps de traversée moyen.
# Un agent doit descendre d'un train, traverser le quai et monter dans le
# suivant dans ce délai minimum.
#
# SOLID — principe O : pour ajouter une nouvelle gare, ajouter simplement
# une entrée ici. Aucun autre code ne doit être modifié.
MIN_TRANSFER_BY_STOP: dict[str, int] = {
    "87212027": 8,   # Strasbourg (grande gare, quais éloignés)
    "87214007": 5,   # Sélestat
    "87214080": 5,   # Colmar
    "87182063": 6,   # Mulhouse
    "87213132": 5,   # Saverne
    "87214205": 4,   # Obernai
    "87213793": 4,   # Molsheim
    "87213587": 4,   # Erstein
    "87213843": 4,   # Barr
    "87214031": 4,   # Ribeauvillé
}

# Valeur par défaut pour toutes les gares non listées (haltes, gares simples)
MIN_TRANSFER_DEFAULT: int = 4


def get_min_transfer(stop_id: str) -> int:
    """
    Retourne le temps de correspondance minimum pour une gare donnée.

    Responsabilité unique : encapsuler la logique de lookup par gare.
    En cas de gare inconnue, retourne MIN_TRANSFER_DEFAULT.

    Usage dans builder.py :
        from shared.constants import get_min_transfer
        min_tr = get_min_transfer(stop_id)
    """
    return MIN_TRANSFER_BY_STOP.get(stop_id, MIN_TRANSFER_DEFAULT)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring ML
# ─────────────────────────────────────────────────────────────────────────────

# Échelle de score de risque de fraude (sortie du modèle ML, discrétisée)
FRAUD_SCORE_MIN: int = 1
FRAUD_SCORE_MAX: int = 4

# Durée à partir de laquelle un contrôle est considéré "pleinement efficace".
# En dessous, le score ML est pondéré proportionnellement à la durée du trajet.
# Formule : score_effectif = score_ml × min(1.0, duration_min / CONTROL_SATURATION_MINUTES)
CONTROL_SATURATION_MINUTES: int = 20

# Score effectif minimal en dessous duquel un arc TRAIN est exclu du graphe MILP.
# Les arcs à fraud_score=0.0 (non scorés) sont conservés pour la connectivité :
# CBC les ignorera naturellement via la fonction objectif.
MIN_FRAUD_SCORE_THRESHOLD: float = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Vacances scolaires — Zone B (académies Strasbourg et Nancy-Metz)
# Source : https://www.education.gouv.fr/calendrier-scolaire
# ─────────────────────────────────────────────────────────────────────────────

VACANCES_ZONE_B_2024_2025 = [
    # 2024-2025
    (date(2024, 10, 19), date(2024, 11,  4)),
    (date(2024, 12, 21), date(2025,  1,  6)),
    (date(2025,  2, 22), date(2025,  3, 10)),
    (date(2025,  4, 19), date(2025,  5,  5)),
    (date(2025,  7,  5), date(2025,  8, 31)),
    # 2025-2026
    (date(2025, 10, 18), date(2025, 11,  3)),
    (date(2025, 12, 20), date(2026,  1,  5)),
    (date(2026,  2, 14), date(2026,  3,  2)),
    (date(2026,  4, 18), date(2026,  5,  4)),
]

JOURS_FERIES_2024_2025 = [
    # 2024
    date(2024,  7, 14),
    date(2024,  8, 15),
    date(2024, 11,  1),
    date(2024, 11, 11),
    date(2024, 12, 25),
    # 2025
    date(2025,  1,  1),
    date(2025,  4, 21),
    date(2025,  5,  1),
    date(2025,  5,  8),
    date(2025,  5, 29),
    date(2025,  6,  9),
    date(2025,  7, 14),
    date(2025,  8, 15),
    date(2025, 11,  1),
    date(2025, 11, 11),
    date(2025, 12, 25),
    # 2026
    date(2026,  1,  1),
    date(2026,  4,  6),
    date(2026,  5,  1),
    date(2026,  5,  8),
    date(2026,  5, 14),
    date(2026,  5, 25),
]