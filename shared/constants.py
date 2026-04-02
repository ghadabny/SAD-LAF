from datetime import date

# ─────────────────────────────────────────────────────────────────────────────
# Contraintes métier LAF
# Source : cahier des charges Direction de Lignes Alsace / UO Bord
# ─────────────────────────────────────────────────────────────────────────────

# Durée minimale d'embarquement/débarquement pour qu'un contrôle soit possible
MIN_BOARD_DURATION_MINUTES: int = 6

# Durée maximale d'une mission agent (contrainte RH)
MAX_MISSION_DURATION_HOURS: int = 6

# Temps minimum de correspondance entre deux trains (contrainte opérationnelle)
MIN_TRANSFER_MINUTES: int = 5

# Temps d'attente maximum acceptable entre deux trains (qualité tournée agent)
# Au-delà, l'arc de correspondance n'est pas créé dans le graphe.
# Valeur calibrée sur le réseau GE : fréquence faible sur axes TGV transfrontaliers.
MAX_TRANSFER_MINUTES: int = 30

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
#
# Format : liste de tuples (date_début, date_fin) INCLUSES
# Année scolaire 2024-2025
#
# Pourquoi ici plutôt que dans temporal.py ?
#   → Ce sont des données métier, pas de la logique de code.
#     L'année prochaine on met à jour uniquement ce fichier.
# ─────────────────────────────────────────────────────────────────────────────

VACANCES_ZONE_B_2024_2025: list[tuple[date, date]] = [
    # Toussaint
    (date(2024, 10, 19), date(2024, 11,  4)),
    # Noël
    (date(2024, 12, 21), date(2025,  1,  6)),
    # Hiver
    (date(2025,  2, 22), date(2025,  3, 10)),
    # Printemps
    (date(2025,  4, 19), date(2025,  5,  5)),
    # Été
    (date(2025,  7,  5), date(2025,  8, 31)),
]

# Jours fériés France 2024-2025 (fixes + Alsace-Moselle)
# L'Alsace a 2 jours fériés supplémentaires : Vendredi Saint et 26 décembre
JOURS_FERIES_2024_2025: list[date] = [
    date(2024,  7, 14),  # Fête nationale
    date(2024,  8, 15),  # Assomption
    date(2024, 11,  1),  # Toussaint
    date(2024, 11, 11),  # Armistice
    date(2024, 12, 25),  # Noël
    date(2024, 12, 26),  # Saint-Étienne (Alsace-Moselle uniquement)
    date(2025,  1,  1),  # Jour de l'an
    date(2025,  4, 18),  # Vendredi Saint (Alsace-Moselle uniquement)
    date(2025,  4, 21),  # Lundi de Pâques
    date(2025,  5,  1),  # Fête du travail
    date(2025,  5,  8),  # Victoire 1945
    date(2025,  5, 29),  # Ascension
    date(2025,  6,  9),  # Lundi de Pentecôte
    date(2025,  7, 14),  # Fête nationale
]