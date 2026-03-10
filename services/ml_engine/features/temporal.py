import pandas as pd
from datetime import date

from services.ml_engine.features.base import BaseFeatureTransformer
from shared.constants import VACANCES_ZONE_B_2024_2025, JOURS_FERIES_2024_2025


class TemporalFeatureTransformer(BaseFeatureTransformer):
    """
    Calcule les features temporelles à partir d'un DataFrame de tronçons.

    Features produites :
        dep_hour          : heure de départ (0-23), extraite de dep_minutes
        dep_minute_of_day : heure en minutes depuis minuit (conservée telle quelle)
        day_of_week       : jour de la semaine (0=lundi ... 6=dimanche)
        is_weekend        : True si samedi ou dimanche
        is_vacances       : True si la date tombe pendant les vacances zone B
        is_jour_ferie     : True si la date est un jour férié (Alsace-Moselle)
        is_peak_hour      : True si l'heure est en heure de pointe
                            (7h-9h ou 17h-19h en semaine)

    Colonnes requises en entrée :
        dep_minutes  : int  — heure de départ en minutes depuis minuit
        service_date : date — date de circulation du train (ex: date(2024, 9, 2))

    Pourquoi fit() ne fait rien ici ?
        Les features temporelles sont purement calculatoires — pas besoin
        d'apprendre quoi que ce soit sur les données d'entraînement.
        Mais on respecte l'interface BaseFeatureTransformer pour que
        pipeline.py puisse appeler fit_transform() uniformément sur toutes
        les classes de features.
    """

    # Heures de pointe : fort trafic de pendulaires = plus de voyageurs
    # = plus de risque de fraude "dans la masse"
    PEAK_MORNING_START: int = 7
    PEAK_MORNING_END: int   = 9
    PEAK_EVENING_START: int = 17
    PEAK_EVENING_END: int   = 19

    def fit(self, df: pd.DataFrame) -> "TemporalFeatureTransformer":
        """
        Rien à apprendre pour les features temporelles.
        Retourne self pour permettre le chaînage fit().transform().
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les colonnes de features temporelles au DataFrame.

        Ne modifie pas le DataFrame d'entrée (travaille sur une copie).
        """
        self._validate(df)
        result = df.copy()

        # ── Heure ────────────────────────────────────────────────────────────
        # dep_minutes = 556 → dep_hour = 9, dep_minute_of_day = 556
        result["dep_hour"] = result["dep_minutes"] // 60
        result["dep_minute_of_day"] = result["dep_minutes"]  # alias explicite

        # ── Jour de la semaine ────────────────────────────────────────────────
        # service_date est une colonne de dates Python (ou Timestamp pandas)
        # .apply() est nécessaire car .dt.dayofweek ne fonctionne pas sur
        # des colonnes de type object contenant des date Python natifs
        result["day_of_week"] = result["service_date"].apply(
            lambda d: d.weekday()  # 0=lundi, 6=dimanche
        )

        # ── Week-end ──────────────────────────────────────────────────────────
        result["is_weekend"] = result["day_of_week"] >= 5

        # ── Vacances scolaires zone B ─────────────────────────────────────────
        # On précalcule un set de toutes les dates de vacances pour
        # une recherche en O(1) au lieu de O(n_periodes) par ligne
        vacances_set = self._build_vacances_set()
        result["is_vacances"] = result["service_date"].apply(
            lambda d: _to_date(d) in vacances_set
        )

        # ── Jours fériés Alsace-Moselle ───────────────────────────────────────
        feries_set = set(JOURS_FERIES_2024_2025)
        result["is_jour_ferie"] = result["service_date"].apply(
            lambda d: _to_date(d) in feries_set
        )

        # ── Heures de pointe ──────────────────────────────────────────────────
        # Seulement en semaine — le week-end n'a pas vraiment de pointe
        is_weekday = ~result["is_weekend"]
        is_morning_peak = (
            (result["dep_hour"] >= self.PEAK_MORNING_START) &
            (result["dep_hour"] < self.PEAK_MORNING_END)
        )
        is_evening_peak = (
            (result["dep_hour"] >= self.PEAK_EVENING_START) &
            (result["dep_hour"] < self.PEAK_EVENING_END)
        )
        result["is_peak_hour"] = is_weekday & (is_morning_peak | is_evening_peak)

        return result

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Vérifie que les colonnes requises sont présentes.
        Fail-fast : vaut mieux une erreur claire au début qu'un KeyError
        mystérieux au milieu du pipeline.
        """
        required = {"dep_minutes", "service_date"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"[TemporalFeatureTransformer] Colonnes manquantes : {missing}\n"
                f"Colonnes présentes : {list(df.columns)}"
            )

    def _build_vacances_set(self) -> set[date]:
        """
        Construit un set de toutes les dates de vacances individuelles
        à partir de la liste de périodes dans constants.py.

        Exemple :
            (date(2024,10,19), date(2024,11,4))
            → {date(2024,10,19), date(2024,10,20), ..., date(2024,11,4)}

        Un set permet une recherche en O(1) : 'd in vacances_set'
        est instantané même avec des milliers de tronçons.
        """
        from datetime import timedelta
        vacances_dates: set[date] = set()
        for debut, fin in VACANCES_ZONE_B_2024_2025:
            current = debut
            while current <= fin:
                vacances_dates.add(current)
                current += timedelta(days=1)
        return vacances_dates


# ── Fonction utilitaire module-level ──────────────────────────────────────────

def _to_date(value) -> date:
    """
    Normalise une valeur en date Python natif.
    Gère les deux cas possibles selon la source du DataFrame :
        - pandas Timestamp (si la colonne vient d'un pd.read_csv avec parse_dates)
        - date Python natif (si la colonne vient de nos fixtures de test)
    """
    if isinstance(value, date):
        return value
    return value.date()  # pandas Timestamp → date Python