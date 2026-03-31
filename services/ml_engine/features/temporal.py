# services/ml_engine/features/temporal.py
import pandas as pd
from datetime import date, timedelta

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

    SOLID — principe D :
        _build_vacances_set() et _build_feries_set() sont appelés UNE SEULE FOIS
        dans fit() et stockés dans self._vacances_set / self._feries_set.
        transform() effectue uniquement la recherche en O(1) dans ces sets.
        Avant ce refactoring, chaque appel à transform() reconstruisait les sets
        depuis zéro — inutile et coûteux sur de grands DataFrames.
    """

    PEAK_MORNING_START: int = 7
    PEAK_MORNING_END: int   = 9
    PEAK_EVENING_START: int = 17
    PEAK_EVENING_END: int   = 19

    def __init__(self):
        # Sets précalculés une seule fois dans fit()
        self._vacances_set: set[date] = set()
        self._feries_set: set[date]   = set()

    def fit(self, df: pd.DataFrame) -> "TemporalFeatureTransformer":
        """
        Précalcule les sets de dates calendaires.

        Ces sets sont dérivés de constantes (constants.py), pas du DataFrame df.
        fit() est le moment correct pour initialiser tout état réutilisable par
        transform() — que les données viennent du train ou de la production.

        Retourne self pour le chaînage fit().transform().
        """
        self._vacances_set = _build_vacances_set()
        self._feries_set   = set(JOURS_FERIES_2024_2025)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute les colonnes de features temporelles au DataFrame.

        Requiert que fit() ait été appelé au préalable (sets précalculés).
        Ne modifie pas le DataFrame d'entrée (travaille sur une copie).
        """
        self._validate(df)
        result = df.copy()

        # ── Heure ────────────────────────────────────────────────────────────
        result["dep_hour"]          = result["dep_minutes"] // 60
        result["dep_minute_of_day"] = result["dep_minutes"]

        # ── Jour de la semaine ────────────────────────────────────────────────
        result["day_of_week"] = result["service_date"].apply(
            lambda d: d.weekday()
        )

        # ── Week-end ──────────────────────────────────────────────────────────
        result["is_weekend"] = result["day_of_week"] >= 5

        # ── Vacances scolaires zone B — lookup O(1) dans le set précalculé ───
        result["is_vacances"] = result["service_date"].apply(
            lambda d: _to_date(d) in self._vacances_set
        )

        # ── Jours fériés Alsace-Moselle — lookup O(1) dans le set précalculé ─
        result["is_jour_ferie"] = result["service_date"].apply(
            lambda d: _to_date(d) in self._feries_set
        )

        # ── Heures de pointe ──────────────────────────────────────────────────
        is_weekday      = ~result["is_weekend"]
        is_morning_peak = (
            (result["dep_hour"] >= self.PEAK_MORNING_START) &
            (result["dep_hour"] <  self.PEAK_MORNING_END)
        )
        is_evening_peak = (
            (result["dep_hour"] >= self.PEAK_EVENING_START) &
            (result["dep_hour"] <  self.PEAK_EVENING_END)
        )
        result["is_peak_hour"] = is_weekday & (is_morning_peak | is_evening_peak)

        return result

    # ── Méthodes privées ──────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        """Vérifie que les colonnes requises sont présentes. Fail-fast."""
        required = {"dep_minutes", "service_date"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"[TemporalFeatureTransformer] Colonnes manquantes : {missing}\n"
                f"Colonnes présentes : {list(df.columns)}"
            )


# ── Fonctions utilitaires module-level ────────────────────────────────────────

def _build_vacances_set() -> set[date]:
    """
    Construit un set de toutes les dates de vacances individuelles
    à partir de la liste de périodes dans constants.py.

    Recherche en O(1) : 'd in vacances_set' est instantané même avec
    des milliers de tronçons.
    """
    vacances_dates: set[date] = set()
    for debut, fin in VACANCES_ZONE_B_2024_2025:
        current = debut
        while current <= fin:
            vacances_dates.add(current)
            current += timedelta(days=1)
    return vacances_dates


def _to_date(value) -> date:
    """
    Normalise une valeur en date Python natif.
    Gère les deux cas : pandas Timestamp et date Python natif.
    """
    if isinstance(value, date):
        return value
    return value.date()