# tests/unit/test_laf_preprocessor.py
import pytest
import pandas as pd
from datetime import date

from services.ml_engine.data.laf.preprocessor import LAFPreprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def preprocessor():
    return LAFPreprocessor()


@pytest.fixture
def cc_brut():
    """DataFrame CC brut simulant la sortie de LAFLoader.load_cc()."""
    return pd.DataFrame([
        # train 117756, tronçon Strasbourg→Sélestat, contrôle valide
        {
            "course_courseNumber":                          "117756",
            "course_departureDate":                         "2024-09-02",
            "station_uicCode":                              "87212027",
            "ticket_travelInformation_origin_uicCode":      "87212027",
            "ticket_travelInformation_destination_uicCode": "87214007",
            "ticket_travelInformation_departureDateTime":   "2024-09-02T08:23:00",
            "verifiedTickets_verificationStatus":           "VALID",
            "verifiedTickets_verificationDateTime":         "2024-09-02T08:30:00",
            "ticket_passengerCount":                        "1",
        },
        # même train, irrégulier
        {
            "course_courseNumber":                          "117756",
            "course_departureDate":                         "2024-09-02",
            "station_uicCode":                              "87212027",
            "ticket_travelInformation_origin_uicCode":      "87212027",
            "ticket_travelInformation_destination_uicCode": "87214007",
            "ticket_travelInformation_departureDateTime":   "2024-09-02T08:23:00",
            "verifiedTickets_verificationStatus":           "IRREGULAR",
            "verifiedTickets_verificationDateTime":         "2024-09-02T08:31:00",
            "ticket_passengerCount":                        "1",
        },
        # train 117758, tronçon différent, valide
        {
            "course_courseNumber":                          "117758",
            "course_departureDate":                         "2024-09-02",
            "station_uicCode":                              "87214007",
            "ticket_travelInformation_origin_uicCode":      "87214007",
            "ticket_travelInformation_destination_uicCode": "87182063",
            "ticket_travelInformation_departureDateTime":   "2024-09-02T09:05:00",
            "verifiedTickets_verificationStatus":           "VALID",
            "verifiedTickets_verificationDateTime":         "2024-09-02T09:10:00",
            "ticket_passengerCount":                        "1",
        },
        # ligne avec uicCode manquant → doit être filtrée
        {
            "course_courseNumber":                          "117760",
            "course_departureDate":                         "2024-09-02",
            "station_uicCode":                              None,
            "ticket_travelInformation_origin_uicCode":      None,
            "ticket_travelInformation_destination_uicCode": None,
            "ticket_travelInformation_departureDateTime":   "2024-09-02T10:00:00",
            "verifiedTickets_verificationStatus":           "VALID",
            "verifiedTickets_verificationDateTime":         "2024-09-02T10:05:00",
            "ticket_passengerCount":                        "1",
        },
    ])


@pytest.fixture
def pv_brut():
    """DataFrame PV brut simulant la sortie de LAFLoader.load_pv()."""
    return pd.DataFrame([
        {
            "course_courseNumber":           "117756",
            "course_departureDate":          "2024-09-02",
            "course_origin_uicCode":         "87212027",
            "course_destination_uicCode":    "87214007",
            "penalties_issueDateTime":       "2024-09-02T08:35:00",
            "penalties_origin_uicCode":      "87212027",
            "penalties_destination_uicCode": "87214007",
            "penalties_offenceType":         "FARE_EVASION",
            "penalties_penaltyStatus":       "ISSUED",
            "penalties_perceptionFee_amountInCents": "5000",
            "penalties_flatFee_amountInCents":       "0",
        },
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLAFPreprocessor:

    # ── clean_cc() ────────────────────────────────────────────────────────────

    def test_clean_cc_supprime_lignes_sans_uic(self, preprocessor, cc_brut):
        """Les lignes avec UIC manquant doivent être supprimées."""
        result = preprocessor.clean_cc(cc_brut)
        assert result["ticket_travelInformation_origin_uicCode"].notna().all()

    def test_clean_cc_parse_dates(self, preprocessor, cc_brut):
        """Les colonnes de dates doivent être converties en datetime."""
        result = preprocessor.clean_cc(cc_brut)
        assert pd.api.types.is_datetime64_any_dtype(
            result["verifiedTickets_verificationDateTime"]
        )

    def test_clean_cc_ajoute_troncon_id(self, preprocessor, cc_brut):
        """
        clean_cc() doit ajouter une colonne troncon_id.
        Format : {origin_uic}_{dest_uic}_{dep_hour}
        """
        result = preprocessor.clean_cc(cc_brut)
        assert "troncon_id" in result.columns

    def test_clean_cc_troncon_id_format(self, preprocessor, cc_brut):
        """troncon_id doit avoir le bon format pour le tronçon 87212027→87214007 à 8h."""
        result = preprocessor.clean_cc(cc_brut)
        # 08:23 → heure = 8
        expected = "87212027_87214007_8"
        assert expected in result["troncon_id"].values

    def test_clean_cc_input_non_modifie(self, preprocessor, cc_brut):
        """clean_cc() ne doit pas modifier le DataFrame d'entrée."""
        colonnes_avant = list(cc_brut.columns)
        preprocessor.clean_cc(cc_brut)
        assert list(cc_brut.columns) == colonnes_avant

    # ── clean_pv() ────────────────────────────────────────────────────────────

    def test_clean_pv_ajoute_troncon_id(self, preprocessor, pv_brut):
        """clean_pv() doit ajouter troncon_id basé sur origin/destination."""
        result = preprocessor.clean_pv(pv_brut)
        assert "troncon_id" in result.columns

    def test_clean_pv_parse_issue_datetime(self, preprocessor, pv_brut):
        """penalties_issueDateTime doit être converti en datetime."""
        result = preprocessor.clean_pv(pv_brut)
        assert pd.api.types.is_datetime64_any_dtype(result["penalties_issueDateTime"])

    # ── build_troncon_stats() ─────────────────────────────────────────────────

    def test_build_troncon_stats_retourne_dataframe(self, preprocessor, cc_brut):
        """build_troncon_stats() doit retourner un DataFrame agrégé."""
        cc_clean = preprocessor.clean_cc(cc_brut)
        stats = preprocessor.build_troncon_stats(cc_clean)
        assert isinstance(stats, pd.DataFrame)

    def test_build_troncon_stats_colonnes_presentes(self, preprocessor, cc_brut):
        """Les colonnes de features historiques doivent être présentes."""
        cc_clean = preprocessor.clean_cc(cc_brut)
        stats = preprocessor.build_troncon_stats(cc_clean)
        colonnes_attendues = [
            "troncon_id", "nb_controles", "nb_irregularites",
            "taux_irregularite", "derniere_date_controle",
        ]
        for col in colonnes_attendues:
            assert col in stats.columns, f"Colonne manquante dans stats : {col}"

    def test_build_troncon_stats_taux_calcul(self, preprocessor, cc_brut):
        """
        Pour 87212027→87214007 : 2 contrôles, 1 IRREGULAR
        → taux_irregularite = 0.5
        """
        cc_clean = preprocessor.clean_cc(cc_brut)
        stats = preprocessor.build_troncon_stats(cc_clean)
        row = stats[stats["troncon_id"] == "87212027_87214007_8"].iloc[0]
        assert row["nb_controles"] == 2
        assert row["nb_irregularites"] == 1
        assert row["taux_irregularite"] == pytest.approx(0.5)

    def test_build_troncon_stats_une_ligne_par_troncon(self, preprocessor, cc_brut):
        """Chaque troncon_id ne doit apparaître qu'une fois dans les stats."""
        cc_clean = preprocessor.clean_cc(cc_brut)
        stats = preprocessor.build_troncon_stats(cc_clean)
        assert stats["troncon_id"].nunique() == len(stats)