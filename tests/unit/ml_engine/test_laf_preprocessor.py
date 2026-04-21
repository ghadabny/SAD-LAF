# tests/unit/test_laf_preprocessor.py
import pytest
import pandas as pd

from services.ml_engine.data.laf.preprocessor import (
    LAFPreprocessor,
    _parse_datetime,
    _build_troncon_id,
    _build_gtfs_join_key,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# IMPORTANT : les statuts utilisent les VRAIES valeurs des données réelles
#   ACCEPTED → titre valide
#   REFUSED  → irrégularité (pas IRREGULAR, FRAUD ou EVADER)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def preprocessor():
    return LAFPreprocessor()


@pytest.fixture
def cc_brut():
    """
    DataFrame CC brut simulant la sortie de LAFLoader.load_cc().
    Statuts réels : ACCEPTED et REFUSED.
    Dates au format français DD/MM/YYYY (format vu dans les vrais fichiers CC).
    """
    return pd.DataFrame([
        # train 839559, tronçon Strasbourg→Sélestat, contrôle ACCEPTÉ
        {
            "course_courseNumber":                          "839559",
            "course_departureDate":                         "25/03/2022",
            "station_uicCode":                              None,
            "ticket_travelInformation_origin_uicCode":      "87212027",
            "ticket_travelInformation_destination_uicCode": "87214007",
            "ticket_travelInformation_departureDateTime":   "25/03/2022 23:09",
            "verifiedTickets_verificationStatus":           "ACCEPTED",
            "verifiedTickets_verificationDateTime":         "25/03/2022 23:09",
            "verifiedTickets_verificationStatusReason":     "OK",
            "ticket_passengerCount":                        "1",
        },
        # même train, même tronçon, REFUSÉ → irrégularité
        {
            "course_courseNumber":                          "839559",
            "course_departureDate":                         "25/03/2022",
            "station_uicCode":                              None,
            "ticket_travelInformation_origin_uicCode":      "87212027",
            "ticket_travelInformation_destination_uicCode": "87214007",
            "ticket_travelInformation_departureDateTime":   "25/03/2022 23:08",
            "verifiedTickets_verificationStatus":           "REFUSED",
            "verifiedTickets_verificationDateTime":         "25/03/2022 23:08",
            "verifiedTickets_verificationStatusReason":     "Titre périmé",
            "ticket_passengerCount":                        "1",
        },
        # autre train, tronçon Sélestat→Mulhouse à 22h, ACCEPTÉ
        {
            "course_courseNumber":                          "88781",
            "course_departureDate":                         "24/03/2022",
            "station_uicCode":                              None,
            "ticket_travelInformation_origin_uicCode":      "87214007",
            "ticket_travelInformation_destination_uicCode": "87182063",
            "ticket_travelInformation_departureDateTime":   "24/03/2022 22:57",
            "verifiedTickets_verificationStatus":           "ACCEPTED",
            "verifiedTickets_verificationDateTime":         "24/03/2022 22:57",
            "verifiedTickets_verificationStatusReason":     "OK",
            "ticket_passengerCount":                        "1",
        },
        # ligne avec UIC manquant → doit être filtrée
        {
            "course_courseNumber":                          "99999",
            "course_departureDate":                         "25/03/2022",
            "station_uicCode":                              None,
            "ticket_travelInformation_origin_uicCode":      None,
            "ticket_travelInformation_destination_uicCode": None,
            "ticket_travelInformation_departureDateTime":   "25/03/2022 10:00",
            "verifiedTickets_verificationStatus":           "ACCEPTED",
            "verifiedTickets_verificationDateTime":         "25/03/2022 10:00",
            "verifiedTickets_verificationStatusReason":     "OK",
            "ticket_passengerCount":                        "1",
        },
    ])


@pytest.fixture
def pv_brut():
    """DataFrame PV brut simulant la sortie de LAFLoader.load_pv()."""
    return pd.DataFrame([
        {
            "course_courseNumber":                      "832859",
            "course_departureDate":                     "2023-05-12",
            "course_origin_uicCode":                    "87182618",
            "course_destination_uicCode":               "87182063",
            "penalties_issueDateTime":                  "2023-05-12T03:17:24.854Z",
            "penalties_origin_uicCode":                 "87182584",
            "penalties_destination_uicCode":            "87182063",
            "penalties_offenceType":                    "TARIFF",
            "penalties_penaltyStatus":                  "ISSUED",
            "penalties_perceptionFee_amountInCents":    "0",
            "penalties_flatFee_amountInCents":          "5000",
        },
        {
            "course_courseNumber":                      "96230",
            "course_departureDate":                     "2023-05-11",
            "course_origin_uicCode":                    "85000109",
            "course_destination_uicCode":               "87212027",
            "penalties_issueDateTime":                  "2023-05-11T14:32:55.416Z",
            "penalties_origin_uicCode":                 "87182063",
            "penalties_destination_uicCode":            "87212027",
            "penalties_offenceType":                    "NON_TARIFF",
            "penalties_penaltyStatus":                  "ISSUED",
            "penalties_perceptionFee_amountInCents":    "0",
            "penalties_flatFee_amountInCents":          "5000",
        },
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tests _parse_datetime() — fonction utilitaire
# ─────────────────────────────────────────────────────────────────────────────

class TestParseDatetime:

    def test_format_iso(self):
        """Format ISO 8601 standard."""
        s = pd.Series(["2022-03-25"])
        result = _parse_datetime(s)
        assert result.iloc[0].year == 2022
        assert result.iloc[0].month == 3
        assert result.iloc[0].day == 25

    def test_format_iso_avec_heure(self):
        """Format ISO avec heure et timezone."""
        s = pd.Series(["2023-05-12T03:17:24.854Z"])
        result = _parse_datetime(s)
        assert result.iloc[0].year == 2023
        assert result.iloc[0].month == 5
        assert result.iloc[0].day == 12
        assert result.iloc[0].hour == 3

    def test_format_francais(self):
        """Format français DD/MM/YYYY — dayfirst requis."""
        s = pd.Series(["25/03/2022"])
        result = _parse_datetime(s)
        assert result.iloc[0].day == 25
        assert result.iloc[0].month == 3
        assert result.iloc[0].year == 2022

    def test_format_francais_avec_heure(self):
        """Format français avec heure DD/MM/YYYY HH:MM."""
        s = pd.Series(["25/03/2022 23:09"])
        result = _parse_datetime(s)
        assert result.iloc[0].hour == 23
        assert result.iloc[0].day == 25

    def test_formats_mixtes(self):
        """Les deux formats dans la même série — tous parsés correctement."""
        s = pd.Series(["25/03/2022", "2023-05-12"])
        result = _parse_datetime(s)
        assert result.iloc[0].day == 25
        assert result.iloc[0].month == 3
        assert result.iloc[1].month == 5
        assert result.iloc[1].day == 12

    def test_valeur_invalide_retourne_nat(self):
        """Une valeur invalide doit retourner NaT, pas planter."""
        s = pd.Series(["invalide"])
        result = _parse_datetime(s)
        assert pd.isna(result.iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# Tests clean_cc()
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanCC:

    def test_supprime_lignes_sans_uic(self, preprocessor, cc_brut):
        """Les lignes avec UIC origin/destination manquant doivent être filtrées."""
        result = preprocessor.clean_cc(cc_brut)
        assert result["ticket_travelInformation_origin_uicCode"].notna().all()
        assert result["ticket_travelInformation_destination_uicCode"].notna().all()

    def test_nb_lignes_apres_filtrage(self, preprocessor, cc_brut):
        """3 lignes valides sur 4 (la 4ème n'a pas d'UIC)."""
        result = preprocessor.clean_cc(cc_brut)
        assert len(result) == 3

    def test_parse_verification_datetime(self, preprocessor, cc_brut):
        """verifiedTickets_verificationDateTime doit être en datetime64."""
        result = preprocessor.clean_cc(cc_brut)
        assert pd.api.types.is_datetime64_any_dtype(
            result["verifiedTickets_verificationDateTime"]
        )

    def test_ajoute_troncon_id(self, preprocessor, cc_brut):
        """troncon_id doit être présent après clean_cc()."""
        result = preprocessor.clean_cc(cc_brut)
        assert "troncon_id" in result.columns

    def test_troncon_id_format(self, preprocessor, cc_brut):
        """
        troncon_id = "{origin}_{dest}_{heure}"
        Pour 87212027→87214007 à 23h : "87212027_87214007_23"
        """
        result = preprocessor.clean_cc(cc_brut)
        assert "87212027_87214007_23" in result["troncon_id"].values

    def test_ajoute_gtfs_join_key(self, preprocessor, cc_brut):
        """gtfs_join_key doit être présent après clean_cc()."""
        result = preprocessor.clean_cc(cc_brut)
        assert "gtfs_join_key" in result.columns

    def test_gtfs_join_key_format_date_francaise(self, preprocessor, cc_brut):
        """
        gtfs_join_key = "{course_courseNumber}_{YYYYMMDD}"
        Pour train 839559 le 25/03/2022 : "839559_20220325"
        """
        result = preprocessor.clean_cc(cc_brut)
        assert "839559_20220325" in result["gtfs_join_key"].values

    def test_input_non_modifie(self, preprocessor, cc_brut):
        """clean_cc() ne doit pas modifier le DataFrame d'entrée."""
        cols_avant = list(cc_brut.columns)
        preprocessor.clean_cc(cc_brut)
        assert list(cc_brut.columns) == cols_avant

    def test_dataframe_vide_retourne_vide(self, preprocessor):
        """Un DataFrame vide en entrée doit retourner un DataFrame vide."""
        result = preprocessor.clean_cc(pd.DataFrame())
        assert result.empty


# ─────────────────────────────────────────────────────────────────────────────
# Tests clean_pv()
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanPV:

    def test_ajoute_troncon_id(self, preprocessor, pv_brut):
        result = preprocessor.clean_pv(pv_brut)
        assert "troncon_id" in result.columns

    def test_parse_issue_datetime(self, preprocessor, pv_brut):
        result = preprocessor.clean_pv(pv_brut)
        assert pd.api.types.is_datetime64_any_dtype(result["penalties_issueDateTime"])

    def test_ajoute_gtfs_join_key(self, preprocessor, pv_brut):
        result = preprocessor.clean_pv(pv_brut)
        assert "gtfs_join_key" in result.columns

    def test_gtfs_join_key_format_date_iso(self, preprocessor, pv_brut):
        """
        Pour train 832859 le 2023-05-12 (format ISO) : "832859_20230512"
        """
        result = preprocessor.clean_pv(pv_brut)
        assert "832859_20230512" in result["gtfs_join_key"].values

    def test_nb_lignes_inchange(self, preprocessor, pv_brut):
        """Les 2 lignes PV sont valides et doivent être conservées."""
        result = preprocessor.clean_pv(pv_brut)
        assert len(result) == 2

    def test_dataframe_vide_retourne_vide(self, preprocessor):
        result = preprocessor.clean_pv(pd.DataFrame())
        assert result.empty


# ─────────────────────────────────────────────────────────────────────────────
# Tests build_troncon_stats()
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildTronconStats:

    def test_retourne_dataframe(self, preprocessor, cc_brut):
        cc_clean = preprocessor.clean_cc(cc_brut)
        stats = preprocessor.build_troncon_stats(cc_clean)
        assert isinstance(stats, pd.DataFrame)

    def test_colonnes_presentes(self, preprocessor, cc_brut):
        cc_clean = preprocessor.clean_cc(cc_brut)
        stats = preprocessor.build_troncon_stats(cc_clean)
        for col in ["troncon_id", "nb_controles", "nb_irregularites",
                    "taux_irregularite", "derniere_date_controle"]:
            assert col in stats.columns, f"Colonne manquante : {col}"

    def test_taux_calcul_avec_refused(self, preprocessor, cc_brut):
        """
        Tronçon 87212027_87214007_23 : 2 contrôles, 1 REFUSED
        → taux_irregularite = 0.5
        Le statut REFUSED doit être reconnu comme irrégularité.
        """
        cc_clean = preprocessor.clean_cc(cc_brut)
        stats = preprocessor.build_troncon_stats(cc_clean)
        row = stats[stats["troncon_id"] == "87212027_87214007_23"].iloc[0]
        assert row["nb_controles"] == 2
        assert row["nb_irregularites"] == 1
        assert row["taux_irregularite"] == pytest.approx(0.5)

    def test_troncon_accepte_uniquement_taux_zero(self, preprocessor, cc_brut):
        """Le tronçon avec seulement un ACCEPTED doit avoir taux = 0."""
        cc_clean = preprocessor.clean_cc(cc_brut)
        stats = preprocessor.build_troncon_stats(cc_clean)
        row = stats[stats["troncon_id"] == "87214007_87182063_22"].iloc[0]
        assert row["nb_irregularites"] == 0
        assert row["taux_irregularite"] == 0.0

    def test_cc_vide_retourne_vide(self, preprocessor):
        """Un CC vide doit retourner un DataFrame vide avec les bonnes colonnes."""
        stats = preprocessor.build_troncon_stats(pd.DataFrame())
        assert len(stats) == 0
        assert "troncon_id" in stats.columns


# ─────────────────────────────────────────────────────────────────────────────
# Tests build_pv_stats()
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPVStats:

    def test_retourne_dataframe(self, preprocessor, pv_brut):
        pv_clean = preprocessor.clean_pv(pv_brut)
        stats = preprocessor.build_pv_stats(pv_clean)
        assert isinstance(stats, pd.DataFrame)

    def test_colonnes_presentes(self, preprocessor, pv_brut):
        pv_clean = preprocessor.clean_pv(pv_brut)
        stats = preprocessor.build_pv_stats(pv_clean)
        for col in ["troncon_id", "nb_pv", "nb_pv_tariff",
                    "nb_pv_non_tariff", "montant_moyen_pv_cents"]:
            assert col in stats.columns

    def test_nb_pv_correct(self, preprocessor, pv_brut):
        """2 PV dans les données → 2 tronçons distincts → 1 PV chacun."""
        pv_clean = preprocessor.clean_pv(pv_brut)
        stats = preprocessor.build_pv_stats(pv_clean)
        assert stats["nb_pv"].sum() == 2

    def test_tariff_vs_non_tariff(self, preprocessor, pv_brut):
        """1 TARIFF + 1 NON_TARIFF dans les données."""
        pv_clean = preprocessor.clean_pv(pv_brut)
        stats = preprocessor.build_pv_stats(pv_clean)
        assert stats["nb_pv_tariff"].sum() == 1
        assert stats["nb_pv_non_tariff"].sum() == 1

    def test_montant_moyen_correct(self, preprocessor, pv_brut):
        """Montant = 0 + 5000 = 5000 centimes pour chaque PV."""
        pv_clean = preprocessor.clean_pv(pv_brut)
        stats = preprocessor.build_pv_stats(pv_clean)
        assert (stats["montant_moyen_pv_cents"] == 5000.0).all()

    def test_pv_vide_retourne_vide(self, preprocessor):
        stats = preprocessor.build_pv_stats(pd.DataFrame())
        assert len(stats) == 0
        assert "troncon_id" in stats.columns