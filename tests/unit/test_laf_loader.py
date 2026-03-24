# tests/unit/test_laf_loader.py
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import io

from services.ml_engine.data.laf.loader import LAFLoader


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — données synthétiques calquées sur les vraies colonnes
# ─────────────────────────────────────────────────────────────────────────────

CC_COLONNES = [
    "course_companyCode", "course_courseNumber", "course_departureDate",
    "course_operatingCarrierCode", "course_transportationMode",
    "verifiedTickets_missionActivity", "station_label", "station_uicCode",
    "ticket_contractTariff", "ticket_contractType", "ticket_fareCode",
    "ticket_networkId", "ticket_passengerCount", "ticket_provider",
    "ticket_travelInformation_courseNumber",
    "ticket_travelInformation_departureDateTime",
    "ticket_travelInformation_destination_label",
    "ticket_travelInformation_destination_uicCode",
    "ticket_travelInformation_origin_label",
    "ticket_travelInformation_origin_uicCode",
    "ticket_travelInformation_spaceValidity",
    "ticket_travelInformation_travelClass",
    "ticket_type", "ticket_validFrom", "ticket_validTo",
    "ticketDocument_documentType", "verifiedTickets_type",
    "verifiedTickets_verificationDateTime",
    "verifiedTickets_verificationStatus",
    "verifiedTickets_verificationStatusReason",
]

PV_COLONNES = [
    "course_companyCode", "course_transportationMode",
    "course_courseNumber", "course_departureDate",
    "course_operatingCarrierCode",
    "course_destination_label", "course_destination_uicCode",
    "course_origin_label", "course_origin_uicCode",
    "penalties_issueDateTime",
    "penalties_origin_label", "penalties_origin_uicCode",
    "penalties_destination_label", "penalties_destination_uicCode",
    "penalties_travelDistance",
    "penalties_perceptionFee_amountInCents",
    "penalties_flatFee_amountInCents",
    "penalties_applicationFee_amountInCents",
    "penalties_applicationFee_currency",
    "penalties_travelClass", "penalties_offenceType",
    "penalties_penaltyStatus", "penalties_identityStatementStatus",
    "penalties_trafficCode", "penalties_offenceLabel",
    "penalties_penaltyCode", "penalties_penaltyType",
    "penalties_ticketAttached", "penalties_otherAttachment",
    "penalties_offender_identityDocument_documentType",
]


def _make_cc_csv(n: int = 5) -> str:
    """Génère un CSV CC synthétique avec n lignes."""
    rows = []
    for i in range(n):
        status = "VALID" if i % 3 != 0 else "IRREGULAR"
        row = {col: "" for col in CC_COLONNES}
        row.update({
            "course_courseNumber":              f"1177{i:02d}",
            "course_departureDate":             "2024-09-02",
            "station_uicCode":                  "87212027",
            "ticket_travelInformation_origin_uicCode":      "87212027",
            "ticket_travelInformation_destination_uicCode": "87214007",
            "ticket_travelInformation_departureDateTime":   "2024-09-02T08:23:00",
            "verifiedTickets_verificationStatus":           status,
            "verifiedTickets_verificationDateTime":         "2024-09-02T08:30:00",
            "ticket_passengerCount":            "1",
        })
        rows.append(row)
    lines = [";".join(CC_COLONNES)]
    for r in rows:
        lines.append(";".join(str(r[c]) for c in CC_COLONNES))
    return "\n".join(lines)


def _make_pv_csv(n: int = 3) -> str:
    """Génère un CSV PV synthétique avec n lignes."""
    rows = []
    for i in range(n):
        row = {col: "" for col in PV_COLONNES}
        row.update({
            "course_courseNumber":           f"1177{i:02d}",
            "course_departureDate":          "2024-09-02",
            "course_origin_uicCode":         "87212027",
            "course_destination_uicCode":    "87214007",
            "penalties_issueDateTime":       "2024-09-02T08:35:00",
            "penalties_origin_uicCode":      "87212027",
            "penalties_destination_uicCode": "87214007",
            "penalties_offenceType":         "FARE_EVASION",
            "penalties_penaltyStatus":       "ISSUED",
            "penalties_perceptionFee_amountInCents": "5000",
        })
        rows.append(row)
    lines = [",".join(PV_COLONNES)]
    for r in rows:
        lines.append(",".join(str(r[c]) for c in PV_COLONNES))
    return "\n".join(lines)


@pytest.fixture
def laf_dir(tmp_path):
    """Crée un dossier LAF temporaire avec les fichiers CSV synthétiques."""
    laf = tmp_path / "laf"
    laf.mkdir()
    (laf / "cc_2022_2026.csv").write_text(_make_cc_csv(10), encoding="utf-8")
    (laf / "pv_2022_2026.csv").write_text(_make_pv_csv(5), encoding="utf-8")
    # SCAN : même structure que CC, fichiers trimestriels
    (laf / "scan_2024_Q1.csv").write_text(_make_cc_csv(8), encoding="utf-8")
    (laf / "scan_2024_Q2.csv").write_text(_make_cc_csv(6), encoding="utf-8")
    return laf


@pytest.fixture
def loader(laf_dir, monkeypatch):
    from shared import config as cfg
    monkeypatch.setattr(cfg.config, "LAF_DIR", laf_dir)
    return LAFLoader()


# ─────────────────────────────────────────────────────────────────────────────
# Tests LAFLoader
# ─────────────────────────────────────────────────────────────────────────────

class TestLAFLoader:

    # ── CC ────────────────────────────────────────────────────────────────────

    def test_load_cc_retourne_dataframe(self, loader):
        """load_cc() doit retourner un DataFrame non vide."""
        df = loader.load_cc()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_cc_colonnes_presentes(self, loader):
        """Les colonnes clés du CC doivent être présentes."""
        df = loader.load_cc()
        colonnes_cles = [
            "course_courseNumber",
            "course_departureDate",
            "station_uicCode",
            "ticket_travelInformation_origin_uicCode",
            "ticket_travelInformation_destination_uicCode",
            "verifiedTickets_verificationStatus",
            "verifiedTickets_verificationDateTime",
        ]
        for col in colonnes_cles:
            assert col in df.columns, f"Colonne manquante dans CC : {col}"

    def test_load_cc_fichier_absent_leve_erreur(self, loader, laf_dir):
        """Si le fichier CC est absent, FileNotFoundError doit être levée."""
        (laf_dir / "cc_2022_2026.csv").unlink()
        with pytest.raises(FileNotFoundError, match="cc_2022_2026.csv"):
            loader.load_cc()

    # ── PV ────────────────────────────────────────────────────────────────────

    def test_load_pv_retourne_dataframe(self, loader):
        df = loader.load_pv()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_pv_colonnes_presentes(self, loader):
        df = loader.load_pv()
        colonnes_cles = [
            "course_courseNumber",
            "course_departureDate",
            "course_origin_uicCode",
            "course_destination_uicCode",
            "penalties_issueDateTime",
            "penalties_offenceType",
            "penalties_penaltyStatus",
        ]
        for col in colonnes_cles:
            assert col in df.columns, f"Colonne manquante dans PV : {col}"

    def test_load_pv_fichier_absent_leve_erreur(self, loader, laf_dir):
        (laf_dir / "pv_2022_2026.csv").unlink()
        with pytest.raises(FileNotFoundError, match="pv_2022_2026.csv"):
            loader.load_pv()

    # ── SCAN ──────────────────────────────────────────────────────────────────

    def test_load_scan_concatene_tous_les_fichiers(self, loader):
        """load_scan() doit concaténer tous les fichiers scan_*.csv."""
        df = loader.load_scan()
        # 8 + 6 = 14 lignes dans nos fixtures
        assert len(df) == 14

    def test_load_scan_sans_fichiers_leve_erreur(self, loader, laf_dir):
        """Si aucun fichier SCAN n'est trouvé, ValueError doit être levée."""
        for f in laf_dir.glob("scan_*.csv"):
            f.unlink()
        with pytest.raises(ValueError, match="Aucun fichier SCAN"):
            loader.load_scan()

    def test_load_scan_colonnes_identiques_cc(self, loader):
        """Les fichiers SCAN ont la même structure que CC."""
        df_scan = loader.load_scan()
        df_cc   = loader.load_cc()
        # Les colonnes doivent être identiques (même structure SCAN et CC)
        assert set(df_scan.columns) == set(df_cc.columns)

    # ── Comportement commun ───────────────────────────────────────────────────

    def test_load_cc_separateur_point_virgule(self, loader):
        """CC utilise ';' comme séparateur — vérifier que le parsing est correct."""
        df = loader.load_cc()
        # Si le séparateur était mal détecté, on aurait 1 seule colonne
        assert len(df.columns) > 5

    def test_load_pv_separateur_virgule(self, loader):
        """PV utilise ',' comme séparateur."""
        df = loader.load_pv()
        assert len(df.columns) > 5