# tests/unit/test_laf_loader.py
import pytest
import pandas as pd
from pathlib import Path

from services.ml_engine.data.laf.loader import LAFLoader


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# Reproduisent fidèlement les vrais fichiers SNCF :
#   - Nommage réel : Extract_DATA_GE_{TAG}_{dates}_{extract_date}.csv
#   - Encoding     : latin-1 (caractères accentués dans les motifs de refus)
#   - CC séparateur ';', SC séparateur ',', PV séparateur ','
#   - Vrais statuts : ACCEPTED, REFUSED (pas IRREGULAR, FRAUD, EVADER)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def laf_dir(tmp_path):
    """Dossier LAF temporaire propre pour chaque test."""
    return tmp_path


@pytest.fixture
def loader(laf_dir, monkeypatch):
    """LAFLoader pointant vers le dossier temporaire."""
    from shared import config as cfg
    monkeypatch.setattr(cfg.config, "LAF_DIR", laf_dir)
    return LAFLoader()


@pytest.fixture
def cc_file(laf_dir):
    """
    Fichier CC synthétique avec nommage réel et données représentatives.
    Séparateur ';', encoding latin-1, statuts ACCEPTED et REFUSED.
    """
    header = ";".join(LAFLoader.CC_COLUMNS)
    # Ligne 1 : contrôle accepté (titre valide)
    ligne1 = (
        "1187;839559;25/03/2022;22888;F;SNCF_TER;null;null;null;TICKET;PT00;;1;;;;"
        "null;;null;;OD;null;DIGITAL_TER;25/03/2022;25/03/2022;DIGITAL_TER_TICKET;"
        "null;25/03/2022 23:09;ACCEPTED;OK"
    )
    # Ligne 2 : contrôle refusé (titre périmé) → IRRÉGULARITÉ
    ligne2 = (
        "1187;839559;25/03/2022;22888;F;SNCF_TER;null;null;null;TICKET;JE25;;1;;;;"
        "null;;null;;OD;null;DIGITAL_TER;25/03/2022;25/03/2022;DIGITAL_TER_TICKET;"
        "null;25/03/2022 23:08;REFUSED;Titre périmé"
    )
    # Ligne 3 : autre train, accepté
    ligne3 = (
        "1187;88781;24/03/2022;22888;F;SNCF_TER;null;null;null;PASS;UE48;;1;;;;"
        "null;;null;;OD;null;DIGITAL_TER;01/03/2022;31/03/2022;DIGITAL_TER_TICKET;"
        "null;24/03/2022 22:57;ACCEPTED;OK"
    )
    content = "\n".join([header, ligne1, ligne2, ligne3])
    filepath = laf_dir / "Extract_DATA_GE_CC_202201-202602_20260320.csv"
    filepath.write_text(content, encoding="latin-1")
    return filepath


@pytest.fixture
def sc_file(laf_dir):
    """
    Fichier SC synthétique avec nommage réel.
    Même colonnes que CC mais séparateur ','.
    """
    header = ",".join(LAFLoader.CC_COLUMNS)
    ligne1 = (
        "1187,839559,2022-03-25,22888,F,SNCF_TER,null,null,null,TICKET,PT00,null,1,null,null,"
        "2022-03-25T00:00:00Z,null,null,null,null,OD,null,DIGITAL_TER,2022-03-25,2022-03-25,"
        "DIGITAL_TER_TICKET,null,2022-03-25T22:09:15Z,ACCEPTED,OK"
    )
    ligne2 = (
        "1187,839559,2022-03-25,22888,F,SNCF_TER,null,null,null,TICKET,JE25,null,1,null,null,"
        "2022-03-25T00:00:00Z,null,null,null,null,OD,null,DIGITAL_TER,2022-03-25,2022-03-25,"
        "DIGITAL_TER_TICKET,null,2022-03-25T22:08:21Z,REFUSED,Titre périmé"
    )
    content = "\n".join([header, ligne1, ligne2])
    filepath = laf_dir / "Extract_DATA_GE_SC_202201-202203_20260320.csv"
    filepath.write_text(content, encoding="latin-1")
    return filepath


@pytest.fixture
def pv_file(laf_dir):
    """Fichier PV synthétique avec nommage réel, séparateur ','."""
    header = ",".join(LAFLoader.PV_COLUMNS)
    ligne1 = (
        "1187,F,832859,2023-05-12,22888,MULHOUSE VILLE,87182063,WESSERLING,87182618,"
        "2023-05-12T03:17:24.854Z,WILLER SUR THUR,87182584,MULHOUSE VILLE,87182063,"
        "26,0,5000,5000,EUR,SECOND,TARIFF,ISSUED,null,TER,"
        "Sans titre de transport,18,STANDARD,false,false,VERBAL_REPORT"
    )
    ligne2 = (
        "1187,F,96230,2023-05-11,22888,STRASBOURG,87212027,BASEL SBB,85000109,"
        "2023-05-11T14:32:55.416Z,MULHOUSE VILLE,87182063,STRASBOURG,87212027,"
        "106,0,5000,5000,EUR,SECOND,NON_TARIFF,ISSUED,null,TER,"
        "Comportement,19,STANDARD,false,false,DOCUMENT_WITH_PHOTO"
    )
    content = "\n".join([header, ligne1, ligne2])
    filepath = laf_dir / "Extract_DATA_GE_PV_202201-202602_20260320.csv"
    filepath.write_text(content, encoding="latin-1")
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLAFLoader:

    # ── load_cc() ─────────────────────────────────────────────────────────────

    def test_load_cc_retourne_dataframe(self, loader, cc_file):
        df = loader.load_cc()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_cc_toutes_colonnes_presentes(self, loader, cc_file):
        """Toutes les colonnes du schéma CC_COLUMNS doivent être présentes."""
        df = loader.load_cc()
        for col in LAFLoader.CC_COLUMNS:
            assert col in df.columns, f"Colonne manquante : {col}"

    def test_load_cc_nb_lignes_correct(self, loader, cc_file):
        assert len(loader.load_cc()) == 3

    def test_load_cc_vrais_statuts(self, loader, cc_file):
        """Les vrais statuts ACCEPTED et REFUSED doivent être présents."""
        df = loader.load_cc()
        statuts = set(df["verifiedTickets_verificationStatus"].unique())
        assert "ACCEPTED" in statuts
        assert "REFUSED" in statuts
        # Ces statuts NE DOIVENT PAS apparaître dans les données réelles
        assert "IRREGULAR" not in statuts
        assert "FRAUD" not in statuts

    def test_load_cc_absent_retourne_dataframe_vide(self, loader):
        """Sans fichier CC, retourne un DataFrame vide avec le bon schéma."""
        df = loader.load_cc()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        for col in LAFLoader.CC_COLUMNS:
            assert col in df.columns

    def test_load_cc_encoding_latin1(self, loader, cc_file):
        """
        Les caractères accentués dans les motifs de refus doivent être lisibles.
        'Titre périmé' doit être décodé correctement depuis latin-1.
        """
        df = loader.load_cc()
        raisons = df["verifiedTickets_verificationStatusReason"].tolist()
        assert any("périmé" in str(r) for r in raisons), (
            "Caractères accentués non décodés — vérifier encoding latin-1"
        )

    # ── load_sc() ─────────────────────────────────────────────────────────────

    def test_load_sc_retourne_dataframe(self, loader, sc_file):
        df = loader.load_sc()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_sc_memes_colonnes_que_cc(self, loader, sc_file):
        """SC doit avoir les mêmes colonnes que CC."""
        df = loader.load_sc()
        for col in LAFLoader.CC_COLUMNS:
            assert col in df.columns

    def test_load_sc_nb_lignes_correct(self, loader, sc_file):
        assert len(loader.load_sc()) == 2

    def test_load_sc_absent_retourne_vide(self, loader):
        df = loader.load_sc()
        assert len(df) == 0
        for col in LAFLoader.CC_COLUMNS:
            assert col in df.columns

    def test_load_sc_multiple_fichiers_concatenes(self, loader, sc_file, laf_dir):
        """Plusieurs fichiers SC doivent être concaténés."""
        # Créer un second fichier SC
        header = ",".join(LAFLoader.CC_COLUMNS)
        ligne = (
            "1187,99999,2022-06-01,22888,F,SNCF_TER,null,null,null,TICKET,PT00,null,1,null,null,"
            "2022-06-01T10:00:00Z,null,null,null,null,OD,null,DIGITAL_TER,2022-06-01,2022-06-01,"
            "DIGITAL_TER_TICKET,null,2022-06-01T10:05:00Z,ACCEPTED,OK"
        )
        second = laf_dir / "Extract_DATA_GE_SC_202204-202206_20260320.csv"
        second.write_text("\n".join([header, ligne]), encoding="latin-1")

        df = loader.load_sc()
        assert len(df) == 3  # 2 du premier fichier + 1 du second

    # ── load_pv() ─────────────────────────────────────────────────────────────

    def test_load_pv_retourne_dataframe(self, loader, pv_file):
        df = loader.load_pv()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_pv_toutes_colonnes_presentes(self, loader, pv_file):
        df = loader.load_pv()
        for col in LAFLoader.PV_COLUMNS:
            assert col in df.columns, f"Colonne manquante : {col}"

    def test_load_pv_nb_lignes_correct(self, loader, pv_file):
        assert len(loader.load_pv()) == 2

    def test_load_pv_types_fraude(self, loader, pv_file):
        """TARIFF et NON_TARIFF doivent être présents."""
        df = loader.load_pv()
        offences = set(df["penalties_offenceType"].unique())
        assert "TARIFF" in offences
        assert "NON_TARIFF" in offences

    def test_load_pv_absent_retourne_vide(self, loader):
        df = loader.load_pv()
        assert len(df) == 0
        for col in LAFLoader.PV_COLUMNS:
            assert col in df.columns

    # ── _detect_separator() ───────────────────────────────────────────────────

    def test_detect_separator_point_virgule(self, loader, cc_file):
        """CC utilise ';' comme séparateur."""
        assert loader._detect_separator(cc_file) == ";"

    def test_detect_separator_virgule(self, loader, sc_file):
        """SC utilise ',' comme séparateur."""
        assert loader._detect_separator(sc_file) == ","

    # ── _find_file() ──────────────────────────────────────────────────────────

    def test_find_file_trouve_cc(self, loader, cc_file):
        result = loader._find_file("CC")
        assert result is not None
        assert result.name == cc_file.name

    def test_find_file_absent_retourne_none(self, loader):
        assert loader._find_file("CC") is None

    def test_find_file_dossier_absent_retourne_none(self, monkeypatch):
        from shared import config as cfg
        monkeypatch.setattr(cfg.config, "LAF_DIR", Path("/dossier/inexistant"))
        loader2 = LAFLoader()
        assert loader2._find_file("CC") is None

    # ── _empty_df() ───────────────────────────────────────────────────────────

    def test_empty_df_colonnes_cc(self, loader):
        df = loader._empty_df(LAFLoader.CC_COLUMNS)
        assert list(df.columns) == LAFLoader.CC_COLUMNS
        assert len(df) == 0

    def test_empty_df_colonnes_pv(self, loader):
        df = loader._empty_df(LAFLoader.PV_COLUMNS)
        assert list(df.columns) == LAFLoader.PV_COLUMNS
        assert len(df) == 0