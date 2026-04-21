# tests/unit/test_formatter.py
"""
Tests unitaires de TourneeFormatter et des fonctions utilitaires.

Couvre :
    _minutes_to_hhmm()     — conversion minutes → HH:MM
    _score_to_priorite()   — conversion score → label HAUTE/MOYENNE/BASSE
    TourneeFormatter.format() — structure du DataFrame produit
    generate_tournee_id()   — unicité et format des IDs
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from services.exporter.formatter import (
    TourneeFormatter,
    _minutes_to_hhmm,
    _score_to_priorite,
    SEUIL_HAUTE,
    SEUIL_MOYENNE,
)
from shared.schemas import (
    ArcSchema,
    ArcTypeSchema,
    NodeSchema,
    OptimizeResponseV2Schema,
    TourneeRequestV2Schema,
    TourneeSchema,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_DATE = date(2026, 4, 14)
NOW          = datetime(2026, 4, 14, 10, 0, 0)

# tournee_id factice réutilisé dans toutes les fixtures de test
_TOURNEE_ID_TEST = "TRN_TEST_20260414_00000001"


def make_node(stop_id: str, name: str, t: int) -> NodeSchema:
    return NodeSchema(stop_id=stop_id, stop_name=name,
                      time_minutes=t, service_date=SERVICE_DATE)


def make_train_arc(dep_id, dep_name, dep_t, arr_id, arr_name, arr_t,
                   train_number="117756", score=0.75, trip_id="T1") -> ArcSchema:
    return ArcSchema(
        source=make_node(dep_id, dep_name, dep_t),
        destination=make_node(arr_id, arr_name, arr_t),
        arc_type=ArcTypeSchema.TRAIN,
        duration_min=arr_t - dep_t,
        trip_id=trip_id,
        train_number=train_number,
        fraud_score=score,
    )


def make_corr_arc(stop_id, name, t_from, t_to) -> ArcSchema:
    return ArcSchema(
        source=make_node(stop_id, name, t_from),
        destination=make_node(stop_id, name, t_to),
        arc_type=ArcTypeSchema.CORRESPONDANCE,
        duration_min=t_to - t_from,
    )


def make_request(agent_id="LAF_042", show_scores=False) -> TourneeRequestV2Schema:
    return TourneeRequestV2Schema(
        gare_depart_id="87212027",
        heure_ps_min=270,
        heure_fs_min=840,
        service_date=SERVICE_DATE,
        agent_id=agent_id,
        show_scores=show_scores,
    )


@pytest.fixture
def tournee_simple() -> TourneeSchema:
    """Tournée avec 2 trains et une correspondance intermédiaire."""
    arc1 = make_train_arc("87212027", "Strasbourg", 480, "87214007", "Sélestat",   509, "117756", 0.80, "T1")
    corr = make_corr_arc("87214007", "Sélestat", 509, 514)
    arc2 = make_train_arc("87214007", "Sélestat",   514, "87214080", "Colmar",     537, "117758", 0.65, "T2")
    return TourneeSchema(
        arcs=[arc1, corr, arc2],
        score_total=1.45,
        duree_totale_minutes=57,
        nb_trains=2,
        gare_depart=make_node("87212027", "Strasbourg", 480),
        gare_arrivee=make_node("87214080", "Colmar", 537),
        service_date=SERVICE_DATE,
        generated_at=NOW,
    )


@pytest.fixture
def response_simple(tournee_simple) -> OptimizeResponseV2Schema:
    # FIX : tournee_id est requis depuis le refactoring Bug 1 (UUID unique par router)
    return OptimizeResponseV2Schema(
        tournee_id=_TOURNEE_ID_TEST,
        tournee=tournee_simple,
        request=make_request(),
        score_perte_pct=5.2,
        warning_messages=["Test warning"],
        optimized_at=NOW,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests _minutes_to_hhmm
# ─────────────────────────────────────────────────────────────────────────────

class TestMinutesToHhmm:

    def test_heure_standard(self):
        assert _minutes_to_hhmm(503) == "08:23"

    def test_minuit(self):
        assert _minutes_to_hhmm(0) == "00:00"

    def test_midi(self):
        assert _minutes_to_hhmm(720) == "12:00"

    def test_fin_de_journee(self):
        assert _minutes_to_hhmm(1439) == "23:59"

    def test_heure_superieure_24h(self):
        """Horaires après minuit (trains de nuit)."""
        assert _minutes_to_hhmm(1500) == "25:00"

    def test_zero_minutes(self):
        assert _minutes_to_hhmm(480) == "08:00"

    def test_minutes_impaires(self):
        assert _minutes_to_hhmm(547) == "09:07"


# ─────────────────────────────────────────────────────────────────────────────
# Tests _score_to_priorite
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreToPriorite:

    def test_score_haute(self):
        assert _score_to_priorite(SEUIL_HAUTE) == "HAUTE"
        assert _score_to_priorite(0.90) == "HAUTE"
        assert _score_to_priorite(1.0)  == "HAUTE"

    def test_score_moyenne(self):
        assert _score_to_priorite(SEUIL_MOYENNE) == "MOYENNE"
        assert _score_to_priorite(0.50)           == "MOYENNE"
        assert _score_to_priorite(SEUIL_HAUTE - 0.01) == "MOYENNE"

    def test_score_basse(self):
        assert _score_to_priorite(0.0)                 == "BASSE"
        assert _score_to_priorite(SEUIL_MOYENNE - 0.01) == "BASSE"
        assert _score_to_priorite(0.20)                 == "BASSE"

    def test_seuils_sont_inclusifs_haute(self):
        """SEUIL_HAUTE lui-même → HAUTE (borne inclusive)."""
        assert _score_to_priorite(SEUIL_HAUTE) == "HAUTE"

    def test_seuils_sont_inclusifs_moyenne(self):
        assert _score_to_priorite(SEUIL_MOYENNE) == "MOYENNE"


# ─────────────────────────────────────────────────────────────────────────────
# Tests TourneeFormatter.format()
# ─────────────────────────────────────────────────────────────────────────────

class TestTourneeFormatter:

    def test_retourne_dataframe(self, response_simple):
        df = TourneeFormatter().format(response_simple)
        assert isinstance(df, pd.DataFrame)

    def test_une_ligne_par_train(self, response_simple):
        """Tournée avec 2 trains → 2 lignes (les correspondances sont exclues)."""
        df = TourneeFormatter().format(response_simple)
        assert len(df) == 2

    def test_colonnes_obligatoires_presentes(self, response_simple):
        df = TourneeFormatter().format(response_simple)
        expected_cols = {
            "tournee_id", "agent_id", "service_date", "generated_at",
            "ordre", "numero_train", "gare_dep", "heure_dep",
            "gare_arr", "heure_arr", "duree_trajet_min", "attente_suivant_min",
            "priorite", "score_total_tournee", "nb_trains", "duree_totale_min",
            "score_perte_pct", "warnings",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_fraud_score_absent_si_show_scores_false(self, response_simple):
        df = TourneeFormatter().format(response_simple)
        assert "fraud_score" not in df.columns

    def test_fraud_score_present_si_show_scores_true(self, tournee_simple):
        req      = make_request(show_scores=True)
        # FIX : tournee_id requis
        response = OptimizeResponseV2Schema(
            tournee_id=_TOURNEE_ID_TEST,
            tournee=tournee_simple,
            request=req,
            optimized_at=NOW,
        )
        df = TourneeFormatter().format(response)
        assert "fraud_score" in df.columns
        assert df["fraud_score"].between(0.0, 1.0).all()

    def test_ordre_incremental(self, response_simple):
        df = TourneeFormatter().format(response_simple)
        assert list(df["ordre"]) == [1, 2]

    def test_gare_dep_et_arr(self, response_simple):
        df = TourneeFormatter().format(response_simple)
        assert df.iloc[0]["gare_dep"] == "Strasbourg"
        assert df.iloc[0]["gare_arr"] == "Sélestat"
        assert df.iloc[1]["gare_dep"] == "Sélestat"
        assert df.iloc[1]["gare_arr"] == "Colmar"

    def test_heure_dep_format_hhmm(self, response_simple):
        df = TourneeFormatter().format(response_simple)
        assert df.iloc[0]["heure_dep"] == "08:00"   # 480 min

    def test_attente_premier_train(self, response_simple):
        """Premier train : attente = durée de la correspondance suivante (5 min)."""
        df = TourneeFormatter().format(response_simple)
        assert df.iloc[0]["attente_suivant_min"] == 5

    def test_attente_dernier_train_est_zero(self, response_simple):
        """Dernier train : aucune correspondance après → attente = 0."""
        df = TourneeFormatter().format(response_simple)
        assert df.iloc[-1]["attente_suivant_min"] == 0

    def test_score_perte_pct_dans_df(self, response_simple):
        df = TourneeFormatter().format(response_simple)
        assert (abs(df["score_perte_pct"] - 5.2) < 1e-5).all()

    def test_warnings_dans_df(self, response_simple):
        df = TourneeFormatter().format(response_simple)
        assert "Test warning" in df.iloc[0]["warnings"]

    def test_tournee_id_unique_par_appel(self, response_simple):
        """Deux appels successifs doivent produire des tournee_id différents."""
        df1 = TourneeFormatter().format(response_simple)
        df2 = TourneeFormatter().format(response_simple)
        assert df1.iloc[0]["tournee_id"] != df2.iloc[0]["tournee_id"]

    def test_tournee_id_format(self, response_simple):
        """Format : TRN_{AGENT_ID}_{YYYYMMDD}_{uid4_court}"""
        df  = TourneeFormatter().format(response_simple)
        tid = df.iloc[0]["tournee_id"]
        assert tid.startswith("TRN_"), f"Doit commencer par TRN_ : {tid}"
        assert "20260414" in tid, f"Date YYYYMMDD attendue dans l'ID : {tid}"
        uid_part = tid.split("_")[-1]
        assert len(uid_part) == 8, f"UID attendu sur 8 caractères, reçu : {uid_part!r}"

    def test_agent_id_dans_df(self, response_simple):
        df = TourneeFormatter().format(response_simple)
        assert (df["agent_id"] == "LAF_042").all()

    def test_df_vide_si_aucun_arc_train(self, tournee_simple):
        """Une tournée sans arc TRAIN retourne un DataFrame vide."""
        arc_corr_only = make_corr_arc("87212027", "Strasbourg", 480, 490)
        tournee_vide  = TourneeSchema(
            arcs=[arc_corr_only],
            score_total=0.0,
            duree_totale_minutes=10,
            nb_trains=1,
            gare_depart=make_node("87212027", "Strasbourg", 480),
            gare_arrivee=make_node("87212027", "Strasbourg", 490),
            service_date=SERVICE_DATE,
        )
        # FIX : tournee_id requis
        response = OptimizeResponseV2Schema(
            tournee_id=_TOURNEE_ID_TEST,
            tournee=tournee_vide,
            request=make_request(),
            optimized_at=NOW,
        )
        df = TourneeFormatter().format(response)
        assert df.empty

    def test_priorite_correcte_haute(self, tournee_simple):
        """Score 0.80 → HAUTE."""
        # FIX : tournee_id requis
        df = TourneeFormatter().format(
            OptimizeResponseV2Schema(
                tournee_id=_TOURNEE_ID_TEST,
                tournee=tournee_simple,
                request=make_request(),
                optimized_at=NOW,
            )
        )
        assert df.iloc[0]["priorite"] == "HAUTE"

    def test_priorite_correcte_moyenne(self, tournee_simple):
        """Score 0.65 → HAUTE (au seuil exact)."""
        # FIX : tournee_id requis
        df = TourneeFormatter().format(
            OptimizeResponseV2Schema(
                tournee_id=_TOURNEE_ID_TEST,
                tournee=tournee_simple,
                request=make_request(),
                optimized_at=NOW,
            )
        )
        assert df.iloc[1]["priorite"] == "HAUTE"   # 0.65 == SEUIL_HAUTE