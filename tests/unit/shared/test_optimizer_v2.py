# tests/unit/test_optimize_v2_schemas.py
"""
Tests unitaires des schémas v2 :
    TourneeRequestV2Schema — validation et propriétés dérivées
    OptimizeResponseV2Schema — construction et valeurs par défaut
"""
from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from shared.schemas import TourneeRequestV2Schema, OptimizeResponseV2Schema


SERVICE_DATE = date(2026, 4, 14)


# ─────────────────────────────────────────────────────────────────────────────
# Tests TourneeRequestV2Schema
# ─────────────────────────────────────────────────────────────────────────────

class TestTourneeRequestV2Schema:

    def test_instanciation_minimale(self):
        req = TourneeRequestV2Schema(
            gare_depart_id="87212027",
            heure_ps_min=270,     # 04h30
            heure_fs_min=840,     # 14h00
            service_date=SERVICE_DATE,
        )
        assert req.gare_depart_id == "87212027"

    def test_heure_depart_min_est_ps_plus_10(self):
        req = TourneeRequestV2Schema(
            gare_depart_id="87212027",
            heure_ps_min=270,
            heure_fs_min=840,
            service_date=SERVICE_DATE,
        )
        assert req.heure_depart_min == 280   # 270 + 10

    def test_duree_max_minutes_derivee(self):
        req = TourneeRequestV2Schema(
            gare_depart_id="87212027",
            heure_ps_min=270,   # 04h30
            heure_fs_min=840,   # 14h00 → 840 - 280 = 560 min
            service_date=SERVICE_DATE,
        )
        assert req.duree_max_minutes == 560

    def test_gare_arrivee_effective_aller_retour_sans_arrivee(self):
        """Mode aller_retour sans gare_arrivee → retour gare départ."""
        req = TourneeRequestV2Schema(
            gare_depart_id="87212027",
            heure_ps_min=270,
            heure_fs_min=840,
            service_date=SERVICE_DATE,
            mode="aller_retour",
        )
        assert req.gare_arrivee_effective == "87212027"

    def test_gare_arrivee_effective_arrivee_explicite(self):
        """gare_arrivee_id explicite prime sur le mode."""
        req = TourneeRequestV2Schema(
            gare_depart_id="87212027",
            heure_ps_min=270,
            heure_fs_min=840,
            service_date=SERVICE_DATE,
            gare_arrivee_id="87182063",   # Mulhouse
            mode="aller_retour",
        )
        assert req.gare_arrivee_effective == "87182063"

    def test_gare_arrivee_effective_decouche(self):
        """Mode découché sans gare_arrivee → None (pas de contrainte)."""
        req = TourneeRequestV2Schema(
            gare_depart_id="87212027",
            heure_ps_min=270,
            heure_fs_min=840,
            service_date=SERVICE_DATE,
            mode="decouche",
        )
        assert req.gare_arrivee_effective is None

    def test_validation_gare_depart_invalide(self):
        with pytest.raises(ValidationError, match="gare_depart_id invalide"):
            TourneeRequestV2Schema(
                gare_depart_id="ABC",
                heure_ps_min=270,
                heure_fs_min=840,
                service_date=SERVICE_DATE,
            )

    def test_validation_gare_arrivee_invalide(self):
        with pytest.raises(ValidationError, match="gare_arrivee_id invalide"):
            TourneeRequestV2Schema(
                gare_depart_id="87212027",
                heure_ps_min=270,
                heure_fs_min=840,
                service_date=SERVICE_DATE,
                gare_arrivee_id="INVALID",
            )

    def test_validation_fs_inferieur_ps(self):
        with pytest.raises(ValidationError):
            TourneeRequestV2Schema(
                gare_depart_id="87212027",
                heure_ps_min=840,   # FS < PS → invalide
                heure_fs_min=270,
                service_date=SERVICE_DATE,
            )

    def test_validation_fenetre_trop_courte(self):
        """PS 08h00, FS 08h20 → duree_max = 10 min < 30 min minimum."""
        with pytest.raises(ValidationError, match="trop courte"):
            TourneeRequestV2Schema(
                gare_depart_id="87212027",
                heure_ps_min=480,   # 08h00
                heure_fs_min=500,   # 08h20 → 500 - 490 = 10 min
                service_date=SERVICE_DATE,
            )

    def test_agent_id_optionnel(self):
        req = TourneeRequestV2Schema(
            gare_depart_id="87212027",
            heure_ps_min=270,
            heure_fs_min=840,
            service_date=SERVICE_DATE,
        )
        assert req.agent_id is None

    def test_show_scores_defaut_false(self):
        req = TourneeRequestV2Schema(
            gare_depart_id="87212027",
            heure_ps_min=270,
            heure_fs_min=840,
            service_date=SERVICE_DATE,
        )
        assert req.show_scores is False

    def test_mode_aller_retour_par_defaut(self):
        req = TourneeRequestV2Schema(
            gare_depart_id="87212027",
            heure_ps_min=270,
            heure_fs_min=840,
            service_date=SERVICE_DATE,
        )
        assert req.mode == "aller_retour"