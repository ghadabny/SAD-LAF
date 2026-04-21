import pytest
from datetime import date, datetime
from pydantic import ValidationError

from shared.schemas import (
    ArcSchema, ArcTypeSchema, NodeSchema,
    ServiceAlertSchema, StopTimeUpdateSchema,
    TronconSchema, TronconScoreSchema,
    TourneeRequestSchema, TourneeSchema,
    TripUpdateSchema,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def node_strasbourg():
    return NodeSchema(
        stop_id="87212027", stop_name="Strasbourg",
        time_minutes=503, service_date=date(2024, 9, 2),
    )


@pytest.fixture
def node_selestat():
    return NodeSchema(
        stop_id="87214007", stop_name="Sélestat",
        time_minutes=532, service_date=date(2024, 9, 2),
    )


@pytest.fixture
def troncon_valide():
    return TronconSchema(
        trip_id="TRIP_001", train_number="117756", service_id="000001",
        stop_sequence=0,
        stop_id_dep="87212027", stop_name_dep="Strasbourg", dep_minutes=503,
        stop_id_arr="87214007", stop_name_arr="Sélestat",   arr_minutes=532,
        duration_min=29,
    )


@pytest.fixture
def arc_train(node_strasbourg, node_selestat):
    return ArcSchema(
        source=node_strasbourg, destination=node_selestat,
        arc_type=ArcTypeSchema.TRAIN, duration_min=29,
        trip_id="TRIP_001", train_number="117756", fraud_score=0.75,
    )


@pytest.fixture
def tournee_request():
    return TourneeRequestSchema(
        gare_depart_id="87212027", heure_depart_min=503,
        duree_max_minutes=360, service_date=date(2024, 9, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests TronconSchema
# ─────────────────────────────────────────────────────────────────────────────

class TestTronconSchema:

    def test_troncon_valide(self, troncon_valide):
        assert troncon_valide.trip_id == "TRIP_001"
        assert troncon_valide.duration_min == 29

    def test_stop_id_invalide_leve_erreur(self):
        with pytest.raises(ValidationError, match="stop_id invalide"):
            TronconSchema(
                trip_id="T1", train_number="117756", service_id="000001",
                stop_sequence=0,
                stop_id_dep="INVALID", stop_name_dep="Strasbourg", dep_minutes=503,
                stop_id_arr="87214007", stop_name_arr="Sélestat", arr_minutes=532,
                duration_min=29,
            )

    def test_duration_nulle_leve_erreur(self):
        with pytest.raises(ValidationError):
            TronconSchema(
                trip_id="T1", train_number="117756", service_id="000001",
                stop_sequence=0,
                stop_id_dep="87212027", stop_name_dep="Strasbourg", dep_minutes=503,
                stop_id_arr="87214007", stop_name_arr="Sélestat", arr_minutes=532,
                duration_min=0,
            )

    def test_dep_minutes_negatif_leve_erreur(self):
        with pytest.raises(ValidationError):
            TronconSchema(
                trip_id="T1", train_number="117756", service_id="000001",
                stop_sequence=0,
                stop_id_dep="87212027", stop_name_dep="Strasbourg", dep_minutes=-1,
                stop_id_arr="87214007", stop_name_arr="Sélestat", arr_minutes=532,
                duration_min=29,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests TronconScoreSchema
# ─────────────────────────────────────────────────────────────────────────────

class TestTronconScoreSchema:

    def test_fraud_score_defaut(self, troncon_valide):
        scored = TronconScoreSchema(**troncon_valide.model_dump())
        assert scored.fraud_score == 0.0

    def test_fraud_score_valide(self, troncon_valide):
        scored = TronconScoreSchema(**troncon_valide.model_dump(), fraud_score=0.75)
        assert scored.fraud_score == 0.75

    def test_fraud_score_hors_bornes_leve_erreur(self, troncon_valide):
        with pytest.raises(ValidationError):
            TronconScoreSchema(**troncon_valide.model_dump(), fraud_score=1.5)


# ─────────────────────────────────────────────────────────────────────────────
# Tests NodeSchema
# ─────────────────────────────────────────────────────────────────────────────

class TestNodeSchema:

    def test_node_valide(self, node_strasbourg):
        assert node_strasbourg.stop_id == "87212027"
        assert node_strasbourg.time_minutes == 503

    def test_label_correct(self, node_strasbourg):
        """503 minutes = 08h23 → label = 'Strasbourg 08:23'"""
        assert node_strasbourg.label == "Strasbourg 08:23"

    def test_time_minutes_negatif_leve_erreur(self):
        with pytest.raises(ValidationError):
            NodeSchema(
                stop_id="87212027", stop_name="Strasbourg",
                time_minutes=-1, service_date=date(2024, 9, 2),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests ArcSchema
# ─────────────────────────────────────────────────────────────────────────────

class TestArcSchema:

    def test_arc_train_valide(self, arc_train):
        assert arc_train.arc_type == ArcTypeSchema.TRAIN
        assert arc_train.fraud_score == 0.75

    def test_arc_correspondance_sans_train_number(self, node_strasbourg, node_selestat):
        arc = ArcSchema(
            source=node_strasbourg, destination=node_selestat,
            arc_type=ArcTypeSchema.CORRESPONDANCE, duration_min=13,
        )
        assert arc.train_number is None
        assert arc.arc_type == ArcTypeSchema.CORRESPONDANCE

    def test_arc_type_serialisation_json(self, arc_train):
        """ArcTypeSchema.TRAIN doit se sérialiser en 'train' dans le JSON."""
        data = arc_train.model_dump()
        assert data["arc_type"] == "train"

    def test_duration_nulle_leve_erreur(self, node_strasbourg, node_selestat):
        with pytest.raises(ValidationError):
            ArcSchema(
                source=node_strasbourg, destination=node_selestat,
                arc_type=ArcTypeSchema.TRAIN, duration_min=0,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests TourneeRequestSchema
# ─────────────────────────────────────────────────────────────────────────────

class TestTourneeRequestSchema:

    def test_request_valide(self, tournee_request):
        assert tournee_request.gare_depart_id == "87212027"
        assert tournee_request.duree_max_minutes == 360

    def test_gare_invalide_leve_erreur(self):
        with pytest.raises(ValidationError, match="gare_depart_id invalide"):
            TourneeRequestSchema(
                gare_depart_id="INVALID",
                heure_depart_min=503,
                service_date=date(2024, 9, 2),
            )

    def test_duree_max_defaut(self):
        req = TourneeRequestSchema(
            gare_depart_id="87212027",
            heure_depart_min=503,
            service_date=date(2024, 9, 2),
        )
        assert req.duree_max_minutes == 360

    def test_heure_depart_hors_bornes_leve_erreur(self):
        with pytest.raises(ValidationError):
            TourneeRequestSchema(
                gare_depart_id="87212027",
                heure_depart_min=1500,
                service_date=date(2024, 9, 2),
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests TourneeSchema
# ─────────────────────────────────────────────────────────────────────────────

class TestTourneeSchema:

    def test_arcs_train_filtre_correspondances(self, arc_train, node_strasbourg, node_selestat):
        arc_corr = ArcSchema(
            source=node_selestat, destination=node_strasbourg,
            arc_type=ArcTypeSchema.CORRESPONDANCE, duration_min=13,
        )
        tournee = TourneeSchema(
            arcs=[arc_train, arc_corr],
            score_total=0.75, duree_totale_minutes=42,
            nb_trains=1, gare_depart=node_strasbourg,
            gare_arrivee=node_selestat, service_date=date(2024, 9, 2),
        )
        assert len(tournee.arcs_train) == 1
        assert tournee.arcs_train[0].arc_type == ArcTypeSchema.TRAIN

    def test_trains_visites(self, arc_train, node_strasbourg, node_selestat):
        tournee = TourneeSchema(
            arcs=[arc_train], score_total=0.75, duree_totale_minutes=29,
            nb_trains=1, gare_depart=node_strasbourg,
            gare_arrivee=node_selestat, service_date=date(2024, 9, 2),
        )
        assert tournee.trains_visites == ["117756"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests GTFS-RT
# ─────────────────────────────────────────────────────────────────────────────

class TestRealtimeSchemas:

    def test_stop_time_update_valide(self):
        stu = StopTimeUpdateSchema(
            stop_id="87212027", stop_sequence=0,
            arrival_delay_seconds=120, departure_delay_seconds=120,
        )
        assert stu.arrival_delay_seconds == 120

    def test_stop_time_update_sans_delay(self):
        stu = StopTimeUpdateSchema(stop_id="87212027", stop_sequence=0)
        assert stu.arrival_delay_seconds is None

    def test_service_alert_is_suppression(self):
        alert = ServiceAlertSchema(alert_id="A001", cause="STRIKE", effect="NO_SERVICE")
        assert alert.is_suppression is True

    def test_service_alert_not_suppression(self):
        alert = ServiceAlertSchema(alert_id="A002", cause="TECHNICAL_PROBLEM", effect="REDUCED_SERVICE")
        assert alert.is_suppression is False

    def test_trip_update_stop_time_updates_vide(self):
        tu = TripUpdateSchema(trip_id="TRIP_001")
        assert tu.stop_time_updates == []