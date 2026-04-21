import pytest
import pandas as pd
from datetime import date

from services.or_engine.graph.builder import TimeExpandedGraphBuilder
from services.or_engine.graph.transition import ArcType, Node


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_DATE = date(2024, 9, 2)  # lundi, jour de référence pour tous les tests


@pytest.fixture
def builder():
    return TimeExpandedGraphBuilder()


@pytest.fixture
def troncons_simples():
    """
    3 tronçons formant un trajet Strasbourg → Sélestat → Colmar
    sur le même train, plus un second train depuis Sélestat → Mulhouse
    qui permet de tester les correspondances.

    Strasbourg 08:23 ──117756──→ Sélestat 08:52 ──117756──→ Colmar 09:15
                                 Sélestat 09:05 ──117758──→ Mulhouse 09:58
    """
    return pd.DataFrame([
        {
            "trip_id": "TRIP_001", "train_number": "117756",
            "stop_id_dep": "87212027", "stop_name_dep": "Strasbourg", "dep_minutes": 8 * 60 + 23,
            "stop_id_arr": "87214007", "stop_name_arr": "Sélestat",   "arr_minutes": 8 * 60 + 52,
            "duration_min": 29,
        },
        {
            "trip_id": "TRIP_001", "train_number": "117756",
            "stop_id_dep": "87214007", "stop_name_dep": "Sélestat", "dep_minutes": 8 * 60 + 52,
            "stop_id_arr": "87214080", "stop_name_arr": "Colmar",   "arr_minutes": 9 * 60 + 15,
            "duration_min": 23,
        },
        {
            "trip_id": "TRIP_002", "train_number": "117758",
            "stop_id_dep": "87214007", "stop_name_dep": "Sélestat", "dep_minutes": 9 * 60 + 5,
            "stop_id_arr": "87182063", "stop_name_arr": "Mulhouse", "arr_minutes": 9 * 60 + 58,
            "duration_min": 53,
        },
    ])


@pytest.fixture
def troncon_trop_court():
    """
    Tronçon de 4 minutes : trop court pour qu'un agent puisse contrôler.
    Doit être ignoré par le builder (MIN_BOARD_DURATION_MINUTES = 6).
    """
    return pd.DataFrame([{
        "trip_id": "TRIP_003", "train_number": "117760",
        "stop_id_dep": "87212027", "stop_name_dep": "Strasbourg", "dep_minutes": 500,
        "stop_id_arr": "87214007", "stop_name_arr": "Sélestat",   "arr_minutes": 504,
        "duration_min": 4,
    }])


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeExpandedGraphBuilder:

    # ── Structure du graphe ───────────────────────────────────────────────────

    def test_build_returns_dict(self, builder, troncons_simples):
        """Le graphe retourné doit être un dictionnaire."""
        graph = builder.build(troncons_simples, SERVICE_DATE)
        assert isinstance(graph, dict)

    def test_nodes_are_node_instances(self, builder, troncons_simples):
        """Les clés du graphe doivent être des instances de Node."""
        graph = builder.build(troncons_simples, SERVICE_DATE)
        for node in graph:
            assert isinstance(node, Node)

    def test_graph_not_empty(self, builder, troncons_simples):
        """Un graphe construit depuis des tronçons valides ne doit pas être vide."""
        graph = builder.build(troncons_simples, SERVICE_DATE)
        assert len(graph) > 0

    # ── Arcs TRAIN ────────────────────────────────────────────────────────────

    def test_train_arcs_count(self, builder, troncons_simples):
        """
        3 tronçons valides → au moins 3 arcs TRAIN.
        (Il peut y avoir plus d'arcs au total avec les correspondances.)
        """
        graph = builder.build(troncons_simples, SERVICE_DATE)
        all_arcs = [arc for arcs in graph.values() for arc in arcs]
        train_arcs = [a for a in all_arcs if a.arc_type == ArcType.TRAIN]
        assert len(train_arcs) == 3

    def test_train_arc_source_correct(self, builder, troncons_simples):
        """Le nœud source de l'arc Strasbourg→Sélestat doit être à 08:23."""
        graph = builder.build(troncons_simples, SERVICE_DATE)
        all_arcs = [arc for arcs in graph.values() for arc in arcs]
        train_arcs = [a for a in all_arcs if a.arc_type == ArcType.TRAIN]

        # On cherche l'arc qui part de Strasbourg
        arc_stras = next(
            a for a in train_arcs if a.source.stop_id == "87212027"
        )
        assert arc_stras.source.time_minutes == 8 * 60 + 23
        assert arc_stras.destination.stop_id == "87214007"
        assert arc_stras.duration_min == 29

    def test_train_arc_has_train_number(self, builder, troncons_simples):
        """Les arcs TRAIN doivent avoir un numéro de train renseigné."""
        graph = builder.build(troncons_simples, SERVICE_DATE)
        all_arcs = [arc for arcs in graph.values() for arc in arcs]
        train_arcs = [a for a in all_arcs if a.arc_type == ArcType.TRAIN]
        for arc in train_arcs:
            assert arc.train_number is not None

    def test_fraud_score_initialized_to_zero(self, builder, troncons_simples):
        """
        À la construction, fraud_score doit être 0.0.
        Il sera mis à jour par LGBMScorer plus tard.
        """
        graph = builder.build(troncons_simples, SERVICE_DATE)
        all_arcs = [arc for arcs in graph.values() for arc in arcs]
        train_arcs = [a for a in all_arcs if a.arc_type == ArcType.TRAIN]
        for arc in train_arcs:
            assert arc.fraud_score == 0.0

    # ── Tronçon trop court ────────────────────────────────────────────────────

    def test_short_troncon_ignored(self, builder, troncon_trop_court):
        """
        Un tronçon de 4 min (< MIN_BOARD_DURATION_MINUTES=6) ne doit
        produire aucun arc TRAIN.
        """
        graph = builder.build(troncon_trop_court, SERVICE_DATE)
        all_arcs = [arc for arcs in graph.values() for arc in arcs]
        train_arcs = [a for a in all_arcs if a.arc_type == ArcType.TRAIN]
        assert len(train_arcs) == 0

    # ── Arcs CORRESPONDANCE ───────────────────────────────────────────────────

    def test_correspondance_arc_exists(self, builder, troncons_simples):
        """
        Sélestat : train 117756 arrive à 08:52, train 117758 part à 09:05.
        Délai = 13 min >= MIN_TRANSFER_MINUTES=5 → arc correspondance attendu.
        """
        graph = builder.build(troncons_simples, SERVICE_DATE)
        all_arcs = [arc for arcs in graph.values() for arc in arcs]
        corr_arcs = [a for a in all_arcs if a.arc_type == ArcType.CORRESPONDANCE]
        assert len(corr_arcs) > 0

    def test_correspondance_arc_at_selestat(self, builder, troncons_simples):
        """L'arc de correspondance doit être à Sélestat (stop_id 87214007)."""
        graph = builder.build(troncons_simples, SERVICE_DATE)
        all_arcs = [arc for arcs in graph.values() for arc in arcs]
        corr_arcs = [a for a in all_arcs if a.arc_type == ArcType.CORRESPONDANCE]

        selestat_corr = [
            a for a in corr_arcs if a.source.stop_id == "87214007"
        ]
        assert len(selestat_corr) > 0

    def test_correspondance_duration_correct(self, builder, troncons_simples):
        """
        La correspondance Sélestat 08:52 → 09:05 doit durer 13 minutes.
        """
        graph = builder.build(troncons_simples, SERVICE_DATE)
        all_arcs = [arc for arcs in graph.values() for arc in arcs]
        corr_arcs = [
            a for a in all_arcs
            if a.arc_type == ArcType.CORRESPONDANCE
            and a.source.stop_id == "87214007"
            and a.source.time_minutes == 8 * 60 + 52
        ]
        assert len(corr_arcs) == 1
        assert corr_arcs[0].duration_min == 13

    # ── Validation ────────────────────────────────────────────────────────────

    def test_missing_column_raises(self, builder):
        """Un DataFrame sans les colonnes requises doit lever ValueError."""
        df_incomplet = pd.DataFrame([{"trip_id": "T1", "train_number": "123"}])
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            builder.build(df_incomplet, SERVICE_DATE)

    # ── Méthodes utilitaires ──────────────────────────────────────────────────

    def test_get_nodes_at_stop(self, builder, troncons_simples):
        """get_nodes_at_stop doit retourner tous les nœuds d'une gare."""
        graph = builder.build(troncons_simples, SERVICE_DATE)
        nodes_selestat = builder.get_nodes_at_stop(graph, "87214007")
        # Sélestat apparaît comme départ ET arrivée → plusieurs nœuds
        assert len(nodes_selestat) > 0
        for node in nodes_selestat:
            assert node.stop_id == "87214007"

    def test_get_reachable_arcs_unknown_node(self, builder, troncons_simples):
        """
        get_reachable_arcs sur un nœud inexistant doit retourner []
        sans lever d'exception.
        """
        graph = builder.build(troncons_simples, SERVICE_DATE)
        fake_node = Node("00000000", "Nulle Part", 600, SERVICE_DATE)
        result = builder.get_reachable_arcs(graph, fake_node)
        assert result == []