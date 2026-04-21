# tests/unit/test_orienteering_sink.py
"""
Tests unitaires de la contrainte de retour (gare_arrivee_id) dans OrienteeringOptimizer.

Couvre :
    _find_sink_nodes()      — identification des nœuds sink dans le sous-graphe
    solve() avec retour     — résolution MILP avec contrainte C2b
    ValueError si infaisable — pas de route vers gare_arrivee dans le budget
    Greedy raise sur retour  — le greedy ne gère pas la contrainte, raise ValueError
"""
from __future__ import annotations

from datetime import date

import pytest

from services.or_engine.graph.transition import Arc, ArcType, Node
from services.or_engine.optimizer.orienteering import OrienteeringOptimizer

SERVICE_DATE = date(2024, 9, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_node(stop_id: str, stop_name: str, time_min: int) -> Node:
    return Node(stop_id=stop_id, stop_name=stop_name,
                time_minutes=time_min, service_date=SERVICE_DATE)


def make_arc(src, dst, arc_type, duration, score=0.0, trip_id="T1") -> Arc:
    return Arc(
        source=src, destination=dst,
        arc_type=arc_type, duration_min=duration,
        trip_id=trip_id if arc_type == ArcType.TRAIN else None,
        train_number="TEST" if arc_type == ArcType.TRAIN else None,
        fraud_score=score,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graphe de test : Strasbourg → Sélestat → Mulhouse avec retour
#
#   [STR 08:00] --TRAIN(30min,0.8)--> [SEL 08:30]
#   [SEL 08:30] --CORR(8min)-------> [SEL 08:38]
#   [SEL 08:38] --TRAIN(30min,0.7)--> [MUL 09:08]
#   [MUL 09:08] --CORR(10min)------> [MUL 09:18]
#   [MUL 09:18] --TRAIN(45min,0.6)--> [STR 10:03]  ← arc de retour
#
# Contrainte : retour à STR (87212027) dans 180 min
# ─────────────────────────────────────────────────────────────────────────────

STR = "87212027"
SEL = "87214007"
MUL = "87182063"

n_str_800  = make_node(STR, "Strasbourg", 8*60)
n_sel_830  = make_node(SEL, "Sélestat",   8*60+30)
n_sel_838  = make_node(SEL, "Sélestat",   8*60+38)
n_mul_908  = make_node(MUL, "Mulhouse",   9*60+8)
n_mul_918  = make_node(MUL, "Mulhouse",   9*60+18)
n_str_1003 = make_node(STR, "Strasbourg", 10*60+3)

arc_str_sel = make_arc(n_str_800, n_sel_830, ArcType.TRAIN, 30, 0.8, "T1")
arc_sel_cor = make_arc(n_sel_830, n_sel_838, ArcType.CORRESPONDANCE, 8)
arc_sel_mul = make_arc(n_sel_838, n_mul_908, ArcType.TRAIN, 30, 0.7, "T2")
arc_mul_cor = make_arc(n_mul_908, n_mul_918, ArcType.CORRESPONDANCE, 10)
arc_mul_str = make_arc(n_mul_918, n_str_1003, ArcType.TRAIN, 45, 0.6, "T3")   # ← retour

GRAPH_AVEC_RETOUR: dict[Node, list[Arc]] = {
    n_str_800:  [arc_str_sel],
    n_sel_830:  [arc_sel_cor],
    n_sel_838:  [arc_sel_mul],
    n_mul_908:  [arc_mul_cor],
    n_mul_918:  [arc_mul_str],
    n_str_1003: [],
}


# ─────────────────────────────────────────────────────────────────────────────
# Tests _find_sink_nodes()
# ─────────────────────────────────────────────────────────────────────────────

class TestFindSinkNodes:

    def test_trouve_noeuds_sink_dans_arcs(self):
        optimizer = OrienteeringOptimizer()
        all_arcs  = [arc_str_sel, arc_sel_cor, arc_sel_mul, arc_mul_cor, arc_mul_str]
        sink_nodes = optimizer._find_sink_nodes(all_arcs, STR)
        # n_str_1003 est une destination d'arc TRAIN → doit être dans sink_nodes
        assert n_str_1003 in sink_nodes

    def test_sink_vide_si_gare_absente(self):
        optimizer = OrienteeringOptimizer()
        all_arcs  = [arc_str_sel]  # aucun arc n'arrive à Mulhouse
        sink_nodes = optimizer._find_sink_nodes(all_arcs, MUL)
        assert len(sink_nodes) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests solve() avec contrainte de retour
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

requires_pulp = pytest.mark.skipif(not PULP_AVAILABLE, reason="PuLP non installé")


class TestSolveAvecRetour:

    @requires_pulp
    def test_solve_avec_retour_strasbourg(self):
        """
        Tournée STR→SEL→MUL→STR dans 200 min.
        La contrainte de retour doit forcer le chemin à se terminer à STR.
        """
        optimizer = OrienteeringOptimizer()
        result    = optimizer.solve(
            graph=GRAPH_AVEC_RETOUR,
            gare_depart_id=STR,
            heure_depart_min=8*60,
            duree_max_minutes=200,
            gare_arrivee_id=STR,
        )

        assert result["nb_trains"] >= 1
        arcs_train = [a for a in result["arcs"] if a.arc_type == ArcType.TRAIN]
        dernier_arc = arcs_train[-1]
        assert dernier_arc.destination.stop_id == STR, \
            f"Dernier arc devrait arriver à {STR}, pas {dernier_arc.destination.stop_id}"

    @requires_pulp
    def test_solve_sans_retour_peut_terminer_ailleurs(self):
        """Sans contrainte de retour, la tournée peut se terminer à Mulhouse."""
        optimizer = OrienteeringOptimizer()
        result    = optimizer.solve(
            graph=GRAPH_AVEC_RETOUR,
            gare_depart_id=STR,
            heure_depart_min=8*60,
            duree_max_minutes=80,      # 80 min : assez pour STR→SEL→MUL, pas pour revenir
            gare_arrivee_id=None,
        )
        assert result["nb_trains"] >= 1

    @requires_pulp
    def test_solve_retour_impossible_leve_valueerror(self):
        """
        Budget de 40 min : assez pour STR→SEL, pas pour revenir à STR.
        Doit lever ValueError (aucun arc de retour accessible dans le budget).
        """
        optimizer = OrienteeringOptimizer()
        with pytest.raises(ValueError, match="Aucun train n'arrive"):
            optimizer.solve(
                graph=GRAPH_AVEC_RETOUR,
                gare_depart_id=STR,
                heure_depart_min=8*60,
                duree_max_minutes=40,       # trop court pour un aller-retour
                gare_arrivee_id=STR,
            )

    @requires_pulp
    def test_solve_gare_arrivee_inexistante_leve_valueerror(self):
        """Gare d'arrivée absente du graphe → ValueError explicite."""
        optimizer = OrienteeringOptimizer()
        with pytest.raises(ValueError, match="Aucun train n'arrive"):
            optimizer.solve(
                graph=GRAPH_AVEC_RETOUR,
                gare_depart_id=STR,
                heure_depart_min=8*60,
                duree_max_minutes=200,
                gare_arrivee_id="00000000",   # gare inexistante
            )

    @requires_pulp
    def test_solve_retour_meme_gare_depart(self):
        """
        Le cas standard aller-retour : départ et arrivée à Strasbourg.
        La route doit inclure le train de retour MUL→STR.
        """
        optimizer = OrienteeringOptimizer()
        result    = optimizer.solve(
            graph=GRAPH_AVEC_RETOUR,
            gare_depart_id=STR,
            heure_depart_min=8*60,
            duree_max_minutes=200,
            gare_arrivee_id=STR,
        )
        trip_ids = [a.trip_id for a in result["arcs"] if a.arc_type == ArcType.TRAIN]
        # T3 = MUL→STR doit être dans la tournée pour satisfaire la contrainte retour
        assert "T3" in trip_ids, f"Le train de retour T3 devrait être inclus. Trains : {trip_ids}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests greedy fallback avec retour
# ─────────────────────────────────────────────────────────────────────────────

class TestGreedyAvecRetour:

    def test_greedy_avec_gare_arrivee_leve_valueerror(self):
        """
        _greedy_fallback avec gare_arrivee_id doit lever ValueError immédiatement
        — le greedy ne peut pas garantir le retour.
        """
        optimizer = OrienteeringOptimizer()
        with pytest.raises(ValueError, match="greedy ne gère pas"):
            optimizer._greedy_fallback(
                graph=GRAPH_AVEC_RETOUR,
                gare_depart_id=STR,
                heure_depart_min=8*60,
                duree_max_minutes=200,
                gare_arrivee_id=STR,
            )

    def test_greedy_sans_gare_arrivee_fonctionne_normalement(self):
        """Sans contrainte retour, le greedy fonctionne comme avant."""
        optimizer = OrienteeringOptimizer()
        result    = optimizer._greedy_fallback(
            graph=GRAPH_AVEC_RETOUR,
            gare_depart_id=STR,
            heure_depart_min=8*60,
            duree_max_minutes=200,
            gare_arrivee_id=None,
        )
        assert result["nb_trains"] >= 1