"""
tests/unit/test_optimizer.py — Tests unitaires de OrienteeringOptimizer.

Structure des tests :
    TestBaseOptimizer          → vérification du contrat abstrait
    TestOrienteeringContrat    → le solve() respecte le format de retour
    TestOrienteeringBudget     → contrainte duree_max_minutes toujours respectée
    TestOrienteeringOptimal    → MILP > greedy sur instance où greedy échoue
    TestOrienteeringEdgeCases  → robustesse sur graphes dégénérés
    TestArcEligibility         → _is_arc_eligible() filtre correctement
    TestGreedyHelpers          → _best_train_arc et _shortest_correspondance
    TestReconstructPath        → _reconstruct_path() ordonne correctement
    TestGreedyWalk             → _greedy_walk() respecte le budget

Stratégie de mocking :
    PuLP n'est pas installé dans tous les environnements CI.
    Les tests marqués @requires_pulp sont conditionnels à pulp disponible.
    Les tests du fallback greedy tournent toujours (pas de dépendance PuLP).

Graphe synthétique :
    Réseau minimal Strasbourg → Sélestat → Colmar avec 3 trains :
        TRAIN_A : Strasbourg 06:00 → Sélestat 06:25   (25 min, score=0.3)
        CORR_1  : Sélestat   06:25 → Sélestat 06:35   (10 min)
        TRAIN_B : Sélestat   06:35 → Colmar   06:58   (23 min, score=0.8)
        TRAIN_C : Strasbourg 06:05 → Colmar   07:10   (65 min, score=0.9)

    Budget 60 min :
        Greedy → TRAIN_A (0.3) seulement — TRAIN_C hors budget
        MILP   → TRAIN_A + CORR_1 + TRAIN_B = 1.1 ✅
"""
import pytest
from datetime import date
from unittest.mock import patch

from services.or_engine.graph.transition import Arc, ArcType, Node
from services.or_engine.optimizer.base import BaseOptimizer
from services.or_engine.optimizer.orienteering import (
    OrienteeringOptimizer,
    _best_train_arc,
    _greedy_walk,
    _reconstruct_path,
    _shortest_correspondance,
)

# ─────────────────────────────────────────────────────────────────────────────
# Détection PuLP
# ─────────────────────────────────────────────────────────────────────────────

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

requires_pulp = pytest.mark.skipif(
    not PULP_AVAILABLE,
    reason="PuLP non installé — tests MILP ignorés"
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de construction
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_DATE = date(2024, 9, 2)


def make_node(stop_id: str, stop_name: str, time_min: int) -> Node:
    return Node(
        stop_id=stop_id,
        stop_name=stop_name,
        time_minutes=time_min,
        service_date=SERVICE_DATE,
    )


def make_arc(
    src: Node,
    dst: Node,
    arc_type: ArcType,
    duration: int,
    score: float = 0.0,
    trip_id: str = None,
    train_number: str = None,
) -> Arc:
    return Arc(
        source=src,
        destination=dst,
        arc_type=arc_type,
        duration_min=duration,
        trip_id=trip_id,
        train_number=train_number,
        fraud_score=score,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def graphe_simple():
    """
    Strasbourg → Sélestat → Colmar — conçu pour que MILP batte le greedy.
    """
    str_06h00 = make_node("87212027", "Strasbourg", 6 * 60)
    str_06h05 = make_node("87212027", "Strasbourg", 6 * 60 + 5)
    sel_06h25 = make_node("87214007", "Sélestat",   6 * 60 + 25)
    sel_06h35 = make_node("87214007", "Sélestat",   6 * 60 + 35)
    col_06h58 = make_node("87214080", "Colmar",     6 * 60 + 58)
    col_07h10 = make_node("87214080", "Colmar",     7 * 60 + 10)

    train_a = make_arc(str_06h00, sel_06h25, ArcType.TRAIN, 25, 0.30, "T_A", "11700")
    corr_1  = make_arc(sel_06h25, sel_06h35, ArcType.CORRESPONDANCE, 10)
    train_b = make_arc(sel_06h35, col_06h58, ArcType.TRAIN, 23, 0.80, "T_B", "11701")
    train_c = make_arc(str_06h05, col_07h10, ArcType.TRAIN, 65, 0.90, "T_C", "11702")

    return {
        str_06h00: [train_a],
        str_06h05: [train_c],
        sel_06h25: [corr_1],
        sel_06h35: [train_b],
        col_06h58: [],
        col_07h10: [],
    }


@pytest.fixture
def graphe_un_seul_train():
    src = make_node("87212027", "Strasbourg", 8 * 60)
    dst = make_node("87214007", "Sélestat",   8 * 60 + 25)
    arc = make_arc(src, dst, ArcType.TRAIN, 25, 0.75, trip_id="T1")
    return {src: [arc], dst: []}


@pytest.fixture
def graphe_vide():
    n = make_node("87212027", "Strasbourg", 8 * 60)
    return {n: []}


# ─────────────────────────────────────────────────────────────────────────────
# TestBaseOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseOptimizer:
    """Vérifie que BaseOptimizer est bien une interface abstraite."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseOptimizer()

    def test_concrete_subclass_must_implement_solve(self):
        class IncompleteOptimizer(BaseOptimizer):
            pass
        with pytest.raises(TypeError):
            IncompleteOptimizer()

    def test_valid_subclass_instantiates(self):
        class MinimalOptimizer(BaseOptimizer):
            def solve(self, graph, gare_depart_id, heure_depart_min, duree_max):
                return {"arcs": [], "score_total": 0.0,
                        "duree_minutes": 0, "nb_trains": 0}
        assert isinstance(MinimalOptimizer(), BaseOptimizer)


# ─────────────────────────────────────────────────────────────────────────────
# TestOrienteeringContrat
# ─────────────────────────────────────────────────────────────────────────────

class TestOrienteeringContrat:
    """Le solve() respecte le contrat de retour de BaseOptimizer."""

    def test_retourne_dict_avec_bonnes_cles(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        for key in ("arcs", "score_total", "duree_minutes", "nb_trains"):
            assert key in result, f"Clé manquante : {key}"

    def test_arcs_est_une_liste(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        assert isinstance(result["arcs"], list)

    def test_score_total_est_float(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        assert isinstance(result["score_total"], float)

    def test_nb_trains_coherent_avec_arcs(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        nb_effectif = sum(1 for a in result["arcs"] if a.arc_type == ArcType.TRAIN)
        assert result["nb_trains"] == nb_effectif

    def test_score_total_coherent_avec_arcs(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        score_calcule = round(
            sum(a.fraud_score for a in result["arcs"] if a.arc_type == ArcType.TRAIN), 4
        )
        assert abs(result["score_total"] - score_calcule) < 1e-3

    def test_un_seul_train_disponible(self, graphe_un_seul_train):
        result = OrienteeringOptimizer().solve(
            graphe_un_seul_train, "87212027", 8 * 60, 60
        )
        assert result["nb_trains"] == 1
        assert abs(result["score_total"] - 0.75) < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# TestOrienteeringBudget
# ─────────────────────────────────────────────────────────────────────────────

class TestOrienteeringBudget:
    """La contrainte duree_max_minutes est TOUJOURS respectée."""

    def test_duree_dans_budget(self, graphe_simple):
        budget = 60
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, budget)
        assert result["duree_minutes"] <= budget

    def test_budget_serre_exclut_train_long(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 30)
        assert result["duree_minutes"] <= 30
        trip_ids = [a.trip_id for a in result["arcs"] if a.arc_type == ArcType.TRAIN]
        assert "T_C" not in trip_ids

    def test_arcs_continus(self, graphe_simple):
        """Chaque arc.destination == arc_suivant.source."""
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        arcs = result["arcs"]
        for i in range(len(arcs) - 1):
            assert arcs[i].destination == arcs[i + 1].source, (
                f"Discontinuité arc {i} → {i + 1}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TestOrienteeringOptimal (nécessite PuLP)
# ─────────────────────────────────────────────────────────────────────────────

class TestOrienteeringOptimal:
    """MILP bat le greedy sur l'instance conçue pour piéger le greedy."""

    @requires_pulp
    def test_milp_superieur_greedy(self, graphe_simple):
        result_milp   = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 60)
        result_greedy = OrienteeringOptimizer()._greedy_fallback(
            graphe_simple, "87212027", 6 * 60, 60
        )
        assert result_milp["score_total"] >= result_greedy["score_total"]

    @requires_pulp
    def test_milp_trouve_deux_trains(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 60)
        assert result["nb_trains"] == 2

    @requires_pulp
    def test_milp_score_optimal_connu(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 60)
        assert abs(result["score_total"] - 1.10) < 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# TestOrienteeringEdgeCases
# ─────────────────────────────────────────────────────────────────────────────

class TestOrienteeringEdgeCases:
    """Robustesse sur les cas limites."""

    def test_gare_inconnue_leve_value_error(self, graphe_simple):
        with pytest.raises(ValueError, match="Aucun train"):
            OrienteeringOptimizer().solve(graphe_simple, "00000000", 6 * 60, 120)

    def test_heure_trop_tard_leve_value_error(self, graphe_simple):
        with pytest.raises(ValueError):
            OrienteeringOptimizer().solve(graphe_simple, "87212027", 23 * 60, 60)

    def test_graphe_sans_train_leve_value_error(self, graphe_vide):
        with pytest.raises(ValueError):
            OrienteeringOptimizer().solve(graphe_vide, "87212027", 8 * 60, 60)

    def test_budget_trop_serre_leve_value_error(self, graphe_simple):
        with pytest.raises(ValueError):
            OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 5)

    def test_scores_nuls_retourne_tournee(self):
        src = make_node("87212027", "Strasbourg", 8 * 60)
        dst = make_node("87214007", "Sélestat",   8 * 60 + 25)
        arc = make_arc(src, dst, ArcType.TRAIN, 25, 0.0, trip_id="T1")
        graph = {src: [arc], dst: []}
        result = OrienteeringOptimizer().solve(graph, "87212027", 8 * 60, 60)
        assert result["nb_trains"] == 1
        assert result["score_total"] == 0.0

    def test_fallback_greedy_si_pulp_absent(self, graphe_simple):
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **k: (
                (_ for _ in ()).throw(ImportError("no pulp"))
                if name == "pulp" else __import__(name, *a, **k)
            ),
        ):
            result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        assert "arcs" in result
        assert result["nb_trains"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# TestArcEligibility
# ─────────────────────────────────────────────────────────────────────────────

class TestArcEligibility:
    """_is_arc_eligible() applique correctement les filtres métier."""

    def test_arc_eligible_score_suffisant(self):
        src = make_node("87212027", "Strasbourg", 480)
        dst = make_node("87214007", "Sélestat",   505)
        arc = make_arc(src, dst, ArcType.TRAIN, 25, score=0.50)
        assert OrienteeringOptimizer()._is_arc_eligible(arc) is True

    def test_arc_ineligible_duree_trop_courte(self):
        src = make_node("87212027", "Strasbourg", 480)
        dst = make_node("87214007", "Sélestat",   483)
        arc = make_arc(src, dst, ArcType.TRAIN, 3, score=0.80)
        assert OrienteeringOptimizer()._is_arc_eligible(arc) is False

    def test_arc_ineligible_score_sous_seuil(self):
        src = make_node("87212027", "Strasbourg", 480)
        dst = make_node("87214007", "Sélestat",   505)
        arc = make_arc(src, dst, ArcType.TRAIN, 25, score=0.05)
        assert OrienteeringOptimizer()._is_arc_eligible(arc) is False

    def test_arc_eligible_score_zero_non_score(self):
        """Un arc à 0.0 (non scoré) doit passer pour préserver la connectivité."""
        src = make_node("87212027", "Strasbourg", 480)
        dst = make_node("87214007", "Sélestat",   505)
        arc = make_arc(src, dst, ArcType.TRAIN, 25, score=0.0)
        assert OrienteeringOptimizer()._is_arc_eligible(arc) is True


# ─────────────────────────────────────────────────────────────────────────────
# TestGreedyHelpers
# ─────────────────────────────────────────────────────────────────────────────

class TestGreedyHelpers:
    """_best_train_arc et _shortest_correspondance sont testables indépendamment."""

    def test_best_train_arc_retourne_meilleur_score(self):
        src = make_node("87212027", "Strasbourg", 480)
        d1  = make_node("87214007", "Sélestat",   505)
        d2  = make_node("87214080", "Colmar",     510)
        arc_low  = make_arc(src, d1, ArcType.TRAIN, 25, score=0.30)
        arc_high = make_arc(src, d2, ArcType.TRAIN, 30, score=0.80)
        result = _best_train_arc([arc_low, arc_high], temps_ecoule=0, duree_max=60)
        assert result == arc_high

    def test_best_train_arc_none_si_hors_budget(self):
        src = make_node("87212027", "Strasbourg", 480)
        dst = make_node("87214007", "Sélestat",   505)
        arc = make_arc(src, dst, ArcType.TRAIN, 25, score=0.80)
        result = _best_train_arc([arc], temps_ecoule=40, duree_max=60)
        assert result is None

    def test_shortest_correspondance_retourne_la_plus_courte(self):
        src  = make_node("87212027", "Strasbourg", 480)
        dst1 = make_node("87212027", "Strasbourg", 490)
        dst2 = make_node("87212027", "Strasbourg", 510)
        corr_courte = make_arc(src, dst1, ArcType.CORRESPONDANCE, 10)
        corr_longue = make_arc(src, dst2, ArcType.CORRESPONDANCE, 30)
        result = _shortest_correspondance(
            [corr_courte, corr_longue], temps_ecoule=0, duree_max=60
        )
        assert result == corr_courte

    def test_shortest_correspondance_none_si_hors_budget(self):
        src = make_node("87212027", "Strasbourg", 480)
        dst = make_node("87212027", "Strasbourg", 510)
        corr = make_arc(src, dst, ArcType.CORRESPONDANCE, 30)
        result = _shortest_correspondance([corr], temps_ecoule=40, duree_max=60)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# TestGreedyWalk
# ─────────────────────────────────────────────────────────────────────────────

class TestGreedyWalk:
    """_greedy_walk() respecte le budget et produit une séquence continue."""

    def test_greedy_walk_budget_respecte(self, graphe_simple):
        noeud_depart = make_node("87212027", "Strasbourg", 6 * 60)
        arcs, _, temps = _greedy_walk(graphe_simple, noeud_depart, 120)
        assert temps <= 120

    def test_greedy_walk_arcs_continus(self, graphe_simple):
        noeud_depart = make_node("87212027", "Strasbourg", 6 * 60)
        arcs, _, _ = _greedy_walk(graphe_simple, noeud_depart, 120)
        for i in range(len(arcs) - 1):
            assert arcs[i].destination == arcs[i + 1].source

    def test_greedy_walk_graphe_vide_retourne_liste_vide(self, graphe_vide):
        noeud_depart = make_node("87212027", "Strasbourg", 8 * 60)
        arcs, score, temps = _greedy_walk(graphe_vide, noeud_depart, 60)
        assert arcs == []
        assert score == 0.0
        assert temps == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestReconstructPath
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructPath:
    """_reconstruct_path() ordonne correctement les arcs actifs CBC."""

    def test_chemin_simple(self):
        a = make_node("87212027", "Strasbourg", 480)
        b = make_node("87214007", "Sélestat",   505)
        c = make_node("87214080", "Colmar",     530)
        arc1 = make_arc(a, b, ArcType.TRAIN, 25, 0.5)
        arc2 = make_arc(b, c, ArcType.TRAIN, 25, 0.7)
        assert _reconstruct_path([arc2, arc1], a) == [arc1, arc2]

    def test_chemin_avec_correspondance(self):
        a  = make_node("87212027", "Strasbourg", 480)
        b  = make_node("87214007", "Sélestat",   505)
        b2 = make_node("87214007", "Sélestat",   515)
        c  = make_node("87214080", "Colmar",     540)
        arc1 = make_arc(a,  b,  ArcType.TRAIN,         25, 0.5)
        corr = make_arc(b,  b2, ArcType.CORRESPONDANCE, 10)
        arc2 = make_arc(b2, c,  ArcType.TRAIN,          25, 0.8)
        assert _reconstruct_path([corr, arc2, arc1], a) == [arc1, corr, arc2]

    def test_chemin_vide_si_depart_absent(self):
        a = make_node("87212027", "Strasbourg", 480)
        b = make_node("87214007", "Sélestat",   505)
        c = make_node("87214080", "Colmar",     530)
        arc = make_arc(b, c, ArcType.TRAIN, 25)
        assert _reconstruct_path([arc], a) == []

    def test_arcs_desordres_reordonnes(self):
        a = make_node("87212027", "Strasbourg", 480)
        b = make_node("87214007", "Sélestat",   505)
        c = make_node("87214080", "Colmar",     530)
        d = make_node("87182063", "Mulhouse",   555)
        arc1 = make_arc(a, b, ArcType.TRAIN, 25)
        arc2 = make_arc(b, c, ArcType.TRAIN, 25)
        arc3 = make_arc(c, d, ArcType.TRAIN, 25)
        for perm in ([arc3, arc1, arc2], [arc2, arc3, arc1]):
            assert _reconstruct_path(perm, a) == [arc1, arc2, arc3]