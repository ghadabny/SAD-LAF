"""
tests/unit/test_optimizer.py — Tests unitaires de OrienteeringOptimizer.

Structure des tests :
    TestBaseOptimizer        → vérification du contrat abstrait
    TestOrienteeringContrat  → le solve() respecte le format de retour
    TestOrienteeringBudget   → contrainte duree_max_minutes toujours respectée
    TestOrienteeringOptimal  → MILP > greedy sur instance où greedy échoue
    TestOrienteeringEdgeCases → robustesse sur graphes dégénérés
    TestReconstructPath      → la reconstruction de chemin est correcte

Stratégie de mocking :
    PuLP n'est pas installé dans tous les environnements CI.
    Les tests marqués @pytest.mark.milp sont conditionnels à pulp disponible.
    Les tests du fallback greedy tournent toujours (pas de dépendance PuLP).

Graphe synthétique :
    Réseau minimal Strasbourg → Sélestat → Colmar avec 3 trains :
        TRAIN_A : Strasbourg 06:00 → Sélestat 06:25   (25 min, score=0.3)
        CORR_1  : Sélestat   06:25 → Sélestat 06:35   (10 min)
        TRAIN_B : Sélestat   06:35 → Colmar   06:58   (23 min, score=0.8)
        TRAIN_C : Strasbourg 06:05 → Colmar   07:10   (65 min, score=0.9)

    Budget 60 min :
        Greedy → prend TRAIN_C (score 0.9, durée 65 min) ❌ dépasse budget
               → revient sur TRAIN_A (score 0.3) car c'est le seul dans budget
        MILP   → prend TRAIN_A + CORR_1 + TRAIN_B (0.3 + 0.8 = 1.1 > 0.9) ✅
"""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from services.or_engine.graph.transition import Arc, ArcType, Node
from services.or_engine.optimizer.base import BaseOptimizer
from services.or_engine.optimizer.orienteering import (
    OrienteeringOptimizer,
    _reconstruct_path,
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
# Helpers
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
    src: Node, dst: Node, arc_type: ArcType,
    duration: int, score: float = 0.0,
    trip_id: str = None, train_number: str = None,
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
# Fixture : graphe synthétique minimal
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def graphe_simple():
    """
    Strasbourg → Sélestat → Colmar avec 3 trains.
    Conçu pour que MILP batte le greedy sur budget de 60 min.

    TRAIN_A : STR 06:00 → SEL 06:25  (25 min, score=0.30)
    CORR_1  : SEL 06:25 → SEL 06:35  (10 min, correspondance)
    TRAIN_B : SEL 06:35 → COL 06:58  (23 min, score=0.80)
    TRAIN_C : STR 06:05 → COL 07:10  (65 min, score=0.90) ← greedy le prend en premier
    """
    str_06h00 = make_node("87212027", "Strasbourg", 6 * 60)
    str_06h05 = make_node("87212027", "Strasbourg", 6 * 60 + 5)
    sel_06h25 = make_node("87214007", "Sélestat",   6 * 60 + 25)
    sel_06h35 = make_node("87214007", "Sélestat",   6 * 60 + 35)
    col_06h58 = make_node("87214080", "Colmar",     6 * 60 + 58)
    col_07h10 = make_node("87214080", "Colmar",     7 * 60 + 10)

    train_a = make_arc(str_06h00, sel_06h25, ArcType.TRAIN, 25, 0.30,
                       trip_id="T_A", train_number="11700")
    corr_1  = make_arc(sel_06h25, sel_06h35, ArcType.CORRESPONDANCE, 10)
    train_b = make_arc(sel_06h35, col_06h58, ArcType.TRAIN, 23, 0.80,
                       trip_id="T_B", train_number="11701")
    train_c = make_arc(str_06h05, col_07h10, ArcType.TRAIN, 65, 0.90,
                       trip_id="T_C", train_number="11702")

    graph = {
        str_06h00: [train_a],
        str_06h05: [train_c],
        sel_06h25: [corr_1],
        sel_06h35: [train_b],
        col_06h58: [],
        col_07h10: [],
    }
    return graph


@pytest.fixture
def graphe_un_seul_train():
    """Graphe avec un seul train disponible."""
    src = make_node("87212027", "Strasbourg", 8 * 60)
    dst = make_node("87214007", "Sélestat",   8 * 60 + 25)
    arc = make_arc(src, dst, ArcType.TRAIN, 25, 0.75, trip_id="T1")
    return {src: [arc], dst: []}


@pytest.fixture
def graphe_vide():
    """Graphe sans aucun arc TRAIN."""
    n = make_node("87212027", "Strasbourg", 8 * 60)
    return {n: []}


# ─────────────────────────────────────────────────────────────────────────────
# TestBaseOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseOptimizer:
    """Vérifie que BaseOptimizer est bien une interface abstraite."""

    def test_cannot_instantiate_directly(self):
        """BaseOptimizer est abstraite — instanciation directe interdite."""
        with pytest.raises(TypeError):
            BaseOptimizer()

    def test_concrete_subclass_must_implement_solve(self):
        """Une sous-classe sans solve() ne peut pas être instanciée."""
        class IncompleteOptimizer(BaseOptimizer):
            pass  # solve() non implémenté

        with pytest.raises(TypeError):
            IncompleteOptimizer()

    def test_valid_subclass_instantiates(self):
        """Une sous-classe avec solve() s'instancie correctement."""
        class MinimalOptimizer(BaseOptimizer):
            def solve(self, graph, gare_depart_id, heure_depart_min, duree_max):
                return {"arcs": [], "score_total": 0.0,
                        "duree_minutes": 0, "nb_trains": 0}

        opt = MinimalOptimizer()
        assert isinstance(opt, BaseOptimizer)


# ─────────────────────────────────────────────────────────────────────────────
# TestOrienteeringContrat
# ─────────────────────────────────────────────────────────────────────────────

class TestOrienteeringContrat:
    """
    Le solve() respecte le contrat de retour défini dans BaseOptimizer,
    que PuLP soit disponible ou non (greedy fallback dans les deux cas).
    """

    def test_retourne_dict_avec_bonnes_cles(self, graphe_simple):
        opt    = OrienteeringOptimizer()
        result = opt.solve(graphe_simple, "87212027", 6 * 60, 120)

        assert isinstance(result, dict)
        for key in ("arcs", "score_total", "duree_minutes", "nb_trains"):
            assert key in result, f"Clé manquante : {key}"

    def test_arcs_est_une_liste(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        assert isinstance(result["arcs"], list)

    def test_score_total_est_float(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        assert isinstance(result["score_total"], float)

    def test_nb_trains_est_int(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        assert isinstance(result["nb_trains"], int)

    def test_nb_trains_coherent_avec_arcs(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        nb_trains_effectif = sum(
            1 for a in result["arcs"] if a.arc_type == ArcType.TRAIN
        )
        assert result["nb_trains"] == nb_trains_effectif

    def test_score_total_coherent_avec_arcs(self, graphe_simple):
        result = OrienteeringOptimizer().solve(graphe_simple, "87212027", 6 * 60, 120)
        score_calcule = round(sum(
            a.fraud_score for a in result["arcs"] if a.arc_type == ArcType.TRAIN
        ), 4)
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

    def test_duree_inferieure_ou_egale_au_budget(self, graphe_simple):
        budget = 60
        result = OrienteeringOptimizer().solve(
            graphe_simple, "87212027", 6 * 60, budget
        )
        assert result["duree_minutes"] <= budget, (
            f"Durée {result['duree_minutes']} > budget {budget}"
        )

    def test_budget_serré_un_seul_train(self, graphe_simple):
        """Budget de 30 min : seul TRAIN_A (25 min) est faisable."""
        result = OrienteeringOptimizer().solve(
            graphe_simple, "87212027", 6 * 60, 30
        )
        assert result["duree_minutes"] <= 30
        # TRAIN_C (65 min) ne peut pas être pris
        trip_ids = [a.trip_id for a in result["arcs"] if a.arc_type == ArcType.TRAIN]
        assert "T_C" not in trip_ids

    def test_budget_large_plusieurs_trains(self, graphe_simple):
        """Budget de 120 min : plusieurs trains possibles."""
        result = OrienteeringOptimizer().solve(
            graphe_simple, "87212027", 6 * 60, 120
        )
        assert result["duree_minutes"] <= 120
        assert result["nb_trains"] >= 1

    def test_continuité_des_arcs(self, graphe_simple):
        """
        Chaque arc doit avoir sa source = destination de l'arc précédent.
        Vérifie la faisabilité physique de la tournée.
        """
        result = OrienteeringOptimizer().solve(
            graphe_simple, "87212027", 6 * 60, 120
        )
        arcs = result["arcs"]
        for i in range(len(arcs) - 1):
            assert arcs[i].destination == arcs[i + 1].source, (
                f"Discontinuité entre arc {i} et {i+1} : "
                f"{arcs[i].destination.label} ≠ {arcs[i+1].source.label}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TestOrienteeringOptimal (nécessite PuLP)
# ─────────────────────────────────────────────────────────────────────────────

class TestOrienteeringOptimal:
    """
    MILP bat le greedy sur l'instance conçue pour ça.

    Instance :
        TRAIN_A 25 min score=0.30  +  CORR 10 min  +  TRAIN_B 23 min score=0.80
        → score MILP = 1.10, durée = 58 min ✅

        TRAIN_C 65 min score=0.90
        → score greedy = 0.90 si budget ≥ 65, sinon greedy prend TRAIN_A
        → avec budget 60 min : TRAIN_C est hors budget → greedy = 0.30

        Avec budget 60 min :
            MILP   = 1.10  (TRAIN_A + CORR + TRAIN_B)
            Greedy = 0.30  (TRAIN_A seulement — TRAIN_C hors budget)
    """

    @requires_pulp
    def test_milp_score_superieur_greedy_sur_instance_adverse(self, graphe_simple):
        """MILP ≥ greedy sur l'instance conçue pour piéger le greedy."""
        budget = 60

        # Score MILP
        opt_milp   = OrienteeringOptimizer(time_limit_seconds=30)
        result_milp = opt_milp.solve(graphe_simple, "87212027", 6 * 60, budget)

        # Score greedy (via fallback forcé)
        opt_greedy    = OrienteeringOptimizer()
        result_greedy = opt_greedy._greedy_fallback(
            graphe_simple, "87212027", 6 * 60, budget
        )

        assert result_milp["score_total"] >= result_greedy["score_total"], (
            f"MILP ({result_milp['score_total']:.4f}) < "
            f"greedy ({result_greedy['score_total']:.4f}) — formulation incorrecte"
        )

    @requires_pulp
    def test_milp_trouve_deux_trains_sur_instance_adverse(self, graphe_simple):
        """MILP doit trouver TRAIN_A + TRAIN_B (2 trains) sur budget 60 min."""
        result = OrienteeringOptimizer(time_limit_seconds=30).solve(
            graphe_simple, "87212027", 6 * 60, 60
        )
        assert result["nb_trains"] == 2, (
            f"MILP n'a trouvé que {result['nb_trains']} train(s) "
            f"alors qu'on attendait 2 (TRAIN_A + TRAIN_B)"
        )

    @requires_pulp
    def test_milp_score_optimal_connu(self, graphe_simple):
        """Sur budget 60 min, le score optimal est 0.30 + 0.80 = 1.10."""
        result = OrienteeringOptimizer(time_limit_seconds=30).solve(
            graphe_simple, "87212027", 6 * 60, 60
        )
        assert abs(result["score_total"] - 1.10) < 1e-3, (
            f"Score MILP = {result['score_total']:.4f}, attendu 1.10"
        )

    @requires_pulp
    def test_milp_budget_suffisant_prend_train_c(self, graphe_simple):
        """Sur budget 120 min, TRAIN_C (65 min, score 0.90) devient accessible."""
        result = OrienteeringOptimizer(time_limit_seconds=30).solve(
            graphe_simple, "87212027", 6 * 60, 120
        )
        # Score optimal : TRAIN_A + CORR + TRAIN_B + (pas assez de temps pour C)
        # ou TRAIN_C seul — le MILP choisit le meilleur
        assert result["score_total"] >= 1.10 or result["score_total"] >= 0.90


# ─────────────────────────────────────────────────────────────────────────────
# TestOrienteeringEdgeCases
# ─────────────────────────────────────────────────────────────────────────────

class TestOrienteeringEdgeCases:
    """Robustesse sur les cas limites."""

    def test_gare_inconnue_leve_value_error(self, graphe_simple):
        """Une gare absente du graphe doit lever ValueError."""
        with pytest.raises(ValueError, match="Aucun train"):
            OrienteeringOptimizer().solve(
                graphe_simple, "00000000", 6 * 60, 120
            )

    def test_heure_depart_trop_tard_leve_value_error(self, graphe_simple):
        """Aucun train disponible après 23:00 → ValueError."""
        with pytest.raises(ValueError):
            OrienteeringOptimizer().solve(
                graphe_simple, "87212027", 23 * 60, 60
            )

    def test_graphe_sans_train_leve_value_error(self, graphe_vide):
        """Un graphe sans arc TRAIN → ValueError."""
        with pytest.raises(ValueError):
            OrienteeringOptimizer().solve(
                graphe_vide, "87212027", 8 * 60, 60
            )

    def test_budget_tres_serre_pas_de_train(self, graphe_simple):
        """Budget de 5 min : aucun train faisable (tous > 5 min) → ValueError."""
        with pytest.raises(ValueError):
            OrienteeringOptimizer().solve(
                graphe_simple, "87212027", 6 * 60, 5
            )

    def test_tous_scores_nuls_retourne_quand_meme_une_tournee(self):
        """Même avec fraud_score=0.0 partout, on doit retourner une tournée."""
        src = make_node("87212027", "Strasbourg", 8 * 60)
        dst = make_node("87214007", "Sélestat",   8 * 60 + 25)
        arc = make_arc(src, dst, ArcType.TRAIN, 25, 0.0, trip_id="T1")
        graph = {src: [arc], dst: []}

        result = OrienteeringOptimizer().solve(graph, "87212027", 8 * 60, 60)
        assert result["nb_trains"] == 1
        assert result["score_total"] == 0.0

    def test_fallback_greedy_si_pulp_absent(self, graphe_simple):
        """Si PuLP n'est pas importable, solve() bascule sur le greedy."""
        with patch("builtins.__import__", side_effect=lambda name, *a, **k:
                   (_ for _ in ()).throw(ImportError("no module named pulp"))
                   if name == "pulp" else __import__(name, *a, **k)):
            # On force l'ImportError à l'import de pulp dans solve()
            result = OrienteeringOptimizer().solve(
                graphe_simple, "87212027", 6 * 60, 120
            )
        assert isinstance(result, dict)
        assert "arcs" in result
        assert result["nb_trains"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# TestReconstructPath
# ─────────────────────────────────────────────────────────────────────────────

class TestReconstructPath:
    """
    _reconstruct_path() ordonne correctement les arcs actifs CBC.
    """

    def test_chemin_simple_deux_arcs(self):
        """A→B→C : reconstruction dans l'ordre."""
        a = make_node("87212027", "Strasbourg", 8 * 60)
        b = make_node("87214007", "Sélestat",   8 * 60 + 25)
        c = make_node("87214080", "Colmar",     8 * 60 + 50)

        arc1 = make_arc(a, b, ArcType.TRAIN, 25, 0.5)
        arc2 = make_arc(b, c, ArcType.TRAIN, 25, 0.7)

        # CBC retourne dans l'ordre inverse (peu importe)
        result = _reconstruct_path([arc2, arc1], a)
        assert result == [arc1, arc2]

    def test_chemin_avec_correspondance(self):
        """A→B (TRAIN) + B→B' (CORR) + B'→C (TRAIN)."""
        a  = make_node("87212027", "Strasbourg", 8 * 60)
        b  = make_node("87214007", "Sélestat",   8 * 60 + 25)
        b2 = make_node("87214007", "Sélestat",   8 * 60 + 35)
        c  = make_node("87214080", "Colmar",     9 * 60)

        arc1 = make_arc(a,  b,  ArcType.TRAIN,         25, 0.5)
        corr = make_arc(b,  b2, ArcType.CORRESPONDANCE, 10)
        arc2 = make_arc(b2, c,  ArcType.TRAIN,          25, 0.8)

        result = _reconstruct_path([corr, arc2, arc1], a)
        assert result == [arc1, corr, arc2]

    def test_chemin_vide_si_depart_absent(self):
        """Si noeud_depart n'est dans aucun arc, résultat vide."""
        a = make_node("87212027", "Strasbourg", 8 * 60)
        b = make_node("87214007", "Sélestat",   8 * 60 + 25)
        c = make_node("87214080", "Colmar",     9 * 60)

        arc = make_arc(b, c, ArcType.TRAIN, 25)
        # a n'est pas source d'arc → chemin vide
        result = _reconstruct_path([arc], a)
        assert result == []

    def test_arcs_desordres_sont_reordres(self):
        """L'ordre d'entrée des arcs dans la liste ne doit pas compter."""
        a = make_node("87212027", "Strasbourg", 8 * 60)
        b = make_node("87214007", "Sélestat",   8 * 60 + 25)
        c = make_node("87214080", "Colmar",     8 * 60 + 50)
        d = make_node("87182063", "Mulhouse",   9 * 60 + 15)

        arc1 = make_arc(a, b, ArcType.TRAIN, 25)
        arc2 = make_arc(b, c, ArcType.TRAIN, 25)
        arc3 = make_arc(c, d, ArcType.TRAIN, 25)

        for permutation in ([arc3, arc1, arc2], [arc2, arc3, arc1],
                            [arc1, arc3, arc2]):
            result = _reconstruct_path(permutation, a)
            assert result == [arc1, arc2, arc3], (
                f"Mauvais ordre pour permutation {[id(a) for a in permutation]}"
            )