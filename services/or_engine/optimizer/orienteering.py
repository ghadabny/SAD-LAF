# services/or_engine/optimizer/orienteering.py
"""
Solveur MILP pour le Problème d'Orienteering ferroviaire LAF.

Formulation mathématique :
─────────────────────────────────────────────────────────────────────────────
Variables de décision :
    x[i] ∈ {0, 1}  — 1 si l'arc i est emprunté dans la tournée

Objectif (maximisation) :
    max  Σ  fraud_score[i] × x[i]      (pour tout arc i de type TRAIN)

Contraintes :
    C1 — Budget temps :
         Σ  duration_min[i] × x[i]  ≤  duree_max_minutes

    C2 — Continuité de flux à chaque nœud intermédiaire :
         Σ x[arc entrant vers n]  =  Σ x[arc sortant de n]

    C3 — Source unique :
         Σ x[arc sortant de noeud_depart]  =  1

    C4 — Domaine des variables :
         x[i] ∈ {0, 1}

Solveur : CBC (COIN-OR Branch and Cut) via PuLP — open source, sans licence.

Dégradation gracieuse :
    Si PuLP/CBC n'est pas installé → fallback automatique vers le greedy.
"""
from __future__ import annotations

import logging
from typing import Optional

from services.or_engine.graph.transition import Arc, ArcType, Node
from services.or_engine.optimizer.base import BaseOptimizer
from shared.constants import MIN_BOARD_DURATION_MINUTES, MIN_FRAUD_SCORE_THRESHOLD

logger = logging.getLogger(__name__)


class OrienteeringOptimizer(BaseOptimizer):
    """
    Implémentation MILP du Problème d'Orienteering ferroviaire.

    Hérite de BaseOptimizer — solver.run() peut l'utiliser sans
    connaître son type concret (principe D de SOLID).

    SOLID — principe S :
        Chaque méthode privée a une responsabilité unique et documentée.
        _solve_milp() orchestre ; les étapes sont déléguées à des méthodes
        dédiées : _build_subgraph, _build_model, _build_flow_index,
        _add_objective, _add_budget_constraint, _add_flow_constraints,
        _add_source_constraint, _run_cbc, _extract_active_arcs,
        _build_result.

    Usage :
        optimizer = OrienteeringOptimizer(time_limit_seconds=30)
        result    = optimizer.solve(graph, "87212027", 480, 360)
    """

    def __init__(self, time_limit_seconds: int = 30):
        self.time_limit_seconds = time_limit_seconds

    # ── Interface publique ────────────────────────────────────────────────────

    def solve(
        self,
        graph: dict[Node, list[Arc]],
        gare_depart_id: str,
        heure_depart_min: int,
        duree_max_minutes: int,
    ) -> dict:
        """
        Résout le problème d'Orienteering via MILP (PuLP/CBC).
        Bascule automatiquement sur le greedy si PuLP n'est pas disponible.
        """
        try:
            import pulp  # noqa: F401
        except ImportError:
            logger.warning(
                "[OrienteeringOptimizer] PuLP non installé — fallback greedy."
            )
            return self._greedy_fallback(
                graph, gare_depart_id, heure_depart_min, duree_max_minutes
            )
        return self._solve_milp(
            graph, gare_depart_id, heure_depart_min, duree_max_minutes
        )

    # ── Orchestration MILP ────────────────────────────────────────────────────

    def _solve_milp(
        self,
        graph: dict[Node, list[Arc]],
        gare_depart_id: str,
        heure_depart_min: int,
        duree_max_minutes: int,
    ) -> dict:
        """
        Orchestre les 5 étapes de la résolution MILP.
        Chaque étape est déléguée à une méthode dédiée (principe S).
        """
        import pulp

        noeud_depart, arcs_accessibles = self._build_subgraph(
            graph, gare_depart_id, heure_depart_min, duree_max_minutes
        )

        model, x, arc_ids = self._build_model(arcs_accessibles)
        idx_sortants, idx_entrants = self._build_flow_index(arcs_accessibles)

        self._add_objective(model, x, arc_ids, arcs_accessibles)
        self._add_budget_constraint(model, x, arc_ids, arcs_accessibles, duree_max_minutes)
        self._add_flow_constraints(model, x, idx_sortants, idx_entrants, noeud_depart)
        self._add_source_constraint(model, x, idx_sortants, noeud_depart)

        status = self._run_cbc(model, pulp)

        if not self._is_feasible(status, pulp):
            logger.warning(
                "[OrienteeringOptimizer] CBC infaisable (status=%s) — fallback greedy.",
                pulp.LpStatus[status],
            )
            return self._greedy_fallback(
                graph, gare_depart_id, heure_depart_min, duree_max_minutes
            )

        arcs_actifs = self._extract_active_arcs(x, arc_ids, arcs_accessibles, pulp)
        if not arcs_actifs:
            logger.warning("[OrienteeringOptimizer] Solution CBC vide — fallback greedy.")
            return self._greedy_fallback(
                graph, gare_depart_id, heure_depart_min, duree_max_minutes
            )

        return self._build_result(arcs_actifs, noeud_depart)

    # ── Étape 1 : construction du sous-graphe ─────────────────────────────────

    def _build_subgraph(
        self,
        graph: dict[Node, list[Arc]],
        gare_depart_id: str,
        heure_depart_min: int,
        duree_max_minutes: int,
    ) -> tuple[Node, list[Arc]]:
        """
        Identifie le nœud de départ et filtre le sous-graphe accessible.
        Lève ValueError si aucun arc exploitable n'est trouvé.
        """
        noeud_depart = self._find_depart_node(graph, gare_depart_id, heure_depart_min)
        if noeud_depart is None:
            raise ValueError(
                f"[OrienteeringOptimizer] Aucun train au départ de la gare "
                f"{gare_depart_id} après "
                f"{heure_depart_min // 60:02d}h{heure_depart_min % 60:02d}."
            )

        arcs_accessibles = self._filter_reachable_arcs(
            graph, noeud_depart, duree_max_minutes
        )
        if not arcs_accessibles:
            raise ValueError(
                f"[OrienteeringOptimizer] Aucun arc accessible depuis "
                f"{gare_depart_id} dans un budget de {duree_max_minutes} min."
            )

        arcs_train = [a for a in arcs_accessibles if a.arc_type == ArcType.TRAIN]
        if not arcs_train:
            raise ValueError(
                f"[OrienteeringOptimizer] Aucun tronçon contrôlable depuis "
                f"{gare_depart_id} dans un budget de {duree_max_minutes} min."
            )

        logger.info(
            "[OrienteeringOptimizer] Sous-graphe : %d arcs (%d TRAIN, %d CORR).",
            len(arcs_accessibles), len(arcs_train),
            len(arcs_accessibles) - len(arcs_train),
        )
        return noeud_depart, arcs_accessibles

    # ── Étape 2 : construction du modèle PuLP ─────────────────────────────────

    def _build_model(self, arcs_accessibles: list[Arc]):
        """
        Crée le modèle PuLP et les variables binaires x[i].
        Retourne (model, x, arc_ids).
        """
        import pulp
        model   = pulp.LpProblem("orienteering_laf", pulp.LpMaximize)
        arc_ids = list(range(len(arcs_accessibles)))
        x       = pulp.LpVariable.dicts("x", arc_ids, cat="Binary")
        return model, x, arc_ids

    # ── Étape 3 : index de flux ────────────────────────────────────────────────

    def _build_flow_index(
        self,
        arcs_accessibles: list[Arc],
    ) -> tuple[dict[Node, list[int]], dict[Node, list[int]]]:
        """
        Construit les index node → indices des arcs sortants/entrants.
        Utilisé par les contraintes de flux C2 et C3.
        """
        idx_sortants: dict[Node, list[int]] = {}
        idx_entrants: dict[Node, list[int]] = {}
        for i, arc in enumerate(arcs_accessibles):
            idx_sortants.setdefault(arc.source,      []).append(i)
            idx_entrants.setdefault(arc.destination, []).append(i)
        return idx_sortants, idx_entrants

    # ── Étape 4a : objectif ───────────────────────────────────────────────────

    def _add_objective(self, model, x, arc_ids, arcs_accessibles):
        """
        Objectif : maximiser la somme des fraud_score sur les arcs TRAIN.
        """
        import pulp
        model += pulp.lpSum(
            arcs_accessibles[i].fraud_score * x[i]
            for i in arc_ids
            if arcs_accessibles[i].arc_type == ArcType.TRAIN
        ), "objectif_score_fraude"

    # ── Étape 4b : contrainte budget ──────────────────────────────────────────

    def _add_budget_constraint(
        self, model, x, arc_ids, arcs_accessibles, duree_max_minutes: int
    ):
        """
        C1 — Budget temps : Σ duration × x[i] ≤ duree_max_minutes.
        """
        import pulp
        model += (
            pulp.lpSum(arcs_accessibles[i].duration_min * x[i] for i in arc_ids)
            <= duree_max_minutes
        ), "budget_temps"

    # ── Étape 4c : contraintes de flux ────────────────────────────────────────

    def _add_flow_constraints(
        self,
        model,
        x,
        idx_sortants: dict[Node, list[int]],
        idx_entrants: dict[Node, list[int]],
        noeud_depart: Node,
    ):
        """
        C2 — Continuité de flux à chaque nœud intermédiaire (≠ départ) :
             Σ x[entrants vers n] = Σ x[sortants de n]
        """
        import pulp
        tous_noeuds = set(idx_sortants.keys()) | set(idx_entrants.keys())
        for node in tous_noeuds:
            if node == noeud_depart:
                continue
            entrants = pulp.lpSum(x[i] for i in idx_entrants.get(node, []))
            sortants = pulp.lpSum(x[i] for i in idx_sortants.get(node, []))
            model += (entrants == sortants), f"flux_{hash(node) % 10**9}"

    # ── Étape 4d : contrainte source ──────────────────────────────────────────

    def _add_source_constraint(
        self,
        model,
        x,
        idx_sortants: dict[Node, list[int]],
        noeud_depart: Node,
    ):
        """
        C3 — Source unique : exactement un arc sortant du nœud de départ.
        """
        import pulp
        sortants_depart = idx_sortants.get(noeud_depart, [])
        if not sortants_depart:
            raise ValueError(
                f"[OrienteeringOptimizer] Nœud de départ {noeud_depart.label} "
                "sans arcs sortants dans le sous-graphe filtré."
            )
        model += (
            pulp.lpSum(x[i] for i in sortants_depart) == 1
        ), "source_unique"

    # ── Étape 5 : résolution CBC ──────────────────────────────────────────────

    def _run_cbc(self, model, pulp) -> int:
        """Lance CBC et retourne le statut de résolution."""
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit_seconds)
        status = model.solve(solver)
        logger.info(
            "[OrienteeringOptimizer] CBC status : %s | score : %.4f",
            pulp.LpStatus[status],
            pulp.value(model.objective) or 0.0,
        )
        return status

    def _is_feasible(self, status: int, pulp) -> bool:
        """Retourne True si CBC a trouvé une solution entière faisable."""
        return status in (
            pulp.constants.LpSolutionOptimal,
            pulp.constants.LpSolutionIntegerFeasible,
        )

    # ── Étape 6 : extraction et reconstruction ────────────────────────────────

    def _extract_active_arcs(
        self, x, arc_ids, arcs_accessibles, pulp
    ) -> list[Arc]:
        """Extrait les arcs dont la variable x[i] = 1 dans la solution CBC."""
        return [
            arcs_accessibles[i]
            for i in arc_ids
            if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5
        ]

    def _build_result(self, arcs_actifs: list[Arc], noeud_depart: Node) -> dict:
        """Reconstruit le chemin ordonné et calcule les métriques de la tournée."""
        arcs_ordonnes = _reconstruct_path(arcs_actifs, noeud_depart)
        score_total   = sum(
            a.fraud_score for a in arcs_ordonnes if a.arc_type == ArcType.TRAIN
        )
        duree_totale = sum(a.duration_min for a in arcs_ordonnes)
        nb_trains    = sum(1 for a in arcs_ordonnes if a.arc_type == ArcType.TRAIN)

        logger.info(
            "[OrienteeringOptimizer] ✅ MILP : %d trains | score=%.4f | durée=%dmin",
            nb_trains, score_total, duree_totale,
        )
        return {
            "arcs":          arcs_ordonnes,
            "score_total":   round(score_total, 4),
            "duree_minutes": duree_totale,
            "nb_trains":     nb_trains,
        }

    # ── Helpers BFS ───────────────────────────────────────────────────────────

    def _find_depart_node(
        self,
        graph: dict[Node, list[Arc]],
        gare_depart_id: str,
        heure_depart_min: int,
    ) -> Optional[Node]:
        """Premier nœud disponible dans la gare à partir de heure_depart_min."""
        candidats = [
            n for n in graph
            if n.stop_id == gare_depart_id
            and n.time_minutes >= heure_depart_min
        ]
        if not candidats:
            return None
        return min(candidats, key=lambda n: n.time_minutes)

    def _filter_reachable_arcs(
        self,
        graph: dict[Node, list[Arc]],
        noeud_depart: Node,
        duree_max_minutes: int,
    ) -> list[Arc]:
        """
        BFS temporel depuis noeud_depart dans le budget duree_max_minutes.

        Filtres sur les arcs TRAIN :
            - duration_min < MIN_BOARD_DURATION_MINUTES → contrôle impossible
            - 0 < fraud_score < MIN_FRAUD_SCORE_THRESHOLD → train sans intérêt LAF
              Les arcs à 0.0 (non scorés) sont conservés pour la connectivité ;
              CBC les ignorera via l'objectif.
        """
        heure_limite = noeud_depart.time_minutes + duree_max_minutes
        visites:     set[Node]  = set()
        a_explorer:  list[Node] = [noeud_depart]
        arcs_result: list[Arc]  = []

        while a_explorer:
            noeud = a_explorer.pop()
            if noeud in visites:
                continue
            visites.add(noeud)

            for arc in graph.get(noeud, []):
                if arc.destination.time_minutes > heure_limite:
                    continue
                if arc.arc_type == ArcType.TRAIN and not self._is_arc_eligible(arc):
                    continue
                arcs_result.append(arc)
                if arc.destination not in visites:
                    a_explorer.append(arc.destination)

        return arcs_result

    def _is_arc_eligible(self, arc: Arc) -> bool:
        """
        Retourne True si un arc TRAIN est éligible au graphe MILP.
        Responsabilité unique : encapsuler les critères de filtrage.
        """
        if arc.duration_min < MIN_BOARD_DURATION_MINUTES:
            return False
        if 0.0 < arc.fraud_score < MIN_FRAUD_SCORE_THRESHOLD:
            return False
        return True

    # ── Greedy fallback ───────────────────────────────────────────────────────

    def _greedy_fallback(
        self,
        graph: dict[Node, list[Arc]],
        gare_depart_id: str,
        heure_depart_min: int,
        duree_max_minutes: int,
    ) -> dict:
        """
        Greedy de secours — activé si PuLP absent ou CBC infaisable.
        Garantit qu'OrienteeringOptimizer retourne toujours une réponse.
        """
        noeud_depart = self._find_depart_node(graph, gare_depart_id, heure_depart_min)
        if noeud_depart is None:
            raise ValueError(
                f"[OrienteeringOptimizer] Aucun train depuis {gare_depart_id} "
                f"après {heure_depart_min // 60:02d}h{heure_depart_min % 60:02d}."
            )

        arcs_tournee, score_total, temps_ecoule = _greedy_walk(
            graph, noeud_depart, duree_max_minutes
        )

        if not arcs_tournee:
            raise ValueError(
                f"[OrienteeringOptimizer] Tournée impossible depuis {gare_depart_id} "
                f"avec un budget de {duree_max_minutes} min."
            )

        nb_trains = sum(1 for a in arcs_tournee if a.arc_type == ArcType.TRAIN)
        return {
            "arcs":          arcs_tournee,
            "score_total":   round(score_total, 4),
            "duree_minutes": temps_ecoule,
            "nb_trains":     nb_trains,
        }


# ── Fonctions utilitaires module-level ────────────────────────────────────────
# Module-level pour être testables indépendamment de la classe (principe S).

def _reconstruct_path(arcs_actifs: list[Arc], noeud_depart: Node) -> list[Arc]:
    """
    Reconstruit la séquence ordonnée d'arcs depuis le set des arcs actifs CBC.

    CBC retourne un ensemble non ordonné — on réordonne en chaîne
    en suivant source → destination depuis noeud_depart.
    """
    index: dict[Node, Arc] = {arc.source: arc for arc in arcs_actifs}
    chemin:   list[Arc]  = []
    visites:  set[Node]  = set()
    noeud_courant = noeud_depart

    while noeud_courant in index and noeud_courant not in visites:
        visites.add(noeud_courant)
        arc = index[noeud_courant]
        chemin.append(arc)
        noeud_courant = arc.destination

    return chemin


def _greedy_walk(
    graph: dict[Node, list[Arc]],
    noeud_depart: Node,
    duree_max_minutes: int,
) -> tuple[list[Arc], float, int]:
    """
    Parcours greedy depuis noeud_depart.

    À chaque étape : prend le meilleur arc TRAIN disponible dans le budget,
    sinon la correspondance la plus courte, sinon s'arrête.

    Retourne (arcs_tournee, score_total, temps_ecoule).
    Séparé de _greedy_fallback pour être testable indépendamment.
    """
    noeud_courant = noeud_depart
    temps_ecoule  = 0
    arcs_tournee: list[Arc] = []
    score_total   = 0.0

    while temps_ecoule < duree_max_minutes:
        arcs_dispo = graph.get(noeud_courant, [])
        if not arcs_dispo:
            break

        meilleur_train = _best_train_arc(arcs_dispo, temps_ecoule, duree_max_minutes)
        if meilleur_train:
            arcs_tournee.append(meilleur_train)
            score_total   += meilleur_train.fraud_score
            temps_ecoule  += meilleur_train.duration_min
            noeud_courant  = meilleur_train.destination
            continue

        corr = _shortest_correspondance(arcs_dispo, temps_ecoule, duree_max_minutes)
        if corr:
            arcs_tournee.append(corr)
            temps_ecoule  += corr.duration_min
            noeud_courant  = corr.destination
        else:
            break

    return arcs_tournee, score_total, temps_ecoule


def _best_train_arc(
    arcs: list[Arc],
    temps_ecoule: int,
    duree_max: int,
) -> Optional[Arc]:
    """Retourne l'arc TRAIN avec le meilleur score dans le budget restant."""
    candidats = [
        a for a in arcs
        if a.arc_type == ArcType.TRAIN
        and temps_ecoule + a.duration_min <= duree_max
    ]
    return max(candidats, key=lambda a: a.fraud_score) if candidats else None


def _shortest_correspondance(
    arcs: list[Arc],
    temps_ecoule: int,
    duree_max: int,
) -> Optional[Arc]:
    """Retourne la correspondance la plus courte dans le budget restant."""
    candidats = [
        a for a in arcs
        if a.arc_type == ArcType.CORRESPONDANCE
        and temps_ecoule + a.duration_min <= duree_max
    ]
    return min(candidats, key=lambda a: a.duration_min) if candidats else None