# services/or_engine/optimizer/orienteering.py
"""
Solveur MILP pour le Problème d'Orienteering ferroviaire LAF.

Formulation mathématique (v2 — avec contrainte de retour optionnelle) :
─────────────────────────────────────────────────────────────────────────────
Variables de décision :
    x[i] ∈ {0, 1}  — 1 si l'arc i est emprunté dans la tournée

Objectif (maximisation) :
    max  Σ  fraud_score[i] × x[i]      (pour tout arc i de type TRAIN)

Contraintes :
    C1 — Budget temps :
         Σ  duration_min[i] × x[i]  ≤  duree_max_minutes

    C2a — Continuité de flux (nœuds intermédiaires, ≠ source, ≠ sink) :
          Σ x[arc entrant vers n]  =  Σ x[arc sortant de n]

    C2b — Sink agrégé (si gare_arrivee_id fourni) :
          Σ x[entrants vers n ∈ Sink] − Σ x[sortants de n ∈ Sink]  =  1
          où Sink = {n | n.stop_id == gare_arrivee_id}

    C3  — Source unique :
          Σ x[arc sortant de noeud_depart]  =  1

    C4  — Domaine des variables :
          x[i] ∈ {0, 1}

Solveur : CBC (COIN-OR Branch and Cut) via PuLP — open source, sans licence.
Dégradation gracieuse : fallback greedy si PuLP/CBC absent ou infaisable.
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

    Nouveauté v2 : paramètre gare_arrivee_id optionnel dans solve().
    Si fourni, le chemin DOIT terminer dans cette gare (contrainte C2b).
    Le solveur lève ValueError si aucune solution n'est feasible avec
    cette contrainte — le router doit en informer l'agent.
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
        gare_arrivee_id: Optional[str] = None,
        excluded_trip_ids: Optional[set] = None,
    ) -> dict:
        """
        Résout le problème d'Orienteering via MILP (PuLP/CBC).
        Bascule automatiquement sur le greedy si PuLP n'est pas disponible.

        Paramètres :
            gare_arrivee_id : code UIC 8 chiffres de la gare d'arrivée cible.
                              None = pas de contrainte de fin (comportement v1).
                              Si fourni et aucune solution possible → ValueError.
        """
        try:
            import pulp  # noqa: F401
        except ImportError:
            logger.warning("[OrienteeringOptimizer] PuLP non installé — fallback greedy.")
            return self._greedy_fallback(
                graph, gare_depart_id, heure_depart_min, duree_max_minutes, gare_arrivee_id,
                excluded_trip_ids=excluded_trip_ids,
            )
        return self._solve_milp(
            graph, gare_depart_id, heure_depart_min, duree_max_minutes, gare_arrivee_id,
            excluded_trip_ids=excluded_trip_ids,
        )

    # ── Orchestration MILP ────────────────────────────────────────────────────

    def _solve_milp(
        self,
        graph: dict[Node, list[Arc]],
        gare_depart_id: str,
        heure_depart_min: int,
        duree_max_minutes: int,
        gare_arrivee_id: Optional[str],
        excluded_trip_ids: Optional[set] = None,
    ) -> dict:
        import pulp

        noeud_depart, arcs_accessibles = self._build_subgraph(
            graph, gare_depart_id, heure_depart_min, duree_max_minutes
        )

        # Validation des nœuds sink si contrainte de retour
        sink_nodes: set[Node] = set()
        if gare_arrivee_id is not None:
            sink_nodes = self._find_sink_nodes(arcs_accessibles, gare_arrivee_id)

            # IMPORTANT : le nœud de départ ne peut jamais être un nœud final —
            # l'agent part de là, il n'y revient qu'à un instant ULTÉRIEUR.
            # Dans le graphe temps-étendu, le retour à la même gare est représenté
            # par un nœud distinct (même stop_id, heure différente). Inclure
            # noeud_depart dans sink_nodes crée une contradiction MILP :
            #   source_unique: x[arc_départ] = 1
            #   sink: x[arc_retour] - x[arc_départ] = 1  → x[arc_retour] = 2 (impossible)
            sink_nodes = sink_nodes - {noeud_depart}

            if not sink_nodes:
                raise ValueError(
                    f"[OrienteeringOptimizer] Aucun train n'arrive à {gare_arrivee_id} "
                    f"dans le budget de {duree_max_minutes} min depuis {gare_depart_id}. "
                    f"Essayez d'élargir la fenêtre PS/FS ou de changer la gare d'arrivée."
                )

        model, x, arc_ids = self._build_model(arcs_accessibles)
        idx_sortants, idx_entrants = self._build_flow_index(arcs_accessibles)

        self._add_objective(model, x, arc_ids, arcs_accessibles, excluded_trip_ids)
        self._add_budget_constraint(model, x, arc_ids, arcs_accessibles, duree_max_minutes)
        self._add_min_duration_constraint(model, x, arc_ids, arcs_accessibles, duree_max_minutes)
        self._add_flow_constraints(
            model, x, idx_sortants, idx_entrants, noeud_depart, sink_nodes
        )
        self._add_source_constraint(model, x, idx_sortants, noeud_depart)
        if sink_nodes:
            self._add_sink_constraint(model, x, idx_sortants, idx_entrants, sink_nodes)
            self._add_min_trains_constraint(model, x, arc_ids, arcs_accessibles)

        status = self._run_cbc(model, pulp)

        if not self._is_feasible(status, pulp):
            if gare_arrivee_id is not None:
                raise ValueError(
                    f"[OrienteeringOptimizer] Impossible de construire une tournée "
                    f"qui revient à {gare_arrivee_id} dans {duree_max_minutes} min. "
                    f"Modifiez les paramètres (heure PS/FS ou gare d'arrivée)."
                )
            logger.warning(
                "[OrienteeringOptimizer] CBC infaisable (status=%s) — fallback greedy.",
                pulp.LpStatus[status],
            )
            return self._greedy_fallback(
                graph, gare_depart_id, heure_depart_min, duree_max_minutes, gare_arrivee_id,
                excluded_trip_ids=excluded_trip_ids,
            )

        arcs_actifs = self._extract_active_arcs(x, arc_ids, arcs_accessibles, pulp)
        if not arcs_actifs:
            if gare_arrivee_id is not None:
                raise ValueError(
                    f"[OrienteeringOptimizer] Solution CBC vide avec contrainte de retour "
                    f"sur {gare_arrivee_id}. Vérifiez les paramètres."
                )
            logger.warning("[OrienteeringOptimizer] Solution CBC vide — fallback greedy.")
            return self._greedy_fallback(
                graph, gare_depart_id, heure_depart_min, duree_max_minutes, gare_arrivee_id,
                excluded_trip_ids=excluded_trip_ids,
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
        noeud_depart = self._find_depart_node(graph, gare_depart_id, heure_depart_min)
        if noeud_depart is None:
            raise ValueError(
                f"[OrienteeringOptimizer] Aucun train au départ de la gare "
                f"{gare_depart_id} après "
                f"{heure_depart_min // 60:02d}h{heure_depart_min % 60:02d}."
            )

        arcs_accessibles = self._filter_reachable_arcs(graph, noeud_depart, duree_max_minutes)
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

    # ── Étape 1b : identification des nœuds sink ──────────────────────────────

    def _find_sink_nodes(
        self,
        arcs_accessibles: list[Arc],
        gare_arrivee_id: str,
    ) -> set[Node]:
        """
        Retourne l'ensemble des nœuds du sous-graphe dont stop_id == gare_arrivee_id.

        Un arc arrive à un sink si son destination.stop_id == gare_arrivee_id.
        Ces nœuds représentent "être à la gare d'arrivée à un instant quelconque".

        Si ce set est vide, la contrainte de retour est physiquement impossible.
        """
        sink_nodes: set[Node] = set()
        for arc in arcs_accessibles:
            if arc.destination.stop_id == gare_arrivee_id:
                sink_nodes.add(arc.destination)
            # Un arc de correspondance à la gare arrivée est aussi un nœud sink
            if arc.source.stop_id == gare_arrivee_id:
                sink_nodes.add(arc.source)
        logger.info(
            "[OrienteeringOptimizer] Sink nodes pour %s : %d nœuds identifiés.",
            gare_arrivee_id, len(sink_nodes),
        )
        return sink_nodes

    # ── Étape 2 : construction du modèle PuLP ─────────────────────────────────

    def _build_model(self, arcs_accessibles: list[Arc]):
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
        idx_sortants: dict[Node, list[int]] = {}
        idx_entrants: dict[Node, list[int]] = {}
        for i, arc in enumerate(arcs_accessibles):
            idx_sortants.setdefault(arc.source,      []).append(i)
            idx_entrants.setdefault(arc.destination, []).append(i)
        return idx_sortants, idx_entrants

    # ── Étape 4a : objectif ───────────────────────────────────────────────────

    def _add_objective(self, model, x, arc_ids, arcs_accessibles, excluded_trip_ids=None):
        """
        Objectif : maximiser le score de fraude.
        Anti-doublons : les trip_ids déjà utilisés par d'autres agents
        reçoivent un malus de 50% pour favoriser la diversité des tournées.
        """
        import pulp
        excluded = excluded_trip_ids or set()
        model += pulp.lpSum(
            arcs_accessibles[i].fraud_score
            * (0.5 if arcs_accessibles[i].trip_id in excluded else 1.0)
            * x[i]
            for i in arc_ids
            if arcs_accessibles[i].arc_type == ArcType.TRAIN
        ), "objectif_score_fraude"

    # ── Étape 4b : contrainte budget ──────────────────────────────────────────

    def _add_budget_constraint(self, model, x, arc_ids, arcs_accessibles, duree_max_minutes):
        import pulp
        model += (
            pulp.lpSum(arcs_accessibles[i].duration_min * x[i] for i in arc_ids)
            <= duree_max_minutes
        ), "budget_temps"

    def _add_min_duration_constraint(self, model, x, arc_ids, arcs_accessibles, duree_max_minutes):
        import pulp
        duree_min = int(duree_max_minutes * 0.80)
        model += (
                pulp.lpSum(arcs_accessibles[i].duration_min * x[i] for i in arc_ids)
                >= duree_min
        ), "duree_minimale"

    # ── Étape 4c : contraintes de flux ────────────────────────────────────────

    def _add_flow_constraints(
        self,
        model,
        x,
        idx_sortants: dict[Node, list[int]],
        idx_entrants: dict[Node, list[int]],
        noeud_depart: Node,
        sink_nodes: set[Node],
    ):
        """
        C2a — Conservation du flux à chaque nœud intermédiaire.

        Nœuds exclus de la conservation standard :
            - noeud_depart : géré par la contrainte source C3
            - sink_nodes   : gérés par la contrainte sink C2b (si définis)

        Si un nœud est à la fois dans sink_nodes et est une escale intermédiaire,
        la contrainte sink agrégée (C2b) force quand même la terminaison.
        """
        import pulp
        # En mode ouvert (pas de sink), on utilise >= au lieu de == :
        # le flux peut se "terminer" à n'importe quel nœud (le chemin s'arrête là).
        # En mode fermé (avec sink), == est correct pour les nœuds intermédiaires
        # car le sink agrégé gère la terminaison via _add_sink_constraint.
        # Note : dans un graphe temps-étendu (DAG), >= n'autorise pas les cycles
        # car le temps ne peut que progresser — la faisabilité est donc garantie.
        open_end = not sink_nodes
        tous_noeuds = set(idx_sortants.keys()) | set(idx_entrants.keys())
        for node in tous_noeuds:
            if node == noeud_depart:
                continue
            if node in sink_nodes:
                continue  # géré par _add_sink_constraint
            entrants = pulp.lpSum(x[i] for i in idx_entrants.get(node, []))
            sortants = pulp.lpSum(x[i] for i in idx_sortants.get(node, []))
            if open_end:
                # Cas ouvert : le flux peut s'arrêter ici (in >= out)
                model += (entrants >= sortants), f"flux_{hash(node) % 10**9}"
            else:
                # Cas fermé : flux doit traverser tous les nœuds intermédiaires (in == out)
                model += (entrants == sortants), f"flux_{hash(node) % 10**9}"

    # ── Étape 4d : contrainte source ──────────────────────────────────────────

    def _add_source_constraint(
        self,
        model,
        x,
        idx_sortants: dict[Node, list[int]],
        noeud_depart: Node,
    ):
        """C3 — Source unique : exactement un arc sortant du nœud de départ."""
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

    # ── Étape 4e : contrainte sink (retour gare) ──────────────────────────────

    def _add_sink_constraint(
        self,
        model,
        x,
        idx_sortants: dict[Node, list[int]],
        idx_entrants: dict[Node, list[int]],
        sink_nodes: set[Node],
    ):
        """
        C2b — Contrainte de retour à la gare d'arrivée.

        Formulation : le flux net agrégé aux nœuds sink doit être +1.
            Σ x[entrants vers n ∈ Sink] − Σ x[sortants de n ∈ Sink] == 1

        Signification : exactement 1 unité de flux se "consomme" à la gare
        d'arrivée, i.e., le chemin se termine obligatoirement dans cette gare.

        La contrainte est sur l'AGRÉGAT de tous les nœuds sink (pas un nœud
        spécifique) car dans le graphe temps-étendu, la gare d'arrivée peut
        être atteinte à plusieurs heures différentes.
        """
        import pulp
        entrants_sink = [
            i
            for n in sink_nodes
            for i in idx_entrants.get(n, [])
        ]
        sortants_sink = [
            i
            for n in sink_nodes
            for i in idx_sortants.get(n, [])
        ]
        if not entrants_sink:
            raise ValueError(
                "[OrienteeringOptimizer] Aucun arc entrant dans les nœuds sink — "
                "la contrainte de retour est physiquement impossible."
            )
        model += (
            pulp.lpSum(x[i] for i in entrants_sink)
            - pulp.lpSum(x[i] for i in sortants_sink)
            == 1
        ), "retour_gare_arrivee"

    def _add_min_trains_constraint(self, model, x, arc_ids, arcs_accessibles):
        """Force au moins 1 arc TRAIN actif — évite la solution triviale (correspondance seule)."""
        import pulp
        train_ids = [i for i in arc_ids if arcs_accessibles[i].arc_type == ArcType.TRAIN]
        if train_ids:
            model += (pulp.lpSum(x[i] for i in train_ids) >= 1), "min_un_train"

    # ── Étape 5 : résolution CBC ──────────────────────────────────────────────

    def _run_cbc(self, model, pulp) -> int:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit_seconds)
        status = model.solve(solver)
        logger.info(
            "[OrienteeringOptimizer] CBC status : %s | score : %.4f",
            pulp.LpStatus[status],
            pulp.value(model.objective) or 0.0,
        )
        return status

    def _is_feasible(self, status: int, pulp) -> bool:
        return status in (
            pulp.constants.LpSolutionOptimal,
            pulp.constants.LpSolutionIntegerFeasible,
        )

    # ── Étape 6 : extraction et reconstruction ────────────────────────────────

    def _extract_active_arcs(self, x, arc_ids, arcs_accessibles, pulp) -> list[Arc]:
        return [
            arcs_accessibles[i]
            for i in arc_ids
            if pulp.value(x[i]) is not None and pulp.value(x[i]) > 0.5
        ]

    def _build_result(self, arcs_actifs: list[Arc], noeud_depart: Node) -> dict:
        arcs_ordonnes = _reconstruct_path(arcs_actifs, noeud_depart)
        score_total   = sum(a.fraud_score for a in arcs_ordonnes if a.arc_type == ArcType.TRAIN)
        duree_totale  = sum(a.duration_min for a in arcs_ordonnes)
        nb_trains     = sum(1 for a in arcs_ordonnes if a.arc_type == ArcType.TRAIN)

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

        IMPORTANT — pourquoi le filtre MIN_FRAUD_SCORE_THRESHOLD est absent ici :

        Le BFS doit garantir la connectivité COMPLETE du graphe, y compris
        les trains a faible score qui peuvent etre les seuls chemins disponibles
        pour revenir a la gare d'arrivee (contrainte aller_retour).

        Exemple : si le seul train de retour a score=0.04, le filtrer ici coupe
        definitivement ce chemin — sink_nodes reste vide et le solver leve
        ValueError au lieu de trouver la solution.

        Le MILP gere naturellement la selection par score :
            - arc score=0.04  → contribue 0.04 a l'objectif, choisi si necessaire
            - arc score=0.80  → contribue 0.80, prioritaire

        Seul filtre physique maintenu : MIN_BOARD_DURATION_MINUTES.
        Un troncon < 6 min est physiquement non controlable, independamment du score.
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
                # Seul filtre physique : duree minimum pour qu'un controle soit possible
                if arc.arc_type == ArcType.TRAIN and arc.duration_min < MIN_BOARD_DURATION_MINUTES:
                    continue
                # Pas de filtre score ici — le MILP gere la selection par valeur
                arcs_result.append(arc)
                if arc.destination not in visites:
                    a_explorer.append(arc.destination)

        return arcs_result

    def _is_arc_eligible(self, arc: Arc) -> bool:
        """
           MÉTHODE NON UTILISÉE DANS LE BFS (voir _filter_reachable_arcs).
           Conservée uniquement pour usage manuel de debug/analyse offline.
           Ne pas appeler depuis le pipeline de production.
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
        gare_arrivee_id: Optional[str] = None,
        excluded_trip_ids: Optional[set] = None,
    ) -> dict:
        """
        Greedy de secours — activé si PuLP absent ou CBC infaisable (sans contrainte retour).

        Note : le greedy ne gère pas la contrainte gare_arrivee_id. Si cette contrainte
        est requise et le MILP a échoué, on lève ValueError plutôt que de retourner
        une tournée incorrecte.
        """
        if gare_arrivee_id is not None:
            raise ValueError(
                f"[OrienteeringOptimizer] Impossible de garantir le retour à "
                f"{gare_arrivee_id} : MILP infaisable et le greedy ne gère pas "
                f"les contraintes de retour. Modifiez la fenêtre PS/FS."
            )

        noeud_depart = self._find_depart_node(graph, gare_depart_id, heure_depart_min)
        if noeud_depart is None:
            raise ValueError(
                f"[OrienteeringOptimizer] Aucun train depuis {gare_depart_id} "
                f"après {heure_depart_min // 60:02d}h{heure_depart_min % 60:02d}."
            )

        arcs_tournee, score_total, temps_ecoule = _greedy_walk(
            graph, noeud_depart, duree_max_minutes, excluded_trip_ids=excluded_trip_ids
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
    excluded_trip_ids: Optional[set] = None,
) -> tuple[list[Arc], float, int]:
    """
    Parcours greedy depuis noeud_depart.
    À chaque étape : prend le meilleur arc TRAIN disponible dans le budget,
    sinon la correspondance la plus courte, sinon s'arrête.
    Les trip_ids dans excluded_trip_ids reçoivent un malus de score (anti-doublons).
    """
    excluded = excluded_trip_ids or set()
    noeud_courant = noeud_depart
    temps_ecoule  = 0
    arcs_tournee: list[Arc] = []
    score_total   = 0.0
    visites: set[Node] = set()

    while temps_ecoule < duree_max_minutes:
        arcs_dispo = graph.get(noeud_courant, [])
        if not arcs_dispo:
            break

        meilleur_train = _best_train_arc(arcs_dispo, temps_ecoule, duree_max_minutes, excluded)
        if meilleur_train:
            arcs_tournee.append(meilleur_train)
            score_total   += meilleur_train.fraud_score
            temps_ecoule  += meilleur_train.duration_min
            noeud_courant  = meilleur_train.destination
            visites.add(noeud_courant)
            continue

        corr = _shortest_correspondance(arcs_dispo, temps_ecoule, duree_max_minutes)
        if corr:
            # Éviter les cycles de correspondance
            if corr.destination in visites:
                break
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
    excluded: set | None = None,
) -> Optional[Arc]:
    """Sélectionne le meilleur arc TRAIN dans le budget, avec malus anti-doublons."""
    excl = excluded or set()
    candidats = [
        a for a in arcs
        if a.arc_type == ArcType.TRAIN
        and temps_ecoule + a.duration_min <= duree_max
    ]
    if not candidats:
        return None
    # Score effectif avec pénalité pour les trips déjà utilisés par d'autres agents
    return max(
        candidats,
        key=lambda a: a.fraud_score * (0.5 if a.trip_id in excl else 1.0),
    )


def _shortest_correspondance(arcs: list[Arc], temps_ecoule: int, duree_max: int) -> Optional[Arc]:
    candidats = [
        a for a in arcs
        if a.arc_type == ArcType.CORRESPONDANCE
        and temps_ecoule + a.duration_min <= duree_max
    ]
    return min(candidats, key=lambda a: a.duration_min) if candidats else None