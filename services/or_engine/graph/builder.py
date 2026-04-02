# services/or_engine/graph/builder.py
import bisect
import logging
from collections import defaultdict
from datetime import date

import pandas as pd

from services.or_engine.graph.transition import Arc, ArcType, Node
from shared.constants import (
    MAX_TRANSFER_MINUTES,
    MIN_BOARD_DURATION_MINUTES,
    MIN_TRANSFER_MINUTES,
)

logger = logging.getLogger(__name__)

# Types internes
_StopIndex = dict[str, list[tuple[int, str]]]  # stop_id → [(minutes, stop_name)]


class TimeExpandedGraphBuilder:
    """
    Construit le graphe temps-étendu à partir des tronçons GTFS.

    Responsabilité unique : construire la structure du graphe.
    Ne sait pas comment les scores sont calculés (rôle du modèle ML).
    Ne sait pas comment le graphe est optimisé (rôle de l'optimiseur).

    Algorithme O(n log n) pour les correspondances :
        Tri des départs par gare + bisect pour trouver la fenêtre valide
        [arr + MIN_TRANSFER, arr + MAX_TRANSFER] en O(log k) par arrivée.
    """

    def __init__(self):
        self.min_board_duration = MIN_BOARD_DURATION_MINUTES
        self.min_transfer       = MIN_TRANSFER_MINUTES
        self.max_transfer       = MAX_TRANSFER_MINUTES

    # ── Interface publique ────────────────────────────────────────────────────

    def build(
        self,
        troncons: pd.DataFrame,
        service_date: date,
    ) -> dict[Node, list[Arc]]:
        """
        Construit le graphe temps-étendu pour une date de service donnée.

        Paramètres :
            troncons     : DataFrame produit par GTFSPreprocessor.build_troncons()
            service_date : date de circulation (ex: date(2024, 9, 2))

        Retourne :
            dict[Node, list[Arc]] — liste d'adjacence du graphe
        """
        self._validate(troncons)

        graph: dict[Node, list[Arc]] = defaultdict(list)
        graph = self._add_train_arcs(graph, troncons, service_date)
        graph = self._add_correspondance_arcs(graph, troncons, service_date)

        self._log_graph_stats(graph, service_date)
        return dict(graph)

    def get_nodes_at_stop(
        self,
        graph: dict[Node, list[Arc]],
        stop_id: str,
    ) -> list[Node]:
        """Retourne tous les nœuds d'une gare donnée dans le graphe."""
        return [node for node in graph if node.stop_id == stop_id]

    def get_reachable_arcs(
        self,
        graph: dict[Node, list[Arc]],
        node: Node,
    ) -> list[Arc]:
        """Retourne les arcs accessibles depuis un nœud. Liste vide si absent."""
        return graph.get(node, [])

    # ── Construction des arcs TRAIN ───────────────────────────────────────────

    def _add_train_arcs(
        self,
        graph: dict,
        troncons: pd.DataFrame,
        service_date: date,
    ) -> dict:
        """
        Ajoute un arc TRAIN pour chaque tronçon GTFS de durée suffisante.
        Délègue le filtrage, la création de nœuds et la création d'arcs
        à des méthodes dédiées (principe S).
        """
        troncons_valides = self._filter_valid_troncons(troncons)
        for _, row in troncons_valides.iterrows():
            arc = self._make_train_arc(row, service_date)
            graph[arc.source].append(arc)
        return graph

    def _filter_valid_troncons(self, troncons: pd.DataFrame) -> pd.DataFrame:
        """Filtre les tronçons dont la durée est suffisante pour un contrôle."""
        valides  = troncons[troncons["duration_min"] >= self.min_board_duration]
        skipped  = len(troncons) - len(valides)
        if skipped > 0:
            logger.info(
                "[GraphBuilder] %d tronçons ignorés (durée < %d min).",
                skipped, self.min_board_duration,
            )
        return valides

    def _make_train_arc(self, row: pd.Series, service_date: date) -> Arc:
        """Crée un arc TRAIN depuis une ligne du DataFrame tronçons."""
        node_dep = self._make_node(
            row["stop_id_dep"], row["stop_name_dep"],
            int(row["dep_minutes"]), service_date,
        )
        node_arr = self._make_node(
            row["stop_id_arr"], row["stop_name_arr"],
            int(row["arr_minutes"]), service_date,
        )
        return Arc(
            source=node_dep,
            destination=node_arr,
            arc_type=ArcType.TRAIN,
            duration_min=int(row["duration_min"]),
            trip_id=str(row["trip_id"]),
            train_number=str(row["train_number"]),
            fraud_score=0.0,
        )

    # ── Construction des arcs de correspondance ───────────────────────────────

    def _add_correspondance_arcs(
        self,
        graph: dict,
        troncons: pd.DataFrame,
        service_date: date,
    ) -> dict:
        """
        Ajoute les arcs de correspondance entre trains dans la même gare.
        Délègue la construction des index et la création d'arcs (principe S).
        """
        arrivees = self._build_arrivees_index(troncons)
        departs  = self._build_departs_index(troncons)

        n_correspondances = 0
        for stop_id in set(arrivees.keys()) & set(departs.keys()):
            n_correspondances += self._add_correspondances_for_stop(
                graph, stop_id, arrivees[stop_id], departs[stop_id], service_date
            )

        logger.info("[GraphBuilder] %d arcs de correspondance ajoutés.", n_correspondances)
        return graph

    def _build_arrivees_index(self, troncons: pd.DataFrame) -> _StopIndex:
        """Construit l'index stop_id → [(arr_minutes, stop_name)] depuis troncons."""
        arrivees: _StopIndex = defaultdict(list)
        for _, row in troncons.iterrows():
            arrivees[row["stop_id_arr"]].append(
                (int(row["arr_minutes"]), row["stop_name_arr"])
            )
        return arrivees

    def _build_departs_index(self, troncons: pd.DataFrame) -> _StopIndex:
        """
        Construit l'index stop_id → [(dep_minutes, stop_name)] trié par minutes.
        Le tri est requis pour que bisect fonctionne correctement.
        """
        departs: _StopIndex = defaultdict(list)
        for _, row in troncons.iterrows():
            departs[row["stop_id_dep"]].append(
                (int(row["dep_minutes"]), row["stop_name_dep"])
            )
        for stop_id in departs:
            departs[stop_id].sort(key=lambda x: x[0])
        return departs

    def _add_correspondances_for_stop(
        self,
        graph: dict,
        stop_id: str,
        arrivees: list[tuple[int, str]],
        departs: list[tuple[int, str]],
        service_date: date,
    ) -> int:
        """
        Ajoute les arcs de correspondance pour une gare donnée.
        Utilise bisect pour ne visiter que les départs dans la fenêtre valide.
        Retourne le nombre d'arcs créés.
        """
        dep_minutes = [d[0] for d in departs]
        n_added     = 0

        for arr_min, stop_name in arrivees:
            lo = bisect.bisect_left(dep_minutes,  arr_min + self.min_transfer)
            hi = bisect.bisect_right(dep_minutes, arr_min + self.max_transfer)

            for dep_min, _ in departs[lo:hi]:
                arc = self._make_correspondance_arc(
                    stop_id, stop_name, arr_min, dep_min, service_date
                )
                graph[arc.source].append(arc)
                n_added += 1

        return n_added

    def _make_correspondance_arc(
        self,
        stop_id: str,
        stop_name: str,
        arr_min: int,
        dep_min: int,
        service_date: date,
    ) -> Arc:
        """Crée un arc CORRESPONDANCE entre deux instants dans la même gare."""
        return Arc(
            source=self._make_node(stop_id, stop_name, arr_min, service_date),
            destination=self._make_node(stop_id, stop_name, dep_min, service_date),
            arc_type=ArcType.CORRESPONDANCE,
            duration_min=dep_min - arr_min,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_node(
        stop_id: str,
        stop_name: str,
        time_minutes: int,
        service_date: date,
    ) -> Node:
        """Factorise la création de Node. Responsabilité unique : éviter la répétition."""
        return Node(
            stop_id=stop_id,
            stop_name=stop_name,
            time_minutes=time_minutes,
            service_date=service_date,
        )

    def _validate(self, troncons: pd.DataFrame) -> None:
        """Vérifie que les colonnes requises sont présentes. Fail-fast."""
        required = {
            "trip_id", "train_number",
            "stop_id_dep", "stop_name_dep", "dep_minutes",
            "stop_id_arr", "stop_name_arr", "arr_minutes",
            "duration_min",
        }
        missing = required - set(troncons.columns)
        if missing:
            raise ValueError(
                f"[GraphBuilder] Colonnes manquantes dans troncons : {missing}"
            )

    @staticmethod
    def _log_graph_stats(graph: dict, service_date: date) -> None:
        """Log les statistiques du graphe construit. Responsabilité unique : reporting."""
        n_nodes = len(graph)
        n_arcs  = sum(len(arcs) for arcs in graph.values())
        logger.info(
            "[GraphBuilder] Graphe construit : %d nœuds, %d arcs (%s).",
            n_nodes, n_arcs, service_date,
        )