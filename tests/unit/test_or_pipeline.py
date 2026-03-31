"""
tests/integration/test_or_pipeline.py — Pipeline OR Engine : inject_scores → tournée.

Tests vérifiés :
    1. inject_scores mute correctement les Arc.fraud_score des arcs TRAIN
    2. inject_scores ignore les arcs CORRESPONDANCE
    3. inject_scores est idempotent sur liste vide
    4. Tournée générée après injection est faisable (contraintes temporelles OK)
    5. Score total de la tournée cohérent avec les scores injectés
    6. fetch_scores_from_api avec mock httpx (pas d'appel réseau réel)

Isolation réseau :
    fetch_scores_from_api est testé avec unittest.mock.patch("httpx.post")
    → Aucun appel réseau en CI/CD, tests rapides et déterministes.

Conventions :
    SERVICE_DATE = date(2024, 9, 2) — lundi de référence pour tous les tests.
    Les fixtures créent des graphes minimaux mais représentatifs.
"""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from services.or_engine.graph.builder import TimeExpandedGraphBuilder
from services.or_engine.graph.transition import Arc, ArcType, Node
from services.or_engine.solver import (
    _greedy_optimize,
    fetch_scores_from_api,
    inject_scores,
)
from shared.schemas import PredictScoreItem


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_DATE = date(2024, 9, 2)  # Lundi, jour de référence


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def node_strasbourg():
    return Node("87212027", "Strasbourg", 8 * 60 + 23, SERVICE_DATE)


@pytest.fixture
def node_selestat_arrivee():
    """Sélestat comme nœud d'arrivée (08:52)."""
    return Node("87214007", "Sélestat", 8 * 60 + 52, SERVICE_DATE)


@pytest.fixture
def node_selestat_depart():
    """Sélestat comme nœud de départ après correspondance (09:05)."""
    return Node("87214007", "Sélestat", 9 * 60 + 5, SERVICE_DATE)


@pytest.fixture
def node_colmar():
    return Node("87214080", "Colmar", 9 * 60 + 15, SERVICE_DATE)


@pytest.fixture
def node_mulhouse():
    return Node("87182063", "Mulhouse", 9 * 60 + 58, SERVICE_DATE)


@pytest.fixture
def arc_train_stras_sel(node_strasbourg, node_selestat_arrivee):
    """Arc TRAIN Strasbourg 08:23 → Sélestat 08:52 (TRIP_001)."""
    return Arc(
        source=node_strasbourg,
        destination=node_selestat_arrivee,
        arc_type=ArcType.TRAIN,
        duration_min=29,
        trip_id="TRIP_001",
        train_number="117756",
        fraud_score=0.0,  # initialisé à 0.0 — sera mis à jour par inject_scores
    )


@pytest.fixture
def arc_train_sel_col(node_selestat_arrivee, node_colmar):
    """Arc TRAIN Sélestat 08:52 → Colmar 09:15 (TRIP_001)."""
    return Arc(
        source=node_selestat_arrivee,
        destination=node_colmar,
        arc_type=ArcType.TRAIN,
        duration_min=23,
        trip_id="TRIP_001",
        train_number="117756",
        fraud_score=0.0,
    )


@pytest.fixture
def arc_corr_sel(node_selestat_arrivee, node_selestat_depart):
    """Arc CORRESPONDANCE Sélestat 08:52 → Sélestat 09:05 (attente en gare)."""
    return Arc(
        source=node_selestat_arrivee,
        destination=node_selestat_depart,
        arc_type=ArcType.CORRESPONDANCE,
        duration_min=13,
    )


@pytest.fixture
def arc_train_sel_mul(node_selestat_depart, node_mulhouse):
    """Arc TRAIN Sélestat 09:05 → Mulhouse 09:58 (TRIP_002)."""
    return Arc(
        source=node_selestat_depart,
        destination=node_mulhouse,
        arc_type=ArcType.TRAIN,
        duration_min=53,
        trip_id="TRIP_002",
        train_number="117758",
        fraud_score=0.0,
    )


@pytest.fixture
def graph_complet(
    node_strasbourg, node_selestat_arrivee, node_selestat_depart, node_colmar, node_mulhouse,
    arc_train_stras_sel, arc_train_sel_col, arc_corr_sel, arc_train_sel_mul,
):
    """
    Graphe complet pour les tests d'intégration :

        Strasbourg 08:23 ──TRAIN(117756)──→ Sélestat 08:52
        Sélestat   08:52 ──TRAIN(117756)──→ Colmar   09:15
        Sélestat   08:52 ──CORR──────────→ Sélestat 09:05
        Sélestat   09:05 ──TRAIN(117758)──→ Mulhouse 09:58

    Tous les fraud_score initialisés à 0.0.
    """
    return {
        node_strasbourg:       [arc_train_stras_sel],
        node_selestat_arrivee: [arc_train_sel_col, arc_corr_sel],
        node_selestat_depart:  [arc_train_sel_mul],
        node_colmar:           [],
        node_mulhouse:         [],
    }


@pytest.fixture
def scores_ml():
    """Scores ML simulant une réponse de l'API predict/batch."""
    return [
        PredictScoreItem(
            trip_id="TRIP_001",
            stop_sequence=0,
            stop_id_dep="87212027",
            dep_minutes=8 * 60 + 23,
            fraud_score=0.85,
        ),
        PredictScoreItem(
            trip_id="TRIP_001",
            stop_sequence=1,
            stop_id_dep="87214007",
            dep_minutes=8 * 60 + 52,
            fraud_score=0.42,
        ),
        PredictScoreItem(
            trip_id="TRIP_002",
            stop_sequence=0,
            stop_id_dep="87214007",
            dep_minutes=9 * 60 + 5,
            fraud_score=0.71,
        ),
    ]


@pytest.fixture
def troncons_df_sample():
    """DataFrame de tronçons cohérent avec le graphe_complet."""
    return pd.DataFrame([
        {
            "trip_id": "TRIP_001", "train_number": "117756",
            "service_id": "000001", "stop_sequence": 0,
            "stop_id_dep": "87212027", "stop_name_dep": "Strasbourg",
            "dep_minutes": 8 * 60 + 23,
            "stop_id_arr": "87214007", "stop_name_arr": "Sélestat",
            "arr_minutes": 8 * 60 + 52, "duration_min": 29,
        },
        {
            "trip_id": "TRIP_001", "train_number": "117756",
            "service_id": "000001", "stop_sequence": 1,
            "stop_id_dep": "87214007", "stop_name_dep": "Sélestat",
            "dep_minutes": 8 * 60 + 52,
            "stop_id_arr": "87214080", "stop_name_arr": "Colmar",
            "arr_minutes": 9 * 60 + 15, "duration_min": 23,
        },
        {
            "trip_id": "TRIP_002", "train_number": "117758",
            "service_id": "000001", "stop_sequence": 0,
            "stop_id_dep": "87214007", "stop_name_dep": "Sélestat",
            "dep_minutes": 9 * 60 + 5,
            "stop_id_arr": "87182063", "stop_name_arr": "Mulhouse",
            "arr_minutes": 9 * 60 + 58, "duration_min": 53,
        },
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tests inject_scores
# ─────────────────────────────────────────────────────────────────────────────

class TestInjectScores:
    """Tests unitaires de la fonction inject_scores()."""

    def test_inject_scores_mute_arcs_train(
        self, graph_complet, scores_ml, node_strasbourg
    ):
        """
        inject_scores doit mettre à jour le fraud_score des arcs TRAIN
        avec les valeurs retournées par l'API ML.
        """
        inject_scores(graph_complet, scores_ml)

        arc_stras_sel = graph_complet[node_strasbourg][0]
        assert arc_stras_sel.arc_type == ArcType.TRAIN
        assert arc_stras_sel.fraud_score == pytest.approx(0.85), \
            "Arc Strasbourg→Sélestat doit avoir fraud_score=0.85"

    def test_inject_scores_tous_les_arcs_train(
        self, graph_complet, scores_ml, node_strasbourg,
        node_selestat_arrivee, node_selestat_depart
    ):
        """
        Tous les arcs TRAIN matchés doivent être mis à jour,
        pas seulement le premier.
        """
        inject_scores(graph_complet, scores_ml)

        # Arc Sélestat 08:52 → Colmar
        arc_sel_col = next(
            a for a in graph_complet[node_selestat_arrivee]
            if a.arc_type == ArcType.TRAIN
        )
        assert arc_sel_col.fraud_score == pytest.approx(0.42)

        # Arc Sélestat 09:05 → Mulhouse
        arc_sel_mul = next(
            a for a in graph_complet[node_selestat_depart]
            if a.arc_type == ArcType.TRAIN
        )
        assert arc_sel_mul.fraud_score == pytest.approx(0.71)

    def test_inject_scores_ignore_correspondances(
        self, graph_complet, scores_ml, node_selestat_arrivee
    ):
        """
        Les arcs CORRESPONDANCE ne doivent PAS être modifiés par inject_scores.
        """
        arc_corr = next(
            a for a in graph_complet[node_selestat_arrivee]
            if a.arc_type == ArcType.CORRESPONDANCE
        )
        fraud_avant = arc_corr.fraud_score  # 0.0 initial

        inject_scores(graph_complet, scores_ml)

        assert arc_corr.fraud_score == fraud_avant, \
            "Les arcs CORRESPONDANCE ne doivent pas être modifiés"

    def test_inject_scores_retourne_none(self, graph_complet, scores_ml):
        """inject_scores est une procédure — retourne None."""
        result = inject_scores(graph_complet, scores_ml)
        assert result is None

    def test_inject_scores_mutation_inplace(
        self, graph_complet, scores_ml, node_strasbourg
    ):
        """
        inject_scores mute les ARC originaux du graphe, pas des copies.
        L'objet Arc en mémoire doit être le même avant et après.
        """
        arc_avant = graph_complet[node_strasbourg][0]
        inject_scores(graph_complet, scores_ml)
        arc_apres = graph_complet[node_strasbourg][0]

        # Même objet en mémoire (pas de copie)
        assert arc_avant is arc_apres
        # Mais le fraud_score a changé
        assert arc_apres.fraud_score == pytest.approx(0.85)

    def test_inject_scores_liste_vide_ne_modifie_pas_graphe(self, graph_complet):
        """inject_scores avec liste vide ne doit pas modifier le graphe."""
        inject_scores(graph_complet, [])

        for node, arcs in graph_complet.items():
            for arc in arcs:
                if arc.arc_type == ArcType.TRAIN:
                    assert arc.fraud_score == 0.0, \
                        "fraud_score doit rester 0.0 si aucun score injecté"

    def test_inject_scores_arc_inconnu_ignoré(self, graph_complet):
        """
        Un score sans arc correspondant dans le graphe ne doit pas
        lever d'exception — il est simplement ignoré (log warning).
        """
        scores_inconnus = [
            PredictScoreItem(
                trip_id="TRIP_INEXISTANT",
                stop_sequence=0,
                stop_id_dep="99999999",
                dep_minutes=700,
                fraud_score=0.99,
            )
        ]

        # Ne doit PAS lever d'exception
        inject_scores(graph_complet, scores_inconnus)

        # Le graphe ne doit pas être modifié
        for node, arcs in graph_complet.items():
            for arc in arcs:
                if arc.arc_type == ArcType.TRAIN:
                    assert arc.fraud_score == 0.0

    def test_inject_scores_plusieurs_appels_idempotent(
        self, graph_complet, scores_ml, node_strasbourg
    ):
        """
        Appeler inject_scores deux fois avec les mêmes scores
        doit produire le même résultat (idempotence).
        """
        inject_scores(graph_complet, scores_ml)
        score_apres_1 = graph_complet[node_strasbourg][0].fraud_score

        inject_scores(graph_complet, scores_ml)
        score_apres_2 = graph_complet[node_strasbourg][0].fraud_score

        assert score_apres_1 == score_apres_2


# ─────────────────────────────────────────────────────────────────────────────
# Tests d'intégration : GraphBuilder → inject_scores → tournée faisable
# ─────────────────────────────────────────────────────────────────────────────

class TestTourneeAvecScoresInjectes:
    """
    Tests de la séquence complète : GraphBuilder → inject_scores → tournée.
    Vérifie que les scores ML influencent correctement l'optimisation.
    """

    def test_tournee_faisable_apres_injection(self, troncons_df_sample, scores_ml):
        """
        Après injection des scores, le greedy solver doit produire
        une tournée qui respecte les contraintes temporelles.
        """
        builder = TimeExpandedGraphBuilder()
        graph   = builder.build(troncons_df_sample, SERVICE_DATE)

        inject_scores(graph, scores_ml)

        tournee = _greedy_optimize(
            graph=graph,
            builder=builder,
            gare_depart_id="87212027",
            heure_depart_min=8 * 60,
            duree_max_minutes=360,
        )

        # Faisabilité : au moins 1 train
        assert tournee["nb_trains"] >= 1, \
            "La tournée doit contrôler au moins 1 train"

        # Faisabilité temporelle : durée dans le budget
        assert tournee["duree_minutes"] <= 360, \
            "La durée de la tournée doit respecter le budget de 360 min"

        # Cohérence : score total ≥ 0
        assert tournee["score_total"] >= 0.0

    def test_score_total_coherent_avec_arcs_train(self, troncons_df_sample, scores_ml):
        """
        Le score_total de la tournée doit être égal à la somme des fraud_score
        des arcs TRAIN effectivement empruntés (pas des correspondances).
        """
        builder = TimeExpandedGraphBuilder()
        graph   = builder.build(troncons_df_sample, SERVICE_DATE)
        inject_scores(graph, scores_ml)

        tournee = _greedy_optimize(
            graph=graph,
            builder=builder,
            gare_depart_id="87212027",
            heure_depart_min=8 * 60,
            duree_max_minutes=360,
        )

        # Recalcul manuel depuis les arcs
        score_manuel = sum(
            a.fraud_score
            for a in tournee["arcs"]
            if a.arc_type == ArcType.TRAIN
        )

        assert tournee["score_total"] == pytest.approx(score_manuel, abs=1e-4), \
            "score_total doit être la somme des fraud_score des arcs TRAIN"

    def test_greedy_prefere_meilleur_score(self, troncons_df_sample, scores_ml):
        """
        Le greedy solver doit choisir l'arc TRAIN avec le meilleur fraud_score
        parmi ceux disponibles depuis la position courante.

        Depuis Strasbourg, un seul arc TRAIN existe (→ Sélestat, score=0.85).
        La tournée doit commencer par cet arc.
        """
        builder = TimeExpandedGraphBuilder()
        graph   = builder.build(troncons_df_sample, SERVICE_DATE)
        inject_scores(graph, scores_ml)

        tournee = _greedy_optimize(
            graph=graph,
            builder=builder,
            gare_depart_id="87212027",
            heure_depart_min=8 * 60,
            duree_max_minutes=360,
        )

        # Premier arc = Strasbourg → Sélestat (seul arc TRAIN disponible au départ)
        premier_arc = tournee["arcs"][0]
        assert premier_arc.arc_type == ArcType.TRAIN
        assert premier_arc.source.stop_id == "87212027"
        assert premier_arc.fraud_score == pytest.approx(0.85)

    def test_tournee_budget_serre_limite_nb_trains(self, troncons_df_sample, scores_ml):
        """
        Avec un budget très serré (40 min), seul 1 train court peut entrer.
        """
        builder = TimeExpandedGraphBuilder()
        graph   = builder.build(troncons_df_sample, SERVICE_DATE)
        inject_scores(graph, scores_ml)

        tournee = _greedy_optimize(
            graph=graph,
            builder=builder,
            gare_depart_id="87212027",
            heure_depart_min=8 * 60,
            duree_max_minutes=40,  # seulement 40 min
        )

        assert tournee["duree_minutes"] <= 40, \
            "Budget de 40 min ne doit pas être dépassé"

    def test_gare_inexistante_leve_value_error(self, troncons_df_sample):
        """
        Demander une tournée depuis une gare absente du graphe
        doit lever ValueError (pas un crash silencieux).
        """
        builder = TimeExpandedGraphBuilder()
        graph   = builder.build(troncons_df_sample, SERVICE_DATE)

        with pytest.raises(ValueError, match="Aucun train"):
            _greedy_optimize(
                graph=graph,
                builder=builder,
                gare_depart_id="99999999",  # gare inexistante
                heure_depart_min=8 * 60,
                duree_max_minutes=360,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests fetch_scores_from_api (mock httpx)
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchScoresFromApi:
    """
    Tests de fetch_scores_from_api avec mock httpx.
    Aucun appel réseau réel — httpx.post est mocké à chaque test.
    """

    @pytest.fixture
    def api_response_valide(self):
        """Corps de réponse JSON valide de POST /predict/batch."""
        return {
            "scores": [
                {
                    "trip_id": "TRIP_001",
                    "stop_sequence": 0,
                    "stop_id_dep": "87212027",
                    "dep_minutes": 503,
                    "fraud_score": 0.85,
                }
            ],
            "scored_at": "2024-09-02T08:00:00",
            "nb_troncons": 1,
        }

    def test_retourne_liste_de_predict_score_items(
        self, troncons_df_sample, api_response_valide
    ):
        """
        Quand l'API répond 200, fetch_scores_from_api doit retourner
        une liste de PredictScoreItem parsée depuis le JSON.
        """
        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = api_response_valide
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            scores = fetch_scores_from_api(
                troncons_df_sample, SERVICE_DATE,
                predict_url="http://test-api/predict/batch",
            )

        assert len(scores) == 1
        assert isinstance(scores[0], PredictScoreItem)
        assert scores[0].fraud_score == pytest.approx(0.85)
        assert scores[0].trip_id == "TRIP_001"

    def test_retourne_liste_vide_si_timeout(self, troncons_df_sample):
        """
        En cas de timeout httpx, retourner [] sans lever d'exception.
        Le solver doit continuer avec fraud_score=0.0.
        """
        import httpx

        with patch("httpx.post", side_effect=httpx.TimeoutException("timeout")):
            scores = fetch_scores_from_api(
                troncons_df_sample, SERVICE_DATE,
                predict_url="http://test-api/predict/batch",
            )

        assert scores == [], \
            "Timeout doit retourner une liste vide (pas d'exception)"

    def test_retourne_liste_vide_si_connexion_refusee(self, troncons_df_sample):
        """En cas de connexion refusée, retourner [] sans lever d'exception."""
        import httpx

        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            scores = fetch_scores_from_api(
                troncons_df_sample, SERVICE_DATE,
                predict_url="http://api-indisponible/predict/batch",
            )

        assert scores == []

    def test_retourne_liste_vide_si_erreur_http(self, troncons_df_sample):
        """En cas d'erreur HTTP 500, retourner [] sans lever d'exception."""
        import httpx

        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Internal Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500, text="Internal Server Error"),
            )
            mock_post.return_value = mock_resp

            scores = fetch_scores_from_api(
                troncons_df_sample, SERVICE_DATE,
                predict_url="http://test-api/predict/batch",
            )

        assert scores == []

    def test_appel_avec_bonne_url(self, troncons_df_sample, api_response_valide):
        """
        fetch_scores_from_api doit appeler httpx.post avec l'URL configurée.
        """
        url_test = "http://test-predict/predict/batch"

        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = api_response_valide
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            fetch_scores_from_api(troncons_df_sample, SERVICE_DATE, predict_url=url_test)

        call_args = mock_post.call_args
        assert call_args[0][0] == url_test, \
            "httpx.post doit être appelé avec l'URL configurée"

    def test_service_date_propagee_dans_payload(self, troncons_df_sample, api_response_valide):
        """
        La service_date doit être présente dans chaque TronconInput du payload JSON.
        C'est la vérification de la propagation de date depuis le contexte GTFS.
        """
        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = api_response_valide
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            fetch_scores_from_api(troncons_df_sample, SERVICE_DATE)

        # Vérification du payload envoyé
        call_kwargs = mock_post.call_args[1]
        payload     = call_kwargs["json"]
        troncons    = payload["troncons"]

        assert len(troncons) > 0
        for troncon in troncons:
            assert "service_date" in troncon, \
                "Chaque TronconInput doit contenir service_date"
            assert troncon["service_date"] == "2024-09-02", \
                f"service_date attendu 2024-09-02, reçu {troncon['service_date']}"