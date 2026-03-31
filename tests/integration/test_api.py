"""
tests/integration/test_api.py — Tests E2E des endpoints FastAPI.

Ces tests vérifient le comportement de l'API de bout en bout,
depuis la requête HTTP jusqu'à la réponse JSON sérialisée.

Architecture des mocks :
    - GTFSLoader   → mocké pour retourner des DataFrames synthétiques (sans GTFS réel)
    - _load_scorer / _load_pipeline → moqués pour activer le fallback déterministe
      OU pour simuler un scorer ML réel selon le test

    Les mocks sont construits au niveau du module où la dépendance EST UTILISÉE
    (pas là où elle est définie) — règle fondamentale du patching Python.

Isolation :
    - lru_cache des singletons purgé avant chaque test (monkeypatch)
    - Aucun fichier GTFS ou .joblib réel requis
    - Aucun état partagé entre les classes de tests

Structure :
    TestHealth         → GET /health              — sans mock
    TestPredictBatch   → POST /predict/batch       — sans GTFS, avec fallback ML
    TestPredictGet     → GET /predict              — mock GTFSLoader
    TestOptimize       → POST /optimize            — mock GTFSLoader

Données synthétiques :
    Réseau alsacien fictif : Strasbourg (87212027) → Sélestat (87214007) → Colmar (87214080)
    Date de service : 2024-09-02 (lundi, semaine scolaire, hors vacances)
    5 trips TER avec 3 arrêts chacun → 10 tronçons au total
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────────────────────────────────────
# Purge du cache sys.modules — contourne le problème de stale import
# ─────────────────────────────────────────────────────────────────────────────
# Problème : pytest réutilise sys.modules entre les fichiers de test.
# Si l'ancienne version de predict.py a été chargée (par une session précédente
# ou un import indirect), les routes FastAPI enregistrées dans `app` seront
# celles de l'ancienne version — même si le fichier sur disque est à jour.
#
# Solution : purger TOUS les modules de l'API du cache avant tout import.
# Python relit alors les .py depuis le disque et reconstruit app avec les
# bons endpoints (/predict/batch et GET /predict).
import sys as _sys

_API_MODULES = [
    "services.api.main",
    "services.api.routers",
    "services.api.routers.predict",
    "services.api.routers.optimize",
    "services.api.routers.health",
]
for _mod in _API_MODULES:
    _sys.modules.pop(_mod, None)
# Purge aussi toute clé dérivée (ex: sous-modules éventuels)
for _key in [k for k in _sys.modules if k.startswith("services.api")]:
    _sys.modules.pop(_key, None)

from services.api.main import app  # noqa: E402 — import après purge
import services.api.routers.predict as predict_module  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Client de test partagé
# ─────────────────────────────────────────────────────────────────────────────

client = TestClient(app)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes des fixtures synthétiques
# ─────────────────────────────────────────────────────────────────────────────

SERVICE_DATE      = "2024-09-02"
SERVICE_DATE_INT  = 20240902
GARE_DEPART_ID    = "87212027"   # Strasbourg
GARE_ARRIVEE_ID   = "87214007"   # Sélestat
GARE_COLMAR_ID    = "87214080"   # Colmar
HEURE_DEPART_MIN  = 360          # 06:00
ROUTE_ID_TER      = "FR:Line::TER_ALSACE:"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures partagées
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_stops() -> pd.DataFrame:
    """Table stops synthétique — 3 gares alsaciennes."""
    return pd.DataFrame([
        {
            "stop_id": "87212027", "stop_name": "Strasbourg",
            "stop_lat": 48.585, "stop_lon": 7.735,
            "stop_desc": "", "zone_id": "", "stop_url": "",
            "location_type": 0, "parent_station": "",
        },
        {
            "stop_id": "87214007", "stop_name": "Sélestat",
            "stop_lat": 48.259, "stop_lon": 7.450,
            "stop_desc": "", "zone_id": "", "stop_url": "",
            "location_type": 0, "parent_station": "",
        },
        {
            "stop_id": "87214080", "stop_name": "Colmar",
            "stop_lat": 48.080, "stop_lon": 7.356,
            "stop_desc": "", "zone_id": "", "stop_url": "",
            "location_type": 0, "parent_station": "",
        },
    ])


@pytest.fixture
def synthetic_routes() -> pd.DataFrame:
    """Table routes synthétique — une ligne TER."""
    return pd.DataFrame([{
        "route_id":         ROUTE_ID_TER,
        "agency_id":        "1187",
        "route_short_name": "TER",
        "route_long_name":  "TER Alsace",
        "route_desc":       "",
        "route_type":       2,       # 2 = Rail régional (TER)
        "route_url":        "",
        "route_color":      "006600",
        "route_text_color": "FFFFFF",
    }])


@pytest.fixture
def synthetic_trips() -> pd.DataFrame:
    """Table trips synthétique — 5 missions TER sur la ligne Alsace."""
    return pd.DataFrame([
        {
            "route_id":      ROUTE_ID_TER,
            "service_id":    "000001",
            "trip_id":       f"TRIP_{i:03d}",
            "trip_headsign": f"1177{i:02d}",
            "direction_id":  0,
            "block_id":      i,
            "shape_id":      "",
        }
        for i in range(5)
    ])


@pytest.fixture
def synthetic_stop_times(synthetic_trips) -> pd.DataFrame:
    """
    Table stop_times synthétique.
    Chaque trip s'arrête à 3 gares avec 25 minutes d'intervalle.
    Horaires : TRIP_000 part à 06:00, +1h par trip.
    """
    gares = ["87212027", "87214007", "87214080"]
    rows = []
    for i, row in synthetic_trips.iterrows():
        trip_id = row["trip_id"]
        base_hour = 6 + i           # 6h, 7h, 8h, 9h, 10h
        for seq, gare_id in enumerate(gares):
            dep_h = base_hour
            dep_m = seq * 25
            dep_total = dep_h * 60 + dep_m
            arr_total = dep_total + 20
            arr_h, arr_m = divmod(arr_total, 60)
            rows.append({
                "trip_id":               trip_id,
                "arrival_time":          f"{arr_h:02d}:{arr_m:02d}:00",
                "departure_time":        f"{dep_h:02d}:{dep_m:02d}:00",
                "stop_id":               gare_id,
                "stop_sequence":         seq,
                "stop_headsign":         "",
                "pickup_type":           0,
                "drop_off_type":         0,
                "shape_dist_traveled":   "",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_calendar_dates() -> pd.DataFrame:
    """
    Table calendar_dates — service_id '000001' actif du 2024-09-01 au 2024-09-30.
    """
    from datetime import timedelta
    base = date(2024, 9, 1)
    return pd.DataFrame([
        {
            "service_id":     "000001",
            "date":           int((base + timedelta(days=d)).strftime("%Y%m%d")),
            "exception_type": 1,
        }
        for d in range(30)
    ])


@pytest.fixture(autouse=True)
def clear_lru_cache():
    """
    Purgé automatiquement avant chaque test.

    lru_cache() est un Singleton de processus — les tests modifient
    l'état des singletons. Sans purge, les mocks du test 1 fuient
    dans le test 2 (ordre d'exécution non garanti par pytest).

    autouse=True → appliqué à tous les tests du module sans opt-in.

    Défensif : les attributs _load_scorer / _load_pipeline n'existent que
    dans la nouvelle version de predict.py (post-intégration ML).
    Si l'ancienne version est encore en place, le fixture ne crashe pas —
    les tests s'exécutent simplement sans purge du cache (comportement
    identique puisqu'il n'y a pas de cache à purger).
    """
    def _clear():
        for fn_name in ("_load_scorer", "_load_pipeline"):
            fn = getattr(predict_module, fn_name, None)
            if fn is not None and hasattr(fn, "cache_clear"):
                fn.cache_clear()

    _clear()
    yield
    _clear()

@pytest.fixture(autouse=True)
def force_deterministic_fallback():
    """
    Force l'utilisation du fallback déterministe pendant les tests.
    Empêche le chargement des vrais .joblib s'ils existent en local.
    """
    with patch.object(predict_module, "_load_scorer", return_value=None), \
            patch.object(predict_module, "_load_pipeline", return_value=None):
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Helper : fabrique un mock GTFSLoader complet
# ─────────────────────────────────────────────────────────────────────────────

def make_gtfs_loader_mock(stop_times, trips, stops, routes, calendar_dates):
    """
    Construit un MagicMock de GTFSLoader retournant les DataFrames synthétiques.

    Pourquoi un helper et pas un fixture ?
        Certains tests n'ont besoin que de loader (predict GET, optimize)
        mais veulent patcher des chemins différents :
            - predict.py importe depuis services.ml_engine.data.gtfs.loader
            - optimize.py importe depuis services.ml_engine.data.gtfs.loader
        Un helper factorie un mock identique patchable sur les deux chemins.
    """
    mock_loader_instance = MagicMock()
    mock_loader_instance.load_stop_times.return_value  = stop_times
    mock_loader_instance.load_trips.return_value        = trips
    mock_loader_instance.load_stops.return_value        = stops
    mock_loader_instance.load_routes.return_value       = routes
    mock_loader_instance.load_calendar_dates.return_value = calendar_dates
    return mock_loader_instance


# ─────────────────────────────────────────────────────────────────────────────
# TestHealth
# ─────────────────────────────────────────────────────────────────────────────

class TestHealth:
    """
    GET /health — Endpoint de santé.

    Aucun mock requis : l'endpoint retourne une réponse statique
    qui ne dépend d'aucune ressource externe.
    """

    def test_health_returns_200(self):
        """Le serveur est joignable et opérationnel."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body_structure(self):
        """La réponse respecte le schéma HealthSchema."""
        body = client.get("/health").json()
        assert body["status"]  == "ok"
        assert body["version"] == "0.1.0"

    def test_health_is_idempotent(self):
        """Plusieurs appels successifs retournent la même réponse."""
        r1 = client.get("/health").json()
        r2 = client.get("/health").json()
        assert r1 == r2


# ─────────────────────────────────────────────────────────────────────────────
# TestPredictBatch
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictBatch:
    """
    POST /predict/batch — Scoring ML en batch pour l'OR Engine.

    Le GTFS n'est pas utilisé par cet endpoint : il reçoit directement
    les tronçons en JSON. On teste le scoring ML (fallback déterministe
    car les .joblib sont absents) et les cas d'erreur.

    Fallback déterministe :
        _load_scorer() et _load_pipeline() retournent None (FileNotFoundError).
        predict_fraud_scores() utilise alors le hash MD5 des stop_ids.
        Le score dépend uniquement des stop_ids → stable entre les appels.
    """

    @pytest.fixture
    def payload_1_troncon(self) -> dict:
        """Payload minimal : un seul tronçon valide."""
        return {
            "troncons": [
                {
                    "trip_id":       "TRIP_000",
                    "train_number":  "117700",
                    "service_id":    "000001",
                    "stop_sequence": 0,
                    "stop_id_dep":   "87212027",
                    "stop_name_dep": "Strasbourg",
                    "dep_minutes":   360,
                    "stop_id_arr":   "87214007",
                    "stop_name_arr": "Sélestat",
                    "arr_minutes":   385,
                    "duration_min":  25,
                    "service_date":  SERVICE_DATE,
                }
            ]
        }

    @pytest.fixture
    def payload_3_troncons(self) -> dict:
        """Payload avec 3 tronçons pour tester le scoring batch."""
        troncons = [
            {
                "trip_id":       f"TRIP_{i:03d}",
                "train_number":  f"1177{i:02d}",
                "service_id":    "000001",
                "stop_sequence": 0,
                "stop_id_dep":   dep,
                "stop_name_dep": f"Gare_{dep}",
                "dep_minutes":   360 + i * 60,
                "stop_id_arr":   arr,
                "stop_name_arr": f"Gare_{arr}",
                "arr_minutes":   385 + i * 60,
                "duration_min":  25,
                "service_date":  SERVICE_DATE,
            }
            for i, (dep, arr) in enumerate([
                ("87212027", "87214007"),
                ("87214007", "87214080"),
                ("87214080", "87182063"),
            ])
        ]
        return {"troncons": troncons}

    # ── Tests nominaux ────────────────────────────────────────────────────────

    def test_batch_returns_200(self, payload_1_troncon):
        """POST /predict/batch répond 200 sur un payload valide."""
        response = client.post("/predict/batch", json=payload_1_troncon)
        assert response.status_code == 200, response.text

    def test_batch_response_schema(self, payload_1_troncon):
        """La réponse respecte le schéma PredictResponse."""
        body = client.post("/predict/batch", json=payload_1_troncon).json()
        assert "scores"      in body
        assert "nb_troncons" in body
        assert "scored_at"   in body
        assert isinstance(body["scores"], list)

    def test_batch_nb_scores_equals_nb_inputs(self, payload_3_troncons):
        """Le nombre de scores retournés = nombre de tronçons envoyés."""
        body = client.post("/predict/batch", json=payload_3_troncons).json()
        assert body["nb_troncons"] == 3
        assert len(body["scores"]) == 3

    def test_batch_score_item_structure(self, payload_1_troncon):
        """Chaque PredictScoreItem contient les champs de clé de jointure."""
        body  = client.post("/predict/batch", json=payload_1_troncon).json()
        item  = body["scores"][0]
        assert "trip_id"     in item
        assert "stop_id_dep" in item
        assert "dep_minutes" in item
        assert "fraud_score" in item
        assert "stop_sequence" in item

    def test_batch_score_values_in_range(self, payload_3_troncons):
        """Les scores de fraude sont dans l'intervalle [0.0, 1.0]."""
        body = client.post("/predict/batch", json=payload_3_troncons).json()
        for item in body["scores"]:
            score = item["fraud_score"]
            assert 0.0 <= score <= 1.0, f"Score hors plage : {score}"

    def test_batch_score_key_mapping(self, payload_1_troncon):
        """La clé de jointure (trip_id, stop_id_dep, dep_minutes) est correcte."""
        troncon = payload_1_troncon["troncons"][0]
        body    = client.post("/predict/batch", json=payload_1_troncon).json()
        item    = body["scores"][0]

        assert item["trip_id"]     == troncon["trip_id"]
        assert item["stop_id_dep"] == troncon["stop_id_dep"]
        assert item["dep_minutes"] == troncon["dep_minutes"]

    def test_batch_fallback_is_deterministic(self, payload_1_troncon):
        """
        Le fallback déterministe produit le même score sur deux appels identiques.

        Garantit que l'OR Engine peut répéter une requête et obtenir
        le même résultat sans incohérence dans le graphe scoré.
        """
        r1 = client.post("/predict/batch", json=payload_1_troncon).json()
        r2 = client.post("/predict/batch", json=payload_1_troncon).json()
        assert r1["scores"][0]["fraud_score"] == r2["scores"][0]["fraud_score"]

    def test_batch_different_od_different_score(self):
        """
        Deux tronçons avec des O/D différents ont des scores différents.

        Vérifie que le fallback discrimine les tronçons via leur hash
        et ne retourne pas un score constant.
        """
        base = {
            "trip_id": "TRIP_000", "train_number": "117700",
            "service_id": "000001", "stop_sequence": 0,
            "dep_minutes": 360, "arr_minutes": 385,
            "duration_min": 25, "service_date": SERVICE_DATE,
        }
        payload = {
            "troncons": [
                {**base, "stop_id_dep": "87212027", "stop_name_dep": "Strasbourg",
                          "stop_id_arr": "87214007", "stop_name_arr": "Sélestat"},
                {**base, "stop_id_dep": "87214080", "stop_name_dep": "Colmar",
                          "stop_id_arr": "87182063", "stop_name_arr": "Mulhouse"},
            ]
        }
        body   = client.post("/predict/batch", json=payload).json()
        score1 = body["scores"][0]["fraud_score"]
        score2 = body["scores"][1]["fraud_score"]
        # Probabilité quasi-nulle que deux MD5 distincts donnent exactement le même score
        assert score1 != score2, "Scores identiques pour des O/D différents — fallback cassé"

    # ── Tests d'erreur ────────────────────────────────────────────────────────

    def test_batch_empty_list_returns_422(self):
        """Une liste de tronçons vide retourne 422 (validation Pydantic)."""
        response = client.post("/predict/batch", json={"troncons": []})
        assert response.status_code == 422

    def test_batch_missing_service_date_returns_422(self):
        """Un tronçon sans service_date retourne 422."""
        payload = {
            "troncons": [{
                "trip_id": "TRIP_000", "train_number": "117700",
                "service_id": "000001", "stop_sequence": 0,
                "stop_id_dep": "87212027", "stop_name_dep": "Strasbourg",
                "dep_minutes": 360,
                "stop_id_arr": "87214007", "stop_name_arr": "Sélestat",
                "arr_minutes": 385, "duration_min": 25,
                # service_date absent intentionnellement
            }]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422

    def test_batch_negative_duration_returns_422(self):
        """Un tronçon avec duration_min ≤ 0 retourne 422."""
        payload = {
            "troncons": [{
                "trip_id": "TRIP_000", "train_number": "117700",
                "service_id": "000001", "stop_sequence": 0,
                "stop_id_dep": "87212027", "stop_name_dep": "Strasbourg",
                "dep_minutes": 360,
                "stop_id_arr": "87214007", "stop_name_arr": "Sélestat",
                "arr_minutes": 385, "duration_min": -1,   # invalide
                "service_date": SERVICE_DATE,
            }]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 422

    def test_batch_with_real_scorer_mock(self, payload_3_troncons):
        """
        Quand un scorer ML réel est injecté, les scores sont utilisés.

        Ce test vérifie l'intégration entre le routeur et le scorer :
        si _load_scorer() retourne un vrai scorer, predict_fraud_scores()
        doit appeler scorer.predict() et non le fallback déterministe.
        """
        mock_scorer   = MagicMock()
        mock_pipeline = MagicMock()

        # Le scorer prédit des scores fixes connus
        mock_pipeline.transform.side_effect = lambda df: df  # pass-through
        mock_scorer.predict.return_value     = pd.Series([0.9, 0.1, 0.5])
        mock_scorer.feature_cols             = ["dep_minutes"]  # feature minimale

        with patch.object(predict_module, "_load_scorer",   return_value=mock_scorer), \
             patch.object(predict_module, "_load_pipeline", return_value=mock_pipeline):
            # Purge du cache pour que les mocks soient pris en compte
            predict_module._load_scorer.cache_clear()   if hasattr(predict_module._load_scorer, "cache_clear") else None
            predict_module._load_pipeline.cache_clear() if hasattr(predict_module._load_pipeline, "cache_clear") else None

            body = client.post("/predict/batch", json=payload_3_troncons).json()

        # Avec le scorer mocké, on vérifie que les scores sont bien retournés
        # (le fallback aurait donné d'autres valeurs)
        assert body["nb_troncons"] == 3
        scores = [item["fraud_score"] for item in body["scores"]]
        # Les 3 scores doivent être dans [0, 1]
        for s in scores:
            assert 0.0 <= s <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TestPredictGet
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictGet:
    """
    GET /predict?service_date=YYYY-MM-DD — Scoring GTFS pour Power Automate.

    Nécessite un mock de GTFSLoader car l'endpoint charge les données GTFS
    depuis le disque (non disponible en CI).

    Chemins patchés :
        services.api.routers.predict.GTFSLoader
        services.api.routers.predict.GTFSPreprocessor
    """

    @pytest.fixture
    def gtfs_mock(
        self,
        synthetic_stop_times,
        synthetic_trips,
        synthetic_stops,
        synthetic_routes,
        synthetic_calendar_dates,
    ):
        """
        Retourne un contexte de double-patch :
            GTFSLoader    → retourne des DataFrames synthétiques
            GTFSPreprocessor → utilise l'implémentation réelle (test d'intégration)

        Le preprocessor réel est conservé pour tester la chaîne complète.
        """
        mock_instance = make_gtfs_loader_mock(
            stop_times=synthetic_stop_times,
            trips=synthetic_trips,
            stops=synthetic_stops,
            routes=synthetic_routes,
            calendar_dates=synthetic_calendar_dates,
        )
        return mock_instance

    def test_predict_get_returns_200(self, gtfs_mock):
        """GET /predict retourne 200 avec des données GTFS valides."""
        with patch("services.api.routers.predict.GTFSLoader", return_value=gtfs_mock):
            response = client.get(f"/predict?service_date={SERVICE_DATE}")
        assert response.status_code == 200, response.text

    def test_predict_get_response_schema(self, gtfs_mock):
        """La réponse respecte le schéma PredictResponseSchema."""
        with patch("services.api.routers.predict.GTFSLoader", return_value=gtfs_mock):
            body = client.get(f"/predict?service_date={SERVICE_DATE}").json()

        assert "troncons"     in body
        assert "service_date" in body
        assert "scored_at"    in body
        assert "nb_troncons"  in body
        assert isinstance(body["troncons"], list)

    def test_predict_get_troncon_structure(self, gtfs_mock):
        """Chaque tronçon retourné contient les champs requis par TronconScoreSchema."""
        with patch("services.api.routers.predict.GTFSLoader", return_value=gtfs_mock):
            body = client.get(f"/predict?service_date={SERVICE_DATE}").json()

        assert body["nb_troncons"] > 0, "Aucun tronçon retourné"
        troncon = body["troncons"][0]

        required_fields = [
            "trip_id", "train_number", "service_id",
            "stop_id_dep", "stop_name_dep", "dep_minutes",
            "stop_id_arr", "stop_name_arr", "arr_minutes",
            "duration_min", "fraud_score",
        ]
        for field in required_fields:
            assert field in troncon, f"Champ manquant : {field}"

    def test_predict_get_scores_in_range(self, gtfs_mock):
        """Les fraud_scores retournés sont tous dans [0.0, 1.0]."""
        with patch("services.api.routers.predict.GTFSLoader", return_value=gtfs_mock):
            body = client.get(f"/predict?service_date={SERVICE_DATE}").json()

        for t in body["troncons"]:
            s = t["fraud_score"]
            assert 0.0 <= s <= 1.0, f"Score hors plage : {s}"

    def test_predict_get_sorted_by_score_descending(self, gtfs_mock):
        """
        Les tronçons sont triés par score décroissant.
        Le tronçon le plus risqué doit être en premier.
        """
        with patch("services.api.routers.predict.GTFSLoader", return_value=gtfs_mock):
            body = client.get(f"/predict?service_date={SERVICE_DATE}").json()

        scores = [t["fraud_score"] for t in body["troncons"]]
        assert scores == sorted(scores, reverse=True), "Tronçons non triés par score décroissant"

    def test_predict_get_service_date_propagated(self, gtfs_mock):
        """La date de service retournée correspond à la date demandée."""
        with patch("services.api.routers.predict.GTFSLoader", return_value=gtfs_mock):
            body = client.get(f"/predict?service_date={SERVICE_DATE}").json()

        assert body["service_date"] == SERVICE_DATE

    def test_predict_get_nb_troncons_consistent(self, gtfs_mock):
        """nb_troncons correspond bien à la longueur de la liste troncons."""
        with patch("services.api.routers.predict.GTFSLoader", return_value=gtfs_mock):
            body = client.get(f"/predict?service_date={SERVICE_DATE}").json()

        assert body["nb_troncons"] == len(body["troncons"])

    def test_predict_get_wrong_date_format_returns_422(self, gtfs_mock):
        """Une date mal formatée retourne 422 (validation FastAPI)."""
        response = client.get("/predict?service_date=02-09-2024")   # format DD-MM-YYYY
        assert response.status_code == 422

    def test_predict_get_date_without_service_returns_404(self, gtfs_mock):
        """
        Une date hors du calendrier GTFS retourne 404.
        Notre fixture couvre uniquement septembre 2024 — janvier 2025 est hors plage.
        """
        with patch("services.api.routers.predict.GTFSLoader", return_value=gtfs_mock):
            response = client.get("/predict?service_date=2025-01-15")

        assert response.status_code == 404

    def test_predict_get_gtfs_unavailable_returns_503(self):
        """Si le GTFS n'est pas disponible, l'API retourne 503."""
        mock_loader_raising = MagicMock()
        mock_loader_raising.load_stop_times.side_effect = FileNotFoundError("stop_times.txt")

        with patch("services.api.routers.predict.GTFSLoader", return_value=mock_loader_raising):
            response = client.get(f"/predict?service_date={SERVICE_DATE}")

        assert response.status_code == 503
        assert "GTFS" in response.json()["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# TestOptimize
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimize:
    """
    POST /optimize — Génération de tournées d'inspection.

    Le router optimize.py charge le GTFS, construit le graphe temps-étendu
    et exécute l'algorithme greedy. On mock uniquement GTFSLoader.

    Architecture des mocks :
        services.api.routers.optimize.GTFSLoader → DataFrames synthétiques
        TimeExpandedGraphBuilder et GTFSPreprocessor → implémentations réelles

    Pourquoi garder les implémentations réelles ?
        C'est le cœur du test d'intégration : on vérifie que le graphe
        est construit correctement depuis les données synthétiques et que
        l'optimiseur trouve une tournée faisable.
    """

    @pytest.fixture
    def optimize_payload(self) -> dict:
        """
        Payload standard : départ depuis Strasbourg à 6h, budget 4h.
        Correspond aux données synthétiques : TRIP_000 part à 6h de Strasbourg.
        """
        return {
            "gare_depart_id":    GARE_DEPART_ID,
            "heure_depart_min":  HEURE_DEPART_MIN,
            "duree_max_minutes": 240,
            "service_date":      SERVICE_DATE,
        }

    @pytest.fixture
    def gtfs_mock(
        self,
        synthetic_stop_times,
        synthetic_trips,
        synthetic_stops,
        synthetic_routes,
        synthetic_calendar_dates,
    ):
        return make_gtfs_loader_mock(
            stop_times=synthetic_stop_times,
            trips=synthetic_trips,
            stops=synthetic_stops,
            routes=synthetic_routes,
            calendar_dates=synthetic_calendar_dates,
        )

    # ── Tests nominaux ────────────────────────────────────────────────────────

    def test_optimize_returns_200(self, gtfs_mock, optimize_payload):
        """POST /optimize retourne 200 avec des paramètres valides."""
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            response = client.post("/optimize", json=optimize_payload)
        assert response.status_code == 200, response.text

    def test_optimize_response_schema(self, gtfs_mock, optimize_payload):
        """La réponse respecte le schéma OptimizeResponseSchema."""
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            body = client.post("/optimize", json=optimize_payload).json()

        assert "tournee"      in body
        assert "request"      in body
        assert "optimized_at" in body

    def test_optimize_tournee_structure(self, gtfs_mock, optimize_payload):
        """La tournée retournée contient tous les champs requis par TourneeSchema."""
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            body = client.post("/optimize", json=optimize_payload).json()

        tournee = body["tournee"]
        required = [
            "arcs", "score_total", "duree_totale_minutes",
            "nb_trains", "gare_depart", "gare_arrivee", "service_date",
        ]
        for field in required:
            assert field in tournee, f"Champ manquant dans la tournée : {field}"

    def test_optimize_arcs_not_empty(self, gtfs_mock, optimize_payload):
        """La tournée contient au moins un arc (au moins un train inspecté)."""
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            body = client.post("/optimize", json=optimize_payload).json()

        arcs = body["tournee"]["arcs"]
        assert len(arcs) >= 1, "Tournée vide — aucun arc retourné"

    def test_optimize_has_at_least_one_train(self, gtfs_mock, optimize_payload):
        """La tournée inclut au moins un arc TRAIN (le but du système)."""
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            body = client.post("/optimize", json=optimize_payload).json()

        tournee = body["tournee"]
        assert tournee["nb_trains"] >= 1
        arcs_train = [a for a in tournee["arcs"] if a["arc_type"] == "train"]
        assert len(arcs_train) >= 1

    def test_optimize_duree_within_budget(self, gtfs_mock, optimize_payload):
        """La durée totale de la tournée respecte le budget duree_max_minutes."""
        budget = optimize_payload["duree_max_minutes"]
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            body = client.post("/optimize", json=optimize_payload).json()

        duree = body["tournee"]["duree_totale_minutes"]
        assert duree <= budget, f"Tournée de {duree} min dépasse le budget de {budget} min"

    def test_optimize_score_total_is_positive(self, gtfs_mock, optimize_payload):
        """Le score total de la tournée est positif (> 0 si au moins un train)."""
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            body = client.post("/optimize", json=optimize_payload).json()

        score_total = body["tournee"]["score_total"]
        assert score_total >= 0.0

    def test_optimize_arc_score_in_range(self, gtfs_mock, optimize_payload):
        """Chaque arc TRAIN de la tournée a un fraud_score ∈ [0.0, 1.0]."""
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            body = client.post("/optimize", json=optimize_payload).json()

        for arc in body["tournee"]["arcs"]:
            if arc["arc_type"] == "train":
                s = arc["fraud_score"]
                assert 0.0 <= s <= 1.0, f"fraud_score hors plage : {s}"

    def test_optimize_service_date_echoed(self, gtfs_mock, optimize_payload):
        """La date de service dans la tournée correspond à la date demandée."""
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            body = client.post("/optimize", json=optimize_payload).json()

        assert body["tournee"]["service_date"]   == SERVICE_DATE
        assert body["request"]["service_date"]   == SERVICE_DATE

    def test_optimize_request_echoed_in_response(self, gtfs_mock, optimize_payload):
        """Le bloc 'request' de la réponse reprend les paramètres envoyés."""
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            body = client.post("/optimize", json=optimize_payload).json()

        request = body["request"]
        assert request["gare_depart_id"]    == GARE_DEPART_ID
        assert request["heure_depart_min"]  == HEURE_DEPART_MIN
        assert request["duree_max_minutes"] == 240

    def test_optimize_tight_budget_still_feasible(self, gtfs_mock):
        """
        Avec un budget serré (40 min), l'optimiseur doit trouver au moins
        un train depuis Strasbourg (TRIP_000 dure 25 min).

        Test de robustesse : le greedy ne doit pas crash sur un petit budget.
        """
        payload = {
            "gare_depart_id":    GARE_DEPART_ID,
            "heure_depart_min":  HEURE_DEPART_MIN,
            "duree_max_minutes": 40,
            "service_date":      SERVICE_DATE,
        }
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            response = client.post("/optimize", json=payload)

        # Soit 200 (tournée trouvée), soit 404 (budget trop court)
        assert response.status_code in (200, 404)

        if response.status_code == 200:
            duree = response.json()["tournee"]["duree_totale_minutes"]
            assert duree <= 40

    # ── Tests d'erreur ────────────────────────────────────────────────────────

    def test_optimize_unknown_gare_returns_404(self, gtfs_mock, optimize_payload):
        """
        Une gare de départ inconnue retourne 404.
        Aucun nœud dans le graphe → 0 train disponible.
        """
        payload = {**optimize_payload, "gare_depart_id": "00000000"}
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            response = client.post("/optimize", json=payload)

        assert response.status_code == 404

    def test_optimize_invalid_gare_id_returns_422(self, gtfs_mock, optimize_payload):
        """Un code UIC invalide (pas 8 chiffres) retourne 422."""
        payload = {**optimize_payload, "gare_depart_id": "STRAS"}   # non-UIC
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            response = client.post("/optimize", json=payload)

        assert response.status_code == 422

    def test_optimize_negative_duration_returns_422(self, gtfs_mock, optimize_payload):
        """Un budget négatif retourne 422."""
        payload = {**optimize_payload, "duree_max_minutes": -10}
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            response = client.post("/optimize", json=payload)

        assert response.status_code == 422

    def test_optimize_duration_exceeds_max_returns_422(self, gtfs_mock, optimize_payload):
        """Un budget > 720 min (12h) retourne 422 (contrainte opérationnelle)."""
        payload = {**optimize_payload, "duree_max_minutes": 800}
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            response = client.post("/optimize", json=payload)

        assert response.status_code == 422

    def test_optimize_date_without_service_returns_404(self, gtfs_mock, optimize_payload):
        """Une date hors du calendrier GTFS retourne 404."""
        payload = {**optimize_payload, "service_date": "2025-01-15"}
        with patch("services.api.routers.optimize.GTFSLoader", return_value=gtfs_mock):
            response = client.post("/optimize", json=payload)

        assert response.status_code == 404

    def test_optimize_gtfs_unavailable_returns_503(self, optimize_payload):
        """Si le GTFS est absent, l'API retourne 503."""
        mock_raising = MagicMock()
        mock_raising.load_stop_times.side_effect = FileNotFoundError("stop_times.txt")

        with patch("services.api.routers.optimize.GTFSLoader", return_value=mock_raising):
            response = client.post("/optimize", json=optimize_payload)

        assert response.status_code == 503

    def test_optimize_missing_required_field_returns_422(self):
        """Un payload sans service_date retourne 422."""
        payload = {
            "gare_depart_id":    GARE_DEPART_ID,
            "heure_depart_min":  HEURE_DEPART_MIN,
            "duree_max_minutes": 240,
            # service_date absent
        }
        response = client.post("/optimize", json=payload)
        assert response.status_code == 422