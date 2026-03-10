import pytest
import pandas as pd
from datetime import datetime, timedelta


# ----------------------------------------------------------------
# Fixtures GTFS synthétiques
# Simulent la structure réelle des fichiers GTFS SNCF
# sans dépendre de données réelles
# ----------------------------------------------------------------

@pytest.fixture
def synthetic_gtfs():
    """
    Génère des données GTFS synthétiques cohérentes :
    - 5 trips
    - 3 arrêts par trip (→ 2 tronçons par trip)
    - Gares du réseau alsacien

    Retourne: (stop_times, trips, stops)
    """
    gares = [
        {"stop_id": "87212027", "stop_name": "Strasbourg", "stop_lat": 48.585, "stop_lon": 7.735},
        {"stop_id": "87214007", "stop_name": "Sélestat", "stop_lat": 48.259, "stop_lon": 7.450},
        {"stop_id": "87214080", "stop_name": "Colmar", "stop_lat": 48.080, "stop_lon": 7.356},
        {"stop_id": "87182063", "stop_name": "Mulhouse", "stop_lat": 47.742, "stop_lon": 7.340},
        {"stop_id": "87191007", "stop_name": "Obernai", "stop_lat": 48.459, "stop_lon": 7.481},
    ]

    stops = pd.DataFrame([
        {**g, "stop_desc": "", "zone_id": "", "stop_url": "",
         "location_type": 0, "parent_station": ""}
        for g in gares
    ])

    trips_data = []
    stop_times_data = []
    base_time = datetime(2024, 9, 2, 6, 0)

    for i in range(5):
        trip_id = f"TRIP_{i:03d}"
        train_number = f"1177{i:02d}"
        route_id = "FR:Line::TER_ALSACE:"
        service_id = "000001"

        trips_data.append({
            "route_id":      route_id,
            "service_id":    service_id,
            "trip_id":       trip_id,
            "trip_headsign": train_number,
            "direction_id":  0,
            "block_id":      i,
            "shape_id":      "",
        })

        # 3 arrêts par trip
        for seq, gare in enumerate(gares[:3]):
            depart = base_time + timedelta(hours=i, minutes=seq * 25)
            arrivee = depart + timedelta(minutes=20)
            stop_times_data.append({
                "trip_id":           trip_id,
                "arrival_time":      arrivee.strftime("%H:%M:%S"),
                "departure_time":    depart.strftime("%H:%M:%S"),
                "stop_id":           gare["stop_id"],
                "stop_sequence":     seq,
                "stop_headsign":     "",
                "pickup_type":       0,
                "drop_off_type":     0,
                "shape_dist_traveled": "",
            })

    trips = pd.DataFrame(trips_data)
    stop_times = pd.DataFrame(stop_times_data)

    return stop_times, trips, stops


@pytest.fixture
def synthetic_routes():
    """
    Génère une table routes synthétique avec des lignes TER (route_type=2).
    """
    return pd.DataFrame([
        {
            "route_id":        "FR:Line::TER_ALSACE:",
            "agency_id":       "1187",
            "route_short_name": "TER",
            "route_long_name": "TER Alsace",
            "route_desc":      "",
            "route_type":      2,   # Rail régional = TER
            "route_url":       "",
            "route_color":     "006600",
            "route_text_color": "FFFFFF",
        }
    ])


@pytest.fixture
def synthetic_calendar_dates():
    """
    Génère un calendrier de circulation synthétique.
    """
    records = []
    for i in range(30):
        date = datetime(2024, 9, 1) + timedelta(days=i)
        records.append({
            "service_id":     "000001",
            "date":           int(date.strftime("%Y%m%d")),
            "exception_type": 1,
        })
    return pd.DataFrame(records)


# ----------------------------------------------------------------
# Fixtures LAF synthétiques
# Simulent l'historique des contrôles LAF
# À adapter quand le format réel sera connu
# ----------------------------------------------------------------

@pytest.fixture
def synthetic_laf():
    """
    Génère un historique de contrôles LAF synthétique.
    Les noms de colonnes seront mis à jour quand le fichier réel sera reçu.
    """
    import numpy as np
    np.random.seed(42)

    records = []
    base_date = datetime(2024, 1, 1)

    for i in range(500):
        trip_id = f"TRIP_{np.random.randint(0, 5):03d}"
        date_controle = base_date + timedelta(days=i % 180)
        nb_voyageurs = np.random.randint(10, 80)
        nb_irreg = np.random.randint(0, 8)

        records.append({
            "trip_id":               trip_id,
            "date_controle":         date_controle,
            "nb_voyageurs_controles": nb_voyageurs,
            "nb_irregularites":      nb_irreg,
            "taux_fraude":           round(nb_irreg / nb_voyageurs, 4),
            "agent_id":              f"AGENT_{np.random.randint(1, 6):02d}",
        })

    return pd.DataFrame(records)