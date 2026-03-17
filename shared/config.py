from pathlib import Path
from dotenv import load_dotenv
import os

# Charge le fichier .env.dev automatiquement
load_dotenv(dotenv_path=Path("config/.env.dev"))


class AppConfig:
    """
    Centralise toutes les variables de configuration du projet.
    Si un chemin change, on le modifie uniquement dans .env.dev.
    Les valeurs par défaut sont utilisées si .env.dev est absent.
    """

    # URLs sources de données
    GTFS_URL: str = os.getenv(
        "GTFS_URL",
        "https://eu.ftp.opendatasoft.com/sncf/plandata/export-opendata-sncf-gtfs.zip"
    )

    # GTFS-RT Trip Updates : retards en temps réel
    # Mis à jour toutes les 2 minutes, trains des 60 prochaines minutes
    GTFS_RT_TRIP_UPDATES_URL: str = os.getenv(
        "GTFS_RT_TRIP_UPDATES_URL",
        "https://proxy.transport.data.gouv.fr/resource/sncf-gtfs-rt-trip-updates"
    )

    # GTFS-RT Service Alerts : suppressions et alertes de service
    # Mis à jour toutes les 2 minutes
    GTFS_RT_SERVICE_ALERTS_URL: str = os.getenv(
        "GTFS_RT_SERVICE_ALERTS_URL",
        "https://proxy.transport.data.gouv.fr/resource/sncf-gtfs-rt-service-alerts"
    )

    # Chemins des données
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
    GTFS_DIR: Path = Path(os.getenv("GTFS_DIR", "data/raw/gtfs"))
    GTFS_RT_DIR: Path = Path(os.getenv("GTFS_RT_DIR", "data/raw/gtfs_rt"))
    LAF_DIR: Path = Path(os.getenv("LAF_DIR", "data/raw/laf"))
    MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "data/models"))
    OUTPUTS_DIR: Path = Path(os.getenv("OUTPUTS_DIR", "data/outputs"))
    PROCESSED_DIR: Path = Path(os.getenv("PROCESSED_DIR", "data/processed"))


# Instance unique importée partout dans le projet
# Usage : from shared.config import config
config = AppConfig()