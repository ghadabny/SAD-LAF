from pathlib import Path
from dotenv import load_dotenv
import os
import yaml

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

    # GTFS Realtime
    GTFS_RT_TRIP_UPDATES_URL: str = os.getenv(
        "GTFS_RT_TRIP_UPDATES_URL",
        "https://proxy.transport.data.gouv.fr/resource/sncf-gtfs-rt-trip-updates"
    )
    GTFS_RT_SERVICE_ALERTS_URL: str = os.getenv(
        "GTFS_RT_SERVICE_ALERTS_URL",
        "https://proxy.transport.data.gouv.fr/resource/sncf-gtfs-rt-service-alerts"
    )
    GTFS_RT_CACHE_DIR: Path = Path(
        os.getenv("GTFS_RT_CACHE_DIR", "data/cache/gtfs_rt")
    )

    # ── Proxy réseau ──────────────────────────────────────────────────────────
    # Obligatoire sur les postes de développement SNCF.
    # Vide sur la ThinkStation et dans Docker (accès internet direct).
    #
    # Source : fichier PAC http://pac.sncf.fr/commun_internet.pac
    # Proxy principal : web-pa-7.access.sncf.fr:8080
    #
    # Dans config/.env.dev (poste SNCF) :
    #     HTTP_PROXY=http://web-pa-7.access.sncf.fr:8080
    #     HTTPS_PROXY=http://web-pa-7.access.sncf.fr:8080
    #
    # Dans config/.env.dev (Docker) :
    #     HTTP_PROXY=
    #     HTTPS_PROXY=
    HTTP_PROXY:  str = os.getenv("HTTP_PROXY",  "")
    HTTPS_PROXY: str = os.getenv("HTTPS_PROXY", "")

    # Chemins des données
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
    GTFS_DIR: Path = Path(os.getenv("GTFS_DIR", "data/raw/gtfs"))
    GTFS_RT_DIR: Path = Path(os.getenv("GTFS_RT_DIR", "data/raw/gtfs_rt"))
    LAF_DIR: Path = Path(os.getenv("LAF_DIR", "data/raw/laf"))
    MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "data/models"))
    OUTPUTS_DIR: Path = Path(os.getenv("OUTPUTS_DIR", "data/outputs"))
    PROCESSED_DIR: Path = Path(os.getenv("PROCESSED_DIR", "data/processed"))
    HYPERPARAMS_PATH: Path = Path('services/ml_engine/config/hyperparameters.yml')

    @property
    def lgbm_params(self) -> dict:
        """Charge et retourne les hyperparamètres LightGBM depuis le fichier YAML."""
        if not self.HYPERPARAMS_PATH.exists():
            print(f"⚠️ Fichier {self.HYPERPARAMS_PATH} introuvable. Utilisation des paramètres par défaut.")
            return {}

        with open(self.HYPERPARAMS_PATH, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            # Retourne le dictionnaire sous la clé 'lgbm', ou un dict vide par défaut
            return data.get('lgbm', {})


# Instance unique importée partout dans le projet
# Usage : from shared.config import config
config = AppConfig()