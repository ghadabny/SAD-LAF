# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — service scheduler
#
# Lance APScheduler qui tourne en continu et déclenche :
#   - le téléchargement GTFS Static tous les jours à 03:00
#   - le fetch GTFS-RT toutes les 2 minutes
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# ── Dépendances ───────────────────────────────────────────────────────────────
COPY services/scheduler/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source ───────────────────────────────────────────────────────────────
# Le scheduler a besoin de :
#   - services/scheduler/  → DailyGTFSFetcher, RealtimeFetcher, main.py
#   - services/ml_engine/  → GTFSDownloader (appelé par DailyGTFSFetcher)
#   - shared/              → config (chemins des dossiers data/)
COPY services/scheduler/   services/scheduler/
COPY services/ml_engine/   services/ml_engine/
COPY shared/               shared/

# ── Lancement ─────────────────────────────────────────────────────────────────
# python -m : exécute le module comme un script
# Le scheduler tourne indéfiniment jusqu'à SIGTERM (docker stop)
CMD ["python", "-m", "services.scheduler.main"]
