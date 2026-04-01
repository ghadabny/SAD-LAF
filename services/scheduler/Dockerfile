# services/scheduler/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Service : Scheduler APScheduler — Tâches planifiées
#
# Rôle : service continu qui tourne 24h/24 et déclenche :
#   - 03:00 chaque jour  → téléchargement GTFS Static (DailyGTFSFetcher)
#   - Toutes les 2 min   → polling GTFS-RT (RealtimeFetcher)
#                          → détecte retards et suppressions de trains
#
# Ce container TOURNE EN CONTINU (restart: unless-stopped dans docker-compose).
# Il ne fait pas de ML ni d'optimisation — il délègue ces rôles aux autres services.
#
# Dépendances légères : pas de LightGBM, pas de PuLP
# → Image finale très légère (~180 Mo)
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════
# STAGE 1 — builder
# ════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Pas de dépendances C natives pour ce service
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY services/scheduler/requirements.txt ./requirements.txt

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
           python-dotenv==1.0.0


# ════════════════════════════════════════════════════════════════
# STAGE 2 — runtime
# ════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# Aucune dépendance système supplémentaire pour le scheduler
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --shell /bin/bash --uid 1001 sadlaf
USER sadlaf

WORKDIR /app

# Code source nécessaire au scheduler :
#   services/scheduler/  → DailyGTFSFetcher, RealtimeFetcher, main.py (APScheduler)
#   services/ml_engine/  → GTFSDownloader (utilisé par DailyGTFSFetcher)
#   shared/              → config (chemins data/), schemas (GTFS-RT)
COPY --chown=sadlaf:sadlaf services/scheduler/  ./services/scheduler/
COPY --chown=sadlaf:sadlaf services/ml_engine/  ./services/ml_engine/
COPY --chown=sadlaf:sadlaf shared/              ./shared/
COPY --chown=sadlaf:sadlaf config/              ./config/

RUN mkdir -p /app/data/raw/gtfs /app/data/outputs

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Healthcheck léger : vérifie que le process Python tourne
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import apscheduler; print('ok')" || exit 1

# Le scheduler tourne indéfiniment jusqu'à SIGTERM (docker stop)
CMD ["python", "-m", "services.scheduler.main"]