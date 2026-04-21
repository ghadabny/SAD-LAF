# services/api/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Service : API FastAPI — Scoring ML + Génération de tournées
#
# Rôle : expose les endpoints HTTP consommés par Power Automate et l'OR Engine.
#   GET  /health
#   GET  /predict?service_date=YYYY-MM-DD    → scores ML triés
#   POST /predict/batch                      → scoring à la demande (OR Engine)
#   POST /optimize                           → tournée optimisée MILP
#
# Multi-stage build :
#   Stage builder  → compile les wheels natives (LightGBM, scipy)
#   Stage runtime  → image finale légère, sans outils de compilation
#
# Taille estimée : ~650 Mo (vs ~1.4 Go en single-stage)
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════
# STAGE 1 — builder : installation des dépendances
# ════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Outils système requis pour compiler les extensions natives
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copier les requirements en premier → cache pip si le code change mais pas les deps
COPY services/api/requirements.txt ./requirements.txt

# Venv isolé → pas d'interférence avec le Python système
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt


# ════════════════════════════════════════════════════════════════
# STAGE 2 — runtime : image finale de production
# ════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# Seule dépendance système runtime : libgomp pour LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier le venv compilé depuis le builder (pas de pip en prod)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Utilisateur non-root (bonne pratique sécurité)
RUN useradd --create-home --shell /bin/bash --uid 1001 sadlaf
USER sadlaf

WORKDIR /app

# Code source — uniquement ce dont l'API a besoin :
#   services/api/       → routers FastAPI (health, predict, optimize)
#   services/ml_engine/ → GTFSLoader, GTFSPreprocessor, LGBMScorer, FeaturePipeline
#   services/or_engine/ → TimeExpandedGraphBuilder, OrienteeringOptimizer, solver
#   shared/             → config, constants, schemas Pydantic
COPY --chown=sadlaf:sadlaf services/api/        ./services/api/
COPY --chown=sadlaf:sadlaf services/ml_engine/  ./services/ml_engine/
COPY --chown=sadlaf:sadlaf services/or_engine/  ./services/or_engine/
COPY --chown=sadlaf:sadlaf services/exporter/   ./services/exporter/
COPY --chown=sadlaf:sadlaf shared/              ./shared/
COPY --chown=sadlaf:sadlaf config/              ./config/

# Dossier de données monté via docker-compose (volume externe)
RUN mkdir -p /app/data/raw/gtfs /app/data/raw/laf \
             /app/data/models /app/data/outputs /app/data/processed

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Healthcheck : Docker redémarre le container si l'API ne répond plus
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# --workers 1 : LRU cache du scorer en mémoire → un seul worker pour cohérence
#               (plusieurs workers = plusieurs caches LightGBM non synchronisés)
CMD ["uvicorn", "services.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]