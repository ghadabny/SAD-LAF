# services/ml_engine/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Service : ML Engine — Entraînement LightGBM
#
# Rôle : job batch lancé une fois par jour (ou manuellement).
#   1. Charge les données LAF (SC, CC, PV) depuis data/raw/laf/
#   2. Charge le GTFS depuis data/raw/gtfs/
#   3. Construit les features (temporelles + historiques LAF)
#   4. Entraîne LightGBM sur le dataset train/test splitté
#   5. Sauvegarde lgbm_scorer.joblib + feature_pipeline.joblib dans data/models/
#
# Ce container NE TOURNE PAS en continu. Il s'exécute, produit les artefacts
# ML et s'arrête (exit 0). L'API charge les .joblib au démarrage suivant.
#
# Démarrage :
#   docker-compose run --rm trainer
#   docker-compose run --rm trainer python services/ml_engine/train.py --force-recompute-laf
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════
# STAGE 1 — builder
# ════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

# LightGBM nécessite build-essential pour compiler ses extensions C++
# pyarrow (cache Parquet) nécessite aussi des libs C
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY services/ml_engine/requirements.txt ./requirements.txt

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
           httpx==0.26.0 \
           python-dotenv==1.0.0 \
           pyarrow>=14.0.0


# ════════════════════════════════════════════════════════════════
# STAGE 2 — runtime
# ════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# libgomp1 requis par LightGBM au runtime (OpenMP multi-thread)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --shell /bin/bash --uid 1001 sadlaf
USER sadlaf

WORKDIR /app

# Code source nécessaire au trainer :
#   services/ml_engine/ → train.py, features/, models/, data/ (LAF + GTFS loaders)
#   shared/             → config, constants, schemas
COPY --chown=sadlaf:sadlaf services/ml_engine/  ./services/ml_engine/
COPY --chown=sadlaf:sadlaf shared/              ./shared/
COPY --chown=sadlaf:sadlaf config/              ./config/

# Dossiers de données (montés depuis l'hôte via volume dans docker-compose)
RUN mkdir -p /app/data/raw/gtfs /app/data/raw/laf \
             /app/data/models /app/data/processed /app/data/outputs

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Commande par défaut : entraînement complet
# Options disponibles :
#   --force-recompute-laf  : recalcule les stats LAF même si le cache est valide
#   --force-recompute-all  : recalcule tout (GTFS + LAF + modèle)
CMD ["python", "services/ml_engine/train.py"]