# services/or_engine/Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Service : OR Engine — Solveur MILP Orienteering
#
# Rôle : job batch lancé après ml_engine (ou à la demande via l'API).
#   1. Charge les tronçons GTFS du jour
#   2. Construit le graphe temps-étendu (TimeExpandedGraphBuilder)
#   3. Appelle POST /predict/batch pour récupérer les scores ML
#   4. Injecte les scores dans le graphe (inject_scores)
#   5. Résout le Problème d'Orienteering via MILP (PuLP/CBC)
#   6. Exporte la tournée optimisée dans data/outputs/
#
# Note : en fonctionnement normal, l'OR Engine est appelé via l'API (POST /optimize)
# et non directement. Ce Dockerfile est utile pour le mode batch standalone
# (génération massive de tournées en dehors de l'API).
#
# Dépendance critique : coinor-cbc (solveur MILP open source)
#   PuLP embarque CBC en Python mais peut utiliser le binaire système
#   pour des performances accrues sur les grandes instances.
# ─────────────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════════════
# STAGE 1 — builder
# ════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY services/or_engine/requirements.txt ./requirements.txt

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
           httpx==0.26.0 \
           python-dotenv==1.0.0 \
           lightgbm==4.3.0 \
           scikit-learn==1.4.0 \
           joblib==1.3.2


# ════════════════════════════════════════════════════════════════
# STAGE 2 — runtime
# ════════════════════════════════════════════════════════════════
FROM python:3.11-slim AS runtime

# coinor-cbc : solveur MILP natif (CBC = COIN-OR Branch and Cut)
# libgomp1   : requis par LightGBM (si scores locaux plutôt que via API)
RUN apt-get update && apt-get install -y --no-install-recommends \
        coinor-cbc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --shell /bin/bash --uid 1001 sadlaf
USER sadlaf

WORKDIR /app

# Code source nécessaire au solveur OR :
#   services/or_engine/  → solver.py, graph/, optimizer/
#   services/ml_engine/  → GTFSLoader, GTFSPreprocessor (chargement GTFS dans solver)
#   shared/              → config, constants, schemas
COPY --chown=sadlaf:sadlaf services/or_engine/  ./services/or_engine/
COPY --chown=sadlaf:sadlaf services/ml_engine/  ./services/ml_engine/
COPY --chown=sadlaf:sadlaf shared/              ./shared/
COPY --chown=sadlaf:sadlaf config/              ./config/

RUN mkdir -p /app/data/raw/gtfs /app/data/models \
             /app/data/outputs /app/data/processed

ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Point d'entrée : solver.py en mode CLI
# Lance une tournée depuis Strasbourg à 08:00 pour la date du jour
CMD ["python", "services/or_engine/solver.py"]