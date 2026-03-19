# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — service ml_engine
#
# Service batch : lancé par le scheduler une fois par jour.
# Il entraîne ou charge le modèle LightGBM, score tous les tronçons
# du jour et écrit les résultats dans data/outputs/.
#
# Contrairement à l'API et au scheduler, ce container ne tourne pas
# en continu — il s'exécute, produit ses résultats et s'arrête.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# ── Dépendances ───────────────────────────────────────────────────────────────
# LightGBM nécessite libgomp pour le multi-threading (OpenMP).
# Sans cette bibliothèque système, lightgbm ne peut pas s'installer
# sur une image slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY services/ml_engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source ───────────────────────────────────────────────────────────────
COPY services/ml_engine/  services/ml_engine/
COPY shared/              shared/

# ── Lancement ─────────────────────────────────────────────────────────────────
# train.py : entraîne le modèle sur les données LAF + score les tronçons
# TODO : sera implémenté quand les données LAF arriveront
CMD ["python", "-m", "services.ml_engine.train"]
