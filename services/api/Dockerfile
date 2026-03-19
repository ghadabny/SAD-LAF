# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — service api
#
# Construit une image légère avec FastAPI + Uvicorn.
# python:3.11-slim = image officielle Python sans les outils de build inutiles
# (gcc, make...) — réduit la taille de l'image de ~900 Mo à ~200 Mo.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Dossier de travail dans le container
# Tous les COPY et CMD s'exécutent depuis /app
WORKDIR /app

# ── Dépendances ───────────────────────────────────────────────────────────────
# On copie requirements.txt EN PREMIER (avant le code).
# Pourquoi ? Docker met en cache chaque instruction.
# Si on copie le code d'abord, moindre changement de code
# invalide le cache pip → réinstallation complète à chaque build.
# En copiant requirements.txt séparément, pip n'est réinstallé
# que si requirements.txt change.
COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source ───────────────────────────────────────────────────────────────
# On copie uniquement ce dont l'API a besoin :
#   - services/api/     → les routers FastAPI
#   - services/ml_engine/ → GTFSLoader, GTFSPreprocessor (utilisés par predict)
#   - services/or_engine/ → GraphBuilder (utilisé par optimize)
#   - shared/           → config, constants, schemas
COPY services/api/        services/api/
COPY services/ml_engine/  services/ml_engine/
COPY services/or_engine/  services/or_engine/
COPY shared/              shared/

# ── Lancement ─────────────────────────────────────────────────────────────────
# --host 0.0.0.0 : écoute sur toutes les interfaces réseau du container
#                  (sinon localhost dans le container ≠ accessible depuis l'hôte)
# --port 8000    : port interne du container (mappé vers l'hôte dans docker-compose)
# --workers 2    : 2 workers pour gérer les requêtes en parallèle
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
