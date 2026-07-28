# SAD-LAF — Système d'Aide à la Décision · Lutte Anti-Fraude

POC de génération de tournées d'inspection optimisées pour les agents LAF TER Alsace.  
Le système croise les données historiques de fraude (CC/SC/PV) avec les horaires GTFS temps réel pour identifier les trains à haut risque et construire la tournée optimale de la journée.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  GTFS Static  →  Fusion RT  →  Graphe temps-étendu          │
│  Scores ML (LightGBM)  →  Solveur MILP (PuLP/CBC)  →  CSV  │
└─────────────────────────────────────────────────────────────┘
        ↓
   POST /optimize/v2  →  EN_ATTENTE  →  VALIDEE / REFUSEE
        ↓
   CSV  →  OneDrive  →  Power Automate  →  Dataverse
```

**Stack :** Python 3.11 · FastAPI · LightGBM · PuLP/CBC · Docker · GitHub Actions

**Services :**
- `api` — FastAPI, scoring ML et génération de tournées (port 8000)
- `scheduler` — APScheduler, mise à jour GTFS-RT toutes les 2 minutes
- `ml_engine` — job manuel de ré-entraînement LightGBM (nouvelle livraison LAF)

---

## Prérequis

- Docker Desktop ou Rancher Desktop
- Python 3.11+ (pour le dev local uniquement)
- Données LAF (`data/raw/laf/`) — fichiers CC, PV, SC fournis par la DSI

---

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/ton-org/SAD-LAF.git
cd SAD-LAF

# 2. Configurer l'environnement
cp config/.env.example config/.env.dev
# Éditer config/.env.dev si nécessaire (proxy SNCF, chemins)

# 3. Créer les dossiers de données
mkdir -p data/raw/gtfs data/raw/laf data/models data/outputs data/processed

# 4. Copier les fichiers LAF dans data/raw/laf/
#    Extract_DATA GE CC *.csv
#    Extract_DATA GE PV *.csv
#    Extract_DATA GE SC *.csv
```

---

## Démarrage

```bash
# Démarrer l'API et le scheduler
docker compose up -d api scheduler

# Vérifier que tout tourne
docker compose ps
curl http://localhost:8000/health
```

Le GTFS est téléchargé automatiquement au premier démarrage du scheduler.

---

## Ré-entraînement du modèle

À lancer uniquement lors d'une nouvelle livraison de données LAF :

```bash
docker compose run --rm ml_engine
```

Durée estimée : 5-15 minutes selon le volume SC (~35M lignes).  
Le modèle est sauvegardé dans `data/models/lgbm_scorer.joblib`.

---

## API — Endpoints principaux

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Statut de l'API |
| GET | `/predict?service_date=YYYY-MM-DD` | Scorer tous les tronçons TER du jour |
| POST | `/predict/batch` | Scorer un batch de tronçons |
| POST | `/optimize/v2` | Générer une tournée optimisée |
| POST | `/optimize/v2/validate` | Valider une tournée (N+1) |
| POST | `/optimize/v2/refuse` | Refuser une tournée (N+1) |
| POST | `/optimize/v2/cancel` | Annuler une tournée validée |
| GET | `/optimize/v2/pending` | Lister les tournées en attente |
| GET | `/optimize/v2/tournee/{id}` | Consulter une tournée |

Documentation interactive : `http://localhost:8000/docs`

### Exemple — Générer une tournée

```bash
curl -X POST http://localhost:8000/optimize/v2 \
  -H "Content-Type: application/json" \
  -d '{
    "gare_depart_id": "87214056",
    "heure_ps_min": 480,
    "heure_fs_min": 780,
    "service_date": "2026-05-14",
    "agent_id": "AGENT_001",
    "mode": "aller_retour"
  }'
```

---

## Tests

```bash
# Installer les dépendances de test
pip install pytest pytest-cov httpx

# Tests unitaires
pytest tests/unit/ -v

# Tests unitaires avec couverture
pytest tests/unit/ --cov=services --cov=shared --cov-report=term-missing

# Tests d'intégration (requiert l'API démarrée)
pytest tests/integration/ -v
```

La collection Postman complète (24 requêtes, 48 assertions) est disponible dans `tests/SAD_LAF_v2.postman_collection.json`.

---

## Déploiement ThinkStation

### Premier déploiement

```bash
# Sur la ThinkStation, cloner le repo
git clone https://github.com/ton-org/SAD-LAF.git
cd SAD-LAF

# Configurer
cp config/.env.example config/.env.dev

# Copier les données depuis le partage réseau
# \\serveur\partage\laf\ → data/raw/laf\

# Démarrer
docker compose up -d api scheduler
```

### CI/CD automatique

Un runner GitHub Actions self-hosted installé sur la ThinkStation redéploie automatiquement à chaque push sur `main`.

Installation du runner :  
`GitHub repo → Settings → Actions → Runners → New self-hosted runner → Windows`

### Démarrage automatique au boot Windows

```powershell
# Créer une tâche planifiée qui lance Docker Compose au démarrage
schtasks /create /tn "SAD-LAF" /tr "docker compose -f C:\SAD-LAF\docker-compose.yml up -d api scheduler" /sc onstart /ru SYSTEM
```

---

## Structure du projet

```
SAD-LAF/
├── services/
│   ├── api/              # FastAPI — scoring et tournées
│   ├── ml_engine/        # LightGBM — entraînement et prédiction
│   │   ├── data/         # Loaders GTFS, LAF, GTFS-RT
│   │   ├── features/     # Transformateurs de features (SOLID)
│   │   └── models/       # LGBMScorer
│   ├── or_engine/        # Solveur MILP orienteering (PuLP/CBC)
│   │   ├── graph/        # Graphe temps-étendu (Node/Arc)
│   │   └── optimizer/    # OrienteeringOptimizer
│   ├── scheduler/        # APScheduler — GTFS-RT + GTFS Static
│   └── exporter/         # Export CSV pour Power Automate
├── shared/               # Config, schemas Pydantic, constantes
├── tests/
│   ├── unit/             # ~40 modules de tests unitaires
│   └── integration/      # Tests E2E API
├── config/
│   └── .env.example      # Template de configuration
├── data/                 # Données (non commitées)
├── docker-compose.yml
└── github/workflows/
    ├── ci.yml            # Tests automatiques à chaque push
    └── deploy.yml        # Déploiement ThinkStation (self-hosted runner)
```

---

## Variables d'environnement

Voir `config/.env.example` pour la liste complète avec descriptions.

| Variable | Défaut | Description |
|----------|--------|-------------|
| `GTFS_URL` | OpenData SNCF | URL du ZIP GTFS statique |
| `HTTP_PROXY` | _(vide)_ | Proxy réseau SNCF (poste interne uniquement) |
| `PREDICT_API_URL` | `http://api:8000/predict/batch` | URL interne Docker |
| `CORS_ALLOWED_ORIGINS` | localhost | Origines autorisées (prod : domaines SNCF) |

---

## Installation données gtfs manuelle

- **Clean Architecture** — séparation stricte domaine / infrastructure
- Aller sur le site : https://ressources.data.sncf.com/explore/dataset/horaires-sncf/information/
- Cliquer sur le lien : https://eu.ftp.opendatasoft.com/sncf/plandata/Export_OpenData_SNCF_GTFS_NewTripId.zip
- Ouvrir le dossier téléchargé
- Copier le contenu et le coller dans le dossier **SAD-LAF/data/raw/gtfs**

---

## Principes d'architecture

- **Clean Architecture** — séparation stricte domaine / infrastructure
- **SOLID** — DIP via injection de dépendances sur tous les orchestrateurs
- **Fail-soft** — dégradation gracieuse si GTFS-RT ou modèle ML indisponible
- **Persistance atomique** — écriture `.tmp` → rename sur BookingStore et exports CSV

---

## Auteur

Ghada Ben Younes — Alternance ingénieur Data / IA · SNCF Technicentre Grand-Est · 2026
