# AGENTS.md — PowerGuard AI

> This file is the **single source of truth** for all AI agents, developers, and contributors
> working on the PowerGuard AI project. Read it in full before making any changes.

---

## 1. Project Summary

**PowerGuard AI** is an intelligent grid resilience platform that predicts power outages,
ranks at-risk electrical assets, and generates prioritised maintenance plans.

| Field | Value |
|---|---|
| **Problem** | Power transformer and substation failures cause major outages |
| **Solution** | ML-based risk prediction + weather fusion + maintenance planning |
| **Team** | NullPointers |
| **Track** | AI |
| **Hackathon** | IBM Bob AI Hackathon |

---

## 2. Architecture Overview

PowerGuard AI follows a **three-tier architecture**:

```
┌──────────────────────────────────────────────────────┐
│                   Frontend (Browser)                 │
│   HTML + CSS + JavaScript + Chart.js                 │
│   Templates: src/frontend/templates/                 │
│   Static:    src/frontend/static/                    │
└───────────────────────┬──────────────────────────────┘
                        │  HTTP (JSON REST API)
┌───────────────────────▼──────────────────────────────┐
│               Backend (Flask — Python)               │
│                                                      │
│  src/backend/app.py          ← Flask app factory     │
│  src/backend/api/            ← Route blueprints      │
│    assets.py                 ← GET/POST /api/assets  │
│    predict.py                ← POST /api/predict     │
│    weather.py                ← GET  /api/weather     │
│    dashboard.py              ← GET  /api/dashboard   │
│  src/backend/core/config.py  ← Centralised config    │
│  src/backend/services/       ← Business logic        │
│    prediction_service.py     ← ML prediction orchestr│
│    weather_service.py        ← Weather data fetcher  │
│    maintenance_service.py    ← Plan generator        │
│  src/backend/db/             ← Database layer        │
│    models.py                 ← SQLAlchemy models     │
│    database.py               ← DB init + helpers     │
│  src/backend/utils/          ← Shared helpers        │
└───────────────────────┬──────────────────────────────┘
                        │  SQLAlchemy ORM
┌───────────────────────▼──────────────────────────────┐
│               Database (SQLite)                      │
│   data/powerguard.db                                 │
│   Tables: assets, sensor_readings, weather_records,  │
│           incidents, risk_predictions, maintenance   │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│           ML Pipeline (scikit-learn — Python)        │
│                                                      │
│  src/ml/data/loader.py           ← CSV → DataFrame   │
│  src/ml/data/preprocessor.py     ← Clean + encode    │
│  src/ml/features/                                    │
│    feature_engineering.py        ← Derive features   │
│  src/ml/models/                                      │
│    risk_classifier.py            ← RF Classifier     │
│    train.py                      ← Training script   │
│    artifacts/                    ← Saved .joblib     │
│  src/ml/evaluation/evaluator.py  ← Metrics + plots   │
│  src/ml/notebooks/               ← EDA notebooks     │
└──────────────────────────────────────────────────────┘
```

---

## 3. Data Model

### 3.1 Core Entities

| Table | Description | Key Columns |
|---|---|---|
| `assets` | Power transformers and substations | `id`, `name`, `asset_type`, `region`, `voltage_kv`, `install_year`, `latitude`, `longitude`, `criticality_score` |
| `sensor_readings` | Time-series health sensor data | `id`, `asset_id`, `timestamp`, `temperature_c`, `oil_level_pct`, `load_factor_pct`, `vibration_hz`, `dissolved_gas_ppm` |
| `weather_records` | Regional weather snapshots | `id`, `region`, `timestamp`, `temperature_c`, `humidity_pct`, `wind_speed_kmh`, `precipitation_mm`, `storm_flag` |
| `incidents` | Historical outage records | `id`, `asset_id`, `incident_date`, `incident_type`, `duration_hours`, `root_cause`, `severity` |
| `risk_predictions` | ML prediction outputs | `id`, `asset_id`, `predicted_at`, `risk_score`, `risk_level`, `contributing_factors`, `model_version` |
| `maintenance_plan` | Generated action items | `id`, `asset_id`, `generated_at`, `priority_rank`, `action_type`, `recommended_date`, `crew_required`, `notes` |

### 3.2 Risk Levels

| Level | Probability Range | Meaning |
|---|---|---|
| **HIGH** | ≥ 0.70 | Immediate action required |
| **MEDIUM** | 0.40 – 0.69 | Schedule maintenance within 7 days |
| **LOW** | < 0.40 | Monitor; schedule during next cycle |

---

## 4. ML Model Design

### 4.1 Problem Framing
Multi-class classification: predict `risk_level` ∈ {LOW, MEDIUM, HIGH} for each asset.

### 4.2 Feature Set (planned)

| Feature | Source | Description |
|---|---|---|
| `asset_age_years` | assets | current year − install_year |
| `criticality_score` | assets | grid impact weight (0–10) |
| `temp_rolling_mean_7d` | sensor_readings | 7-day rolling mean of temperature |
| `temp_rolling_std_7d` | sensor_readings | 7-day rolling std of temperature |
| `oil_level_mean_30d` | sensor_readings | 30-day rolling mean of oil level |
| `load_over_90_count_30d` | sensor_readings | count of readings where load > 90% in 30 days |
| `vibration_mean_7d` | sensor_readings | 7-day rolling mean of vibration |
| `days_since_last_incident` | incidents | days since most recent incident |
| `incident_count_12m` | incidents | incident count in last 12 months |
| `weather_risk_index` | weather_records | composite: wind × precipitation × storm_flag |
| `humidity_mean_7d` | weather_records | 7-day rolling mean of humidity |

### 4.3 Algorithm
- **Primary**: `RandomForestClassifier` (scikit-learn) — interpretable, feature importances
- **Pipeline**: `StandardScaler` → `RandomForestClassifier`
- **Hyperparameters**: tuned via `GridSearchCV` or `RandomizedSearchCV`
- **Serialisation**: `joblib.dump` → `src/ml/models/artifacts/risk_model_v{N}.joblib`

### 4.4 Evaluation Metrics
- Accuracy, Precision, Recall, F1 (per class + weighted)
- ROC-AUC (one-vs-rest)
- Confusion matrix

---

## 5. REST API Contract

All responses use JSON. All list endpoints support `?limit=` and `?offset=` pagination.

| Method | Endpoint | Description | Phase |
|---|---|---|---|
| GET | `/api/assets/` | List all assets with latest risk scores | 2 |
| GET | `/api/assets/<id>` | Single asset detail + sensor history | 2 |
| POST | `/api/assets/` | Register a new asset | 2 |
| PUT | `/api/assets/<id>` | Update asset metadata | 2 |
| POST | `/api/predict/asset/<id>` | Run prediction for one asset | 3 |
| POST | `/api/predict/batch` | Run predictions for all assets | 3 |
| GET | `/api/predict/latest` | Most recent predictions (all assets) | 3 |
| GET | `/api/weather/current/<region>` | Current weather for a region | 3 |
| GET | `/api/weather/forecast/<region>` | 5-day forecast for a region | 3 |
| GET | `/api/dashboard/summary` | KPI counts (total, high, medium, low) | 4 |
| GET | `/api/dashboard/risk-map` | Assets with lat/lon + risk level | 4 |
| GET | `/api/dashboard/maintenance` | Prioritised maintenance plan | 4 |

### Standard Error Response
```json
{
  "error": "Human-readable message",
  "code":  "MACHINE_READABLE_CODE"
}
```

---

## 6. Development Roadmap

### Phase 1 — Foundation ✅ (Current)
- [x] Define architecture and data model
- [x] Create folder structure and stub files
- [x] Write AGENTS.md and README.md
- [x] Flask app factory (no routes yet)
- [x] Centralised configuration
- [x] Frontend HTML/CSS skeleton + Chart.js scaffold
- [x] requirements.txt

### Phase 2 — Data Layer & Backend Skeleton
- [ ] SQLAlchemy models (`src/backend/db/models.py`)
- [ ] Database init + seed script (`src/scripts/init_db.py`)
- [ ] Synthetic sample data generator (`src/scripts/generate_sample_data.py`)
- [ ] Assets API blueprint (`src/backend/api/assets.py`)
- [ ] Unit tests for models and config

### Phase 3 — ML Pipeline
- [ ] Data loader and preprocessor
- [ ] Feature engineering module
- [ ] `RiskClassifier` training and serialisation
- [ ] Model evaluator (metrics + confusion matrix)
- [ ] Prediction service
- [ ] Predict API blueprint
- [ ] Weather service + Weather API blueprint
- [ ] EDA notebook (`src/ml/notebooks/01_eda.ipynb`)

### Phase 4 — Dashboard & Integration
- [ ] Maintenance service (scoring + ranking)
- [ ] Dashboard API blueprint
- [ ] Live KPI cards wired to API
- [ ] Risk distribution chart wired to API
- [ ] Asset health trend chart wired to API
- [ ] Asset risk rankings table wired to API
- [ ] Maintenance plan page wired to API

### Phase 5 — Polish & Submission
- [ ] Responsive CSS refinements
- [ ] Error handling and loading states in JS
- [ ] Integration tests
- [ ] `docs/setup-guide.md` (how to run)
- [ ] `docs/architecture.md` (detailed diagrams)
- [ ] `docs/solution-overview.md`
- [ ] Demo screenshots and video
- [ ] `submission.yaml` filled in

---

## 7. Coding Standards

### Python
- Python 3.11+
- Follow **PEP 8** (max line length: 100 characters)
- Use **type hints** on all function signatures
- Use **docstrings** on all public classes and functions (Google style)
- No `import *`; import what you need explicitly
- Use `pathlib.Path` for file paths, never string concatenation
- Never hard-code secrets — always read from `os.environ` via `config.py`

### Flask
- Use the **app factory pattern** (`create_app()`) — never a global `app` instance
- Group routes into **Blueprints** by resource (`/api/assets`, `/api/predict`, etc.)
- Return JSON with proper HTTP status codes (200 / 201 / 400 / 404 / 500)
- Validate request inputs before processing; return 400 on bad input
- Do not put business logic in route handlers — delegate to `services/`

### JavaScript
- Vanilla JS (ES6+) — no frameworks
- All API calls use `fetch()` with `.then()/.catch()` or `async/await`
- Chart.js charts must be destroyed before re-creating on data refresh
- Keep `dashboard.js`, `assets.js`, etc. as separate files per page

### HTML / CSS
- Semantic HTML5 elements (`<nav>`, `<main>`, `<section>`, `<footer>`)
- BEM-like class naming for CSS (`.kpi-card__label`, `.kpi-card--high`)
- No inline styles except where unavoidable
- Mobile-first; use CSS Grid / Flexbox for layout

### Testing
- All new Python functions must have at least one unit test
- Use `pytest` with `pytest-flask` for Flask route tests
- Test files mirror source structure: `src/tests/unit/test_<module>.py`

### Git
- Branch naming: `feature/<short-description>`, `fix/<short-description>`
- Commit messages: `type: short description` (e.g., `feat: add risk prediction endpoint`)
- Never commit `.env`, `*.db`, `*.joblib`, `__pycache__/`, or `venv/`

---

## 8. File & Folder Reference

```
powerguard-ai/
├── run.py                          ← Flask development entry point
├── data/
│   ├── raw/                        ← Real/external datasets (gitignored)
│   ├── processed/                  ← Preprocessed CSVs (gitignored)
│   └── sample/                     ← Synthetic sample data (committed)
├── src/
│   ├── backend/
│   │   ├── app.py                  ← Flask app factory
│   │   ├── api/
│   │   │   ├── assets.py           ← /api/assets blueprint
│   │   │   ├── predict.py          ← /api/predict blueprint
│   │   │   ├── weather.py          ← /api/weather blueprint
│   │   │   └── dashboard.py        ← /api/dashboard blueprint
│   │   ├── core/
│   │   │   └── config.py           ← Centralised config (env vars)
│   │   ├── db/
│   │   │   ├── models.py           ← SQLAlchemy ORM models
│   │   │   └── database.py         ← DB init / session helpers
│   │   ├── services/
│   │   │   ├── prediction_service.py
│   │   │   ├── weather_service.py
│   │   │   └── maintenance_service.py
│   │   └── utils/                  ← Shared helper functions
│   ├── ml/
│   │   ├── data/
│   │   │   ├── loader.py           ← Load CSV → DataFrame
│   │   │   └── preprocessor.py     ← Clean + encode
│   │   ├── features/
│   │   │   └── feature_engineering.py
│   │   ├── models/
│   │   │   ├── risk_classifier.py  ← RF classifier wrapper
│   │   │   ├── train.py            ← Training script
│   │   │   └── artifacts/          ← Saved .joblib model files
│   │   ├── evaluation/
│   │   │   └── evaluator.py        ← Metrics + plots
│   │   └── notebooks/              ← Jupyter EDA notebooks
│   ├── frontend/
│   │   ├── templates/
│   │   │   ├── index.html          ← Dashboard page
│   │   │   ├── assets.html         ← Asset registry page
│   │   │   └── maintenance.html    ← Maintenance plan page
│   │   └── static/
│   │       ├── css/main.css        ← Global stylesheet
│   │       ├── js/dashboard.js     ← Dashboard charts + table
│   │       └── images/             ← Icons, logos
│   ├── scripts/
│   │   ├── init_db.py              ← Create DB tables
│   │   └── generate_sample_data.py ← Synthetic data generator
│   ├── tests/
│   │   ├── unit/                   ← pytest unit tests
│   │   └── integration/            ← pytest Flask integration tests
│   ├── config/                     ← Extra config files (logging.yaml etc.)
│   └── requirements.txt            ← Python dependencies
├── docs/
│   ├── architecture.md             ← Detailed architecture diagrams
│   ├── problem-statement.md        ← Problem framing
│   ├── solution-overview.md        ← How it works
│   └── setup-guide.md              ← How to install and run
├── demo/                           ← Screenshots + demo video links
├── presentation/                   ← Slide deck
├── AGENTS.md                       ← THIS FILE
└── README.md                       ← Project README
```

---

## 9. Environment Setup (Quick Reference)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r src/requirements.txt

# 3. Configure environment
cp src/.env.example .env
# Edit .env with your values

# 4. Run the application
python run.py
# Open http://localhost:5000 in your browser
```

---

## 10. Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| Flask (not FastAPI) | Simpler, more familiar for students; Jinja2 templates avoid separate frontend build step |
| SQLite (not PostgreSQL) | Zero-setup for development and demo; easily replaced by swapping the DB URI |
| RandomForestClassifier | Interpretable, handles mixed feature types, built-in feature importances, no GPU required |
| scikit-learn Pipeline | Prevents data leakage between train/test; scaler + model serialised together |
| Vanilla JS (no React) | Reduces complexity; Chart.js handles all visualisation needs |
| App factory pattern | Enables test isolation via `TestingConfig`; avoids circular import issues |
| Blueprints for routes | Keeps each resource's API in one file; easy to add/remove features |
| Services layer | Keeps route handlers thin; business logic is testable independently of HTTP |

---

*Last updated: Phase 1 — Foundation*
