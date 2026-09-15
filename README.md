# ⚡ PowerGuard AI

> **Predicting power grid failures before they happen.**  
> An intelligent asset risk platform combining sensor health data, weather forecasts,
> and historical incident records to protect electrical infrastructure.

---

## 👥 Team

| Field | Value |
|---|---|
| **Team Name** | NullPointers |
| **Track** | AI |
| **Hackathon** | IBM Bob AI Hackathon |

---

## 🎯 Problem Statement

Power transformer and substation failures can cause widespread, costly outages that affect
homes, hospitals, and critical infrastructure. Grid operators currently rely on reactive
maintenance — they respond after a failure occurs rather than preventing it.

There is no unified system that combines **asset health sensor data**, **weather forecasts**,
and **historical incident records** to predict which assets are most likely to fail,
rank them by how severely their failure would affect the grid, and automatically
generate a prioritised maintenance and crew pre-positioning plan.

---

## 💡 Solution

PowerGuard AI is a web-based decision-support platform that ingests real-time sensor
readings (temperature, oil level, load factor, vibration, dissolved gas), regional weather
data, and historical outage records to train a machine learning risk classifier.

The system assigns each power transformer and substation a **risk score** and **risk level**
(HIGH / MEDIUM / LOW), ranks assets by the combined impact of failure probability and grid
criticality, and outputs a **prioritised maintenance plan** with recommended actions and
crew pre-positioning guidance — turning reactive maintenance into proactive grid resilience.

---

## ✨ Key Features

- **🔴 Risk Prediction** — RandomForest classifier predicts outage risk (HIGH / MEDIUM / LOW) for each asset, trained on sensor + weather + incident features
- **📊 Interactive Dashboard** — Real-time KPI cards, risk distribution chart, and asset health trend visualisation powered by Chart.js
- **🗺️ Asset Risk Rankings** — Ranked table of all grid assets sorted by combined risk score × criticality weight
- **🔧 Maintenance Plan Generator** — Automatically produces a prioritised action list with recommended dates and crew requirements
- **🌦️ Weather Integration** — Fuses live or stored weather data (storm flag, wind, precipitation, humidity) into risk scoring

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python 3.11, JavaScript (ES6+), HTML5, CSS3 |
| **Backend Framework** | Flask 3.0 (app factory + Blueprints) |
| **Database** | SQLite via SQLAlchemy ORM |
| **Machine Learning** | scikit-learn (RandomForestClassifier, StandardScaler, Pipeline) |
| **Data Processing** | pandas, NumPy |
| **Frontend Charts** | Chart.js 4 |
| **IBM Technologies** | IBM Bob AI (development assistant) |
| **Other** | python-dotenv, joblib, requests |

---

## 📁 Repository Structure

```
powerguard-ai/
├── run.py                          ← Flask development entry point
├── AGENTS.md                       ← Architecture, rules & roadmap
├── data/
│   ├── raw/                        ← Real/external datasets (gitignored)
│   ├── processed/                  ← Preprocessed CSVs (gitignored)
│   └── sample/                     ← Synthetic sample data (committed)
├── src/
│   ├── backend/
│   │   ├── app.py                  ← Flask app factory
│   │   ├── api/                    ← REST API blueprints
│   │   │   ├── assets.py           ←   /api/assets
│   │   │   ├── predict.py          ←   /api/predict
│   │   │   ├── weather.py          ←   /api/weather
│   │   │   └── dashboard.py        ←   /api/dashboard
│   │   ├── core/config.py          ← Centralised configuration
│   │   ├── db/                     ← SQLAlchemy models + DB init
│   │   └── services/               ← Business logic layer
│   ├── ml/
│   │   ├── data/                   ← Loader + Preprocessor
│   │   ├── features/               ← Feature engineering
│   │   ├── models/                 ← Classifier + training script
│   │   ├── evaluation/             ← Metrics + plots
│   │   └── notebooks/              ← Jupyter EDA notebooks
│   ├── frontend/
│   │   ├── templates/              ← Jinja2 HTML pages
│   │   └── static/                 ← CSS + JavaScript + images
│   ├── scripts/                    ← DB init + sample data generator
│   ├── tests/                      ← pytest unit + integration tests
│   └── requirements.txt
├── docs/
│   ├── architecture.md
│   ├── problem-statement.md
│   ├── solution-overview.md
│   └── setup-guide.md
├── demo/
└── presentation/
```

---

## ⚡ How to Run

### Prerequisites
- Python 3.11 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/powerguard-ai.git
cd powerguard-ai

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install Python dependencies
pip install -r src/requirements.txt

# 4. Configure environment variables
cp src/.env.example .env
# Open .env and set SECRET_KEY (and optionally OPENWEATHER_API_KEY)

# 5. Initialise the database  (Phase 2 — run once)
# python src/scripts/init_db.py

# 6. Start the Flask development server
python run.py
```

Open **http://localhost:5000** in your browser.

> **Note:** The backend API and ML model are scaffolded but not yet implemented.
> The dashboard skeleton is visible, but data will be live in Phase 2–4.

---

## 🗺️ Development Roadmap

| Phase | Status | Description |
|---|---|---|
| **Phase 1 — Foundation** | ✅ Complete | Architecture, folder structure, Flask skeleton, CSS/JS scaffold |
| **Phase 2 — Data Layer** | 🔲 Next | SQLAlchemy models, DB init, sample data generator, Assets API |
| **Phase 3 — ML Pipeline** | 🔲 Planned | Feature engineering, RF classifier, training, Predict + Weather APIs |
| **Phase 4 — Dashboard** | 🔲 Planned | Live charts, risk table, maintenance plan page, full API integration |
| **Phase 5 — Polish** | 🔲 Planned | Tests, docs, responsive UI, demo video, submission.yaml |

---

## 🖥️ Demo

| Artifact | Link |
|---|---|
| 📹 Demo Video | [See demo/demo-video-link.txt](demo/demo-video-link.txt) |
| 🌐 Live Demo | [See demo/live-demo-url.txt](demo/live-demo-url.txt) |
| 🖼️ Screenshots | [See demo/screenshots/](demo/screenshots/) |
| 📊 Presentation | [See presentation/](presentation/) |

---

## ⚠️ Known Limitations (Phase 1)

- Backend API endpoints are **scaffolded but not yet implemented** — all return placeholder responses
- The ML model has **not been trained** — feature engineering and training scripts are stubs
- The dashboard **does not fetch live data** yet — Chart.js canvases are empty
- **No authentication** — this is a prototype/demo system
- Weather integration requires an optional OpenWeatherMap API key; defaults to stored records

---

## 🏅 Architecture Highlights

The most noteworthy aspect of this submission is the **clean three-tier architecture**:
a thin Flask API layer, a dedicated services layer for business logic (prediction, weather,
maintenance planning), and an independently runnable ML pipeline. This separation means
each component can be tested, replaced, or extended without touching the others —
exactly the kind of production-quality thinking that differentiates a hackathon prototype
from a real system.

---

## 📄 Architecture & Rules

See [`AGENTS.md`](AGENTS.md) for the full architecture reference, data model,
API contract, coding standards, and development roadmap.

---
