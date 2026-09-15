"""Smoke-test every route and API endpoint."""
from src.backend.app import create_app
from src.backend.core.config import TestingConfig

app = create_app(TestingConfig())
client = app.test_client()

tests = []

def check(label, r, expected_status=200, check_key=None):
    ok = r.status_code == expected_status
    if check_key and ok:
        data = r.get_json() or {}
        ok = check_key in data
    tests.append((ok, label, r.status_code))
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label} -> {r.status_code}")

# Page routes
check("GET /",            client.get("/"))
check("GET /assets",      client.get("/assets"))
check("GET /maintenance", client.get("/maintenance"))

# API health
check("GET /api/health",  client.get("/api/health"), check_key="status")

# Assets API
check("GET /api/assets/",           client.get("/api/assets/"),           check_key="assets")
check("GET /api/assets/1",          client.get("/api/assets/1"),          check_key="name")
check("GET /api/assets/1/readings", client.get("/api/assets/1/readings"), check_key="readings")

# Predict API
check("POST /api/predict/1",   client.post("/api/predict/1"),   check_key="risk_level")
check("POST /api/predict/all", client.post("/api/predict/all"), check_key="predictions")

# Dashboard API
check("GET /api/dashboard/summary",
      client.get("/api/dashboard/summary"), check_key="risk_counts")
check("GET /api/dashboard/risk-trend",
      client.get("/api/dashboard/risk-trend"), check_key="trends")
check("GET /api/dashboard/map-data",
      client.get("/api/dashboard/map-data"), check_key="assets")
check("GET /api/dashboard/feature-importance",
      client.get("/api/dashboard/feature-importance"), check_key="feature_importances")

# Weather API
check("POST /api/weather/query",
      client.post("/api/weather/query",
                  json={"state": "TX", "tmpf": 95, "relh": 80, "sknt": 25}),
      check_key="severity_class")
check("GET /api/weather/maintenance",
      client.get("/api/weather/maintenance"), check_key="plan")
check("GET /api/weather/maintenance?regenerate=true",
      client.get("/api/weather/maintenance?regenerate=true"), check_key="plan")

# Predict payload
payload = {
    "hydrogen_ppm": 650, "oxygen_ppm": 8000, "nitrogen_ppm": 35000,
    "methane_ppm": 280,  "co_ppm": 1800,     "co2_ppm": 14000,
    "ethylene_ppm": 250, "ethane_ppm": 180,  "acetylene_ppm": 18.0,
    "dbds_mg_kg": 4.2,   "power_factor_pct": 2.1,
    "interfacial_tension_mNm": 16.0, "dielectric_kv": 22.0,
    "water_content_ppm": 48.0, "state": "TX",
}
check("POST /api/predict/payload (HIGH DGA)",
      client.post("/api/predict/payload", json=payload), check_key="risk_level")

passed = sum(1 for ok, _, __ in tests if ok)
total  = len(tests)
print(f"\n  {passed}/{total} checks passed.")
if passed < total:
    print("  FAILURES:")
    for ok, label, code in tests:
        if not ok:
            print(f"    {label} -> {code}")
