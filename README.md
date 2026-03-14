# Email Categorization System — Knowledge Transfer (KT) Document

## 1) Purpose and Scope

This repository contains a production-ready email categorization platform with:
- ML training and inference pipeline
- FastAPI backend with authentication and persistence
- Streamlit frontend for classification, analytics, and admin operations
- Deployment artifacts for Docker and Render
- Monitoring, drift checks, retraining pipeline, and model lifecycle APIs

This document is intended for a **new engineer or AI agent** to take ownership quickly.

---

## 2) Current Project Status

Implemented and available:
- End-to-end classification flow (API + frontend)
- Feedback capture flow
- Prediction history and analytics endpoints
- Model info and reload endpoints
- Admin endpoints for retraining, drift check, model version management, A/B test results/promote
- Prometheus `/metrics` endpoint
- Scheduled batch processor (hourly/daily jobs)
- CI workflow + Docker configs

Assumed deployment state:
- API and frontend deployed on Render
- baseline smoke tests already completed

---

## 3) High-Level Architecture

### 3.1 Runtime Components

1. **Frontend (Streamlit)**
   - User enters sender/subject/body
   - Calls API with `X-API-Key`
   - Displays prediction, confidence, analytics, admin actions

2. **API (FastAPI)**
   - Handles authentication, validation, routing, business logic
   - Stores emails, predictions, feedback, model versions, retraining jobs in PostgreSQL
   - Exposes monitoring/admin endpoints

3. **Database (PostgreSQL / Neon)**
   - Core persistent store for operational and ML metadata

4. **Model Artifacts**
   - Trained model/vectorizer artifacts read by inference
   - Retrained artifacts stored with version metadata

5. **Optional Monitoring Stack**
   - Prometheus + Grafana using `docker-compose.monitoring.yml`

### 3.2 Request/Data Flow

1. Frontend sends classify request (`/categorize/`)
2. API service loads active model/vectorizer and predicts label+confidence
3. API persists email + prediction rows
4. User may submit feedback (`/feedback/`)
5. Feedback powers drift/retraining logic
6. Admin can trigger retraining and activate model versions

---

## 4) Repository Layout (What Matters Most)

- `api/` — FastAPI app
  - `main.py` — app bootstrap, middleware, router wiring, startup/shutdown hooks
  - `routes/` — endpoint groups (`categorization`, `predictions`, `feedback`, `models`, `admin`, `health`)
  - `services/` — business logic
  - `monitoring.py`, `model_monitoring.py`, `alerts.py`, `retraining_pipeline.py`, `ab_testing.py`, `batch_processor.py`
- `database/` — SQLAlchemy models, connection, repository helpers
- `inference/` — model loading/categorizer runtime
- `training/` and `feature_pipeline/` — model training and preprocessing
- `frontend/` — Streamlit UI
  - `app.py`
  - `pages/1_Email_Classifier.py`
  - `pages/2_Analytics.py`
  - `pages/3_Admin_Panel.py`
- `tests/` — test suite
- `Dockerfile.api`, `Dockerfile.frontend`, `docker-compose.yml`
- `render.yaml`, `.github/workflows/ci-cd.yml`

---

## 5) Environment Variables (Required and Optional)

### 5.1 API Service

Required:
- `DATABASE_URL`
- `SECRET_KEY`
- `API_KEY_1`
- `API_KEY_2`
- `ENVIRONMENT=production`

Optional but recommended:
- `SENTRY_DSN`
- `LOG_LEVEL`
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_EMAIL`
- `AB_CHAMPION_VERSION`, `AB_CHALLENGER_VERSION`, `AB_TRAFFIC_SPLIT`

### 5.2 Frontend Service

Required:
- `API_BASE_URL` (public URL of API service)
- `API_KEY` (same as `API_KEY_1`)
- `API_TIMEOUT_SECONDS` (default `30`)

---

## 6) Local Development Runbook

### 6.1 Prerequisites
- Python 3.11+ (project currently uses a venv with Python 3.12)
- PostgreSQL connection (Neon)
- `.env` populated with required values

### 6.2 Start API

```powershell
.\venv\Scripts\Activate.ps1
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6.3 Start Frontend

```powershell
.\venv\Scripts\Activate.ps1
streamlit run frontend/app.py
```

### 6.4 Local Smoke Test
- Open frontend (`http://localhost:8501`)
- Classify one email
- Submit feedback
- Check analytics/admin pages
- Open `http://localhost:8000/health/`

---

## 7) Docker Runbook

### 7.1 Start app stack

```powershell
docker compose up --build
```

### 7.2 Stop stack

```powershell
docker compose down
```

### 7.3 Optional monitoring stack (with app stack)

```powershell
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

Grafana: `http://localhost:3000`
Prometheus: `http://localhost:9090`

---

## 8) API Surface (Operationally Important)

### 8.1 Public/Health
- `GET /health/`

### 8.2 Core Prediction
- `POST /categorize/`
- `POST /feedback/`
- `GET /predictions/`
- `GET /predictions/statistics/overview`

### 8.3 Model Ops
- `GET /models/info`
- `GET /models/categories`
- `POST /models/reload`

### 8.4 Admin Ops
- `GET /admin/models`
- `POST /admin/models/{version}/activate`
- `DELETE /admin/models/{version}`
- `GET /admin/models/{version}/performance`
- `POST /admin/retrain`
- `GET /admin/retraining/status`
- `GET /admin/ab-test/results`
- `POST /admin/ab-test/promote`
- `GET /admin/drift`

### 8.5 Monitoring
- `GET /metrics`

Authentication:
- Most endpoints require `X-API-Key`

---

## 9) Operations Playbook

### 9.1 Retraining
1. Ensure sufficient corrected feedback exists
2. Trigger `POST /admin/retrain`
3. Poll `GET /admin/retraining/status`
4. If new version is acceptable, activate via `POST /admin/models/{version}/activate`

### 9.2 Drift Handling
1. Run `GET /admin/drift`
2. If drift is detected repeatedly, trigger retraining
3. Validate model performance before activation

### 9.3 A/B Testing
1. Set `AB_CHAMPION_VERSION`, `AB_CHALLENGER_VERSION`, `AB_TRAFFIC_SPLIT`
2. Redeploy API
3. Monitor `GET /admin/ab-test/results`
4. Promote challenger when metrics justify it

---

## 10) CI/CD and Deployment Notes

- CI workflow: `.github/workflows/ci-cd.yml`
- Render config: `render.yaml`
- API and frontend should be independently deployable services
- After env var changes, always redeploy service

### 10.1 Deployed URLs

- Backend API: https://email-system-api.onrender.com
- Frontend App: https://email-system-frontend.onrender.com

---

## 11) Common Failure Modes and Quick Fixes

1. **401 Unauthorized**
   - Check `X-API-Key` and service env vars (`API_KEY_1`, frontend `API_KEY`)

2. **500 during categorization**
   - Check model artifacts and DB connectivity
   - Inspect Render logs

3. **Drift endpoint reports insufficient data**
   - Normal in low-volume early phase; collect more predictions

4. **Retraining skipped**
   - Not enough correction records yet; continue collecting feedback

5. **Frontend cannot reach API**
   - Verify `API_BASE_URL` in frontend service env and API health URL

---

## 12) Code Ownership Pointers (Where to Modify What)

- API route behavior changes: `api/routes/*.py`
- Business logic: `api/services/*.py`
- Model lifecycle/retraining: `api/retraining_pipeline.py`, `api/routes/admin.py`
- Monitoring/metrics: `api/monitoring.py`, `api/model_monitoring.py`, `api/batch_processor.py`
- Frontend UX and API wiring: `frontend/pages/*.py`, `frontend/utils/api_client.py`
- DB schema/model changes: `database/models.py`, `database/create_schema.sql`

---

## 13) Suggested Next Improvements

- Add role-based admin authorization (not just API key)
- Add automated integration tests for `/admin/*` routes
- Add migration tooling discipline for schema evolution
- Add dashboard JSON export/versioning for Grafana
- Add retries/backoff for external SMTP/Sentry operations

---

## 14) Handover Checklist (For New Engineer / AI)

1. Read this README fully
2. Verify local API + frontend startup
3. Validate `/health/`, `/docs`, `/metrics`
4. Execute one classify + feedback flow
5. Test one admin retraining cycle end-to-end
6. Confirm model activation flow works
7. Confirm deployment env vars match this doc
8. Review CI runs on latest commit

If all 8 pass, handover is complete.
