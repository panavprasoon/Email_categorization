# Email Categorization System — Interview Reference Guide

## 1) Problem Statement

Organizations receive large volumes of emails (support, billing, meeting, spam, urgent, etc.). Manual triage is slow and error-prone. This project automates first-level classification to:
- reduce handling time,
- improve routing consistency,
- collect correction feedback for model improvement.

---

## 2) Scope and Features Delivered

### Core Features
- Single email categorization (`POST /categorize/`)
- Batch-friendly backend structure
- Confidence scoring with predicted category
- Prediction history and analytics
- User feedback endpoint for correction capture

### Model Ops / MLOps Features
- Model version registry in DB
- Model metadata and active model switching
- Drift check endpoint
- Retraining trigger + status tracking
- A/B testing config and promotion endpoints

### Reliability & Observability
- Health and readiness endpoints
- Prometheus metrics endpoint (`/metrics`)
- Structured logging and middleware
- Scheduled batch processor hooks

### Security
- API key-based authentication (`X-API-Key`)
- Environment-driven secrets
- `.env.example` pattern for safe sharing

---

## 3) Tech Stack (What to Say Clearly)

## Backend
- **Python 3.12**
- **FastAPI** for REST APIs
- **Pydantic** for request/response validation
- **SQLAlchemy** ORM
- **PostgreSQL (Neon)** as primary datastore
- **Uvicorn** ASGI server

## ML / NLP
- **scikit-learn** for model training/inference
- **NLTK** for text processing resources (wordnet, stopwords, punkt)
- **NumPy / Pandas / SciPy** for numerical and data operations
- **joblib** for model/vectorizer serialization

## Frontend
- **Streamlit** for interactive UI (classification, analytics, admin pages)

## DevOps / Deployment
- **Docker** (separate API + frontend images)
- **Render** deployment
- **GitHub Actions** CI/CD workflow
- **Prometheus** for metrics scraping

## Testing
- **pytest**, **pytest-asyncio**, **httpx**

---

## 4) High-Level Architecture

1. User submits email in Streamlit frontend.
2. Frontend sends request to FastAPI backend (`X-API-Key` header).
3. Backend preprocesses text and loads active model/vectorizer.
4. Model returns category + confidence.
5. Backend stores email/prediction in PostgreSQL.
6. Frontend shows result and optional feedback form.
7. Feedback writes correction data, used for retraining/drift workflows.
8. Admin routes manage model versions, retraining, A/B promotion.

---

## 5) Data Flow (Interviewer-Friendly)

### Online Inference Flow
- Input: `sender`, `subject`, `body`
- Steps:
  - validate payload,
  - preprocess text,
  - vectorize features,
  - predict label/probability,
  - persist output,
  - return response.

### Feedback Loop
- User disagrees with prediction → submits correct category.
- Feedback stored in DB and used as supervised signal for future retraining.

### Retraining Flow (Admin)
- Trigger retraining endpoint.
- Pipeline builds updated dataset and trains new model.
- Metrics captured and new version registered.
- Admin compares and activates/promotes selected version.

---

## 6) Important API Groups You Should Remember

### Public / Health
- `GET /health/`

### Predictions / Core
- `POST /categorize/`
- `GET /predictions/`
- `GET /predictions/statistics/overview`

### Feedback
- `POST /feedback/`

### Model Management
- `GET /models/info`
- `GET /models/categories`
- `POST /models/reload`

### Admin
- `GET /admin/models`
- `POST /admin/models/{version}/activate`
- `DELETE /admin/models/{version}`
- `POST /admin/retrain`
- `GET /admin/retraining/status`
- `GET /admin/drift`
- `GET /admin/ab-test/results`
- `POST /admin/ab-test/promote`

### Monitoring
- `GET /metrics`

---

## 7) Database Entities (Conceptual)

- Emails / incoming records
- Predictions (category, confidence, timestamp)
- Feedback records (original vs corrected label)
- Model versions (version, status, metrics, activation)
- Retraining jobs/status

What to emphasize: **traceability** from prediction to correction to model iteration.

---

## 8) Deployment Summary

- API service and frontend service are deployed separately on Render.
- Frontend talks to API using `API_BASE_URL`.
- API requires DB URL and auth keys via environment variables.
- Dockerfiles are split by service for cleaner deploy/build.

Expected public URLs:
- Backend: `https://email-system-api.onrender.com`
- Frontend: `https://email-system-frontend.onrender.com`

---

## 9) CI/CD and Quality Practices

- GitHub Actions installs dependencies and runs targeted tests.
- Workflow now includes NLTK corpus download (wordnet, omw-1.4, stopwords, punkt) before tests.
- Docker build job runs after test pass.

What to say in interview:
- “I stabilized CI by making tests deterministic and explicitly provisioning runtime resources (NLTK datasets), so build failures are reproducible and actionable.”

---

## 10) Key Engineering Decisions + Why

1. **FastAPI + Streamlit split**
   - Keeps inference/business logic separate from UI and simplifies deployment.

2. **PostgreSQL over file-based storage**
   - Needed durability, queryability, and model lifecycle metadata at scale.

3. **Feedback-driven retraining hooks**
   - Designed for continuous improvement, not one-time model training.

4. **Metrics + health endpoints**
   - Enables production monitoring and SRE-style troubleshooting.

5. **Versioned model management**
   - Supports safer rollout, rollback, and experimentation (A/B).

---

## 11) Challenges Faced + How You Solved Them

### A) CI test collection failures
- Cause: import-time strict config validation and missing optional packages/data.
- Fix:
  - removed hard-fail at import,
  - added environment-safe DB fallback behavior,
  - added required plotting dependencies,
  - downloaded required NLTK corpora in CI.

### B) Environment consistency
- Cause: local env had resources unavailable in CI.
- Fix: explicit provisioning in workflow and safer defaults.

### C) Secret hygiene
- Cause: risk from local `.env` handling.
- Fix: moved to `.env.example` workflow and reinforced repo hygiene.

---

## 12) Performance / Scalability Talking Points

- API designed statelessly, can scale horizontally behind load balancer.
- DB pooling configured for managed Postgres.
- Caching and batch hooks available for throughput optimization.
- Retraining is asynchronous/admin-triggered to avoid request-path latency impact.

---

## 13) Security Talking Points

- API-key authentication for protected routes.
- Secrets via environment variables; never hardcoded in source.
- Production deployment separated from local `.env` development usage.
- Operational recommendation: periodic key rotation and secret scanning.

---

## 14) Limitations (Be Honest in Interview)

- API-key auth is simpler than full RBAC/OAuth.
- Some legacy tests are still being modernized for full integration coverage.
- Drift/retraining thresholds may require domain tuning with real traffic.

Good framing:
- “I prioritized production-ready core flow first, then planned phased hardening for auth granularity and broader integration testing.”

---

## 15) Improvements You Can Propose (Shows Senior Thinking)

- Add JWT/RBAC for admin vs user operations.
- Add background queue (Celery/RQ) for retraining and heavy batch jobs.
- Add model explainability layer (top tokens/feature contribution).
- Add canary rollout automation and rollback policy.
- Add migration discipline and schema versioning workflow.
- Add full E2E smoke suite against deployed preview environments.

---

## 16) Frequently Asked Interview Questions + Sample Answers

### Q1) “What exactly did you build yourself?”
I built the API layer, model integration, persistence workflow, frontend interaction flow, and deployment/CI automation. I also implemented the model lifecycle controls (feedback, retraining hooks, version activation) and stabilized CI by fixing environment-related test failures.

### Q2) “How does your model improve over time?”
The system captures user feedback on wrong predictions. Those corrected labels become supervised data for retraining. New model versions are registered with metrics and can be activated after validation.

### Q3) “How do you prevent bad models from going live?”
New versions are tracked separately, evaluated, and only activated through controlled admin endpoints. A/B testing and performance checks help compare challenger vs champion before promotion.

### Q4) “How do you monitor production health?”
I exposed `/health/` and `/metrics`, added logging middleware, and integrated Prometheus-compatible metrics for observability.

### Q5) “What was the toughest debugging issue?”
A CI failure caused by missing NLP resources and strict import-time config assumptions. I fixed it by making initialization environment-safe and explicitly downloading required NLTK corpora in CI.

### Q6) “Why FastAPI for this project?”
FastAPI gives fast development, strong typing, automatic docs, and async-friendly architecture—ideal for ML-backed APIs with clear contracts.

### Q7) “How would you scale this to 10x traffic?”
Scale API instances horizontally, optimize DB indexing/pooling, add request/result caching, move heavy tasks to async workers, and apply autoscaling plus robust observability.

---

## 18) 2-Minute Interview Pitch Script

“I developed an email categorization platform that combines ML inference with production API engineering. Users submit email text through a Streamlit frontend, the FastAPI backend classifies it using a trained scikit-learn model, stores predictions in PostgreSQL, and returns category with confidence. I built feedback capture so incorrect predictions become training signal, then added admin controls for retraining, model version activation, drift checks, and A/B comparison. On the ops side, I containerized both services, deployed to Render, added CI with GitHub Actions, and integrated monitoring endpoints. A key part of the project was hardening reliability—fixing CI environment drift, dependency/resource provisioning, and configuration safety. The result is not just an ML model, but a full lifecycle ML product with deployable architecture and maintainability.”

---

## 19) Last-Minute Revision Checklist (Night Before Interview)

- Can I explain the end-to-end request flow in 5 steps?
- Can I justify each tech choice (FastAPI, PostgreSQL, Streamlit, scikit-learn)?
- Can I explain one challenge + fix (CI/NLTK/config hardening)?
- Can I explain how feedback improves model quality?
- Can I name at least 3 future improvements with rationale?

If yes, you are interview-ready for this project.
