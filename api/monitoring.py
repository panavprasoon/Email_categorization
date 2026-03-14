"""
Prometheus Metrics Module

Defines all Prometheus metrics for monitoring prediction volume,
latency, confidence, and feedback. Exposes a /metrics endpoint.
"""

from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Response

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

prediction_counter = Counter(
    "email_predictions_total",
    "Total number of email predictions made",
    ["category"],
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Time spent processing a single prediction (seconds)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

active_requests = Gauge(
    "active_requests",
    "Number of requests currently being processed",
)

feedback_counter = Counter(
    "feedback_submissions_total",
    "Total feedback submissions",
    ["feedback_type"],
)

model_confidence = Histogram(
    "model_confidence_score",
    "Distribution of model confidence scores",
    buckets=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

batch_prediction_counter = Counter(
    "batch_email_predictions_total",
    "Total number of emails processed in batch mode",
)


# ---------------------------------------------------------------------------
# /metrics endpoint handler
# ---------------------------------------------------------------------------

async def metrics_endpoint() -> Response:
    """Return Prometheus-format metrics for scraping."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
