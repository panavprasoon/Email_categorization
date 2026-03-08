"""
Health Check Routes

Provides endpoints for monitoring service health and readiness.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.dependencies import get_db, get_email_categorizer
from api.models import HealthCheckResponse
from api.config import settings
from inference.categorizer import EmailCategorizer

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/", response_model=HealthCheckResponse)
async def health_check(
    db: Session = Depends(get_db),
    categorizer: EmailCategorizer = Depends(get_email_categorizer)
):
    """
    Basic health check endpoint.
    
    Returns service status, database connectivity, and model status.
    """
    # Check database connection
    try:
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    
    # Check model status
    model_status = "loaded" if categorizer.model is not None else "not_loaded"
    
    # Overall status
    overall_status = "healthy" if (db_status == "connected" and model_status == "loaded") else "unhealthy"
    
    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.APP_VERSION,
        database=db_status,
        model=model_status
    )


@router.get("/live")
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.
    Returns 200 if service is running.
    """
    return {"status": "alive", "timestamp": datetime.utcnow()}


@router.get("/ready")
async def readiness_probe(
    db: Session = Depends(get_db),
    categorizer: EmailCategorizer = Depends(get_email_categorizer)
):
    """
    Kubernetes readiness probe endpoint.
    Returns 200 if service is ready to accept traffic.
    """
    # Check database
    try:
        db.execute("SELECT 1")
    except Exception:
        return {"status": "not_ready", "reason": "database_unavailable"}, 503
    
    # Check model
    if categorizer.model is None:
        return {"status": "not_ready", "reason": "model_not_loaded"}, 503
    
    return {"status": "ready", "timestamp": datetime.utcnow()}
