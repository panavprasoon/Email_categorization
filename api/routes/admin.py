"""
Admin Routes — model management, retraining, A/B testing, drift monitoring.

All endpoints require a valid X-API-Key header.

Prefix : /admin
Tags   : ["Admin"]
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db, verify_api_key
from database.models import ModelVersion, Prediction, RetrainingJob

router = APIRouter(prefix="/admin", tags=["Admin"])


# ---------------------------------------------------------------------------
# Helper: resolve A/B manager lazily (imported at call-time to avoid circular
# imports at module load).
# ---------------------------------------------------------------------------

def _get_ab_manager():
    from api.ab_testing import build_ab_manager_from_env
    return build_ab_manager_from_env()


# ===========================================================================
# MODEL VERSION MANAGEMENT
# ===========================================================================


@router.get("/models")
async def list_models(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """
    List all registered model versions.

    Returns each version's metadata, training metrics, and active status.
    """
    models = (
        db.query(ModelVersion)
        .order_by(ModelVersion.created_at.desc())
        .all()
    )

    return [
        {
            "version": m.version,
            "is_active": m.is_active,
            "accuracy": m.accuracy,
            "precision_score": m.precision_score,
            "recall_score": m.recall_score,
            "f1_score": m.f1_score,
            "training_samples": m.training_samples,
            "deployed_at": m.deployed_at.isoformat() if m.deployed_at else None,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in models
    ]


@router.post("/models/{version}/activate")
async def activate_model(
    version: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """
    Set a specific model version as the active (production) model.

    Deactivates all other versions first.
    """
    target = (
        db.query(ModelVersion).filter(ModelVersion.version == version).first()
    )
    if target is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found")

    # Deactivate all
    db.query(ModelVersion).update({"is_active": False})
    # Activate target
    target.is_active = True
    target.deployed_at = datetime.utcnow()
    db.commit()

    return {"message": f"Model version '{version}' is now active"}


@router.delete("/models/{version}")
async def delete_model(
    version: str,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """
    Delete a model version and its artifact files.

    Active models cannot be deleted — deactivate first.
    """
    target = (
        db.query(ModelVersion).filter(ModelVersion.version == version).first()
    )
    if target is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found")
    if target.is_active:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the active model. Activate a different version first.",
        )

    # Delete artifact files if they exist
    for path_attr in ("model_path", "vectorizer_path"):
        path = getattr(target, path_attr, None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass  # Non-fatal — record is still removed

    db.delete(target)
    db.commit()

    return {"message": f"Model version '{version}' deleted"}


@router.get("/models/{version}/performance")
async def model_performance(
    version: str,
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """
    Live performance metrics for a specific model version based on recent
    predictions (corrections / correction rate / avg confidence).
    """
    mv = db.query(ModelVersion).filter(ModelVersion.version == version).first()
    if mv is None:
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found")

    cutoff = datetime.utcnow() - timedelta(days=days)
    preds = (
        db.query(Prediction)
        .filter(
            Prediction.model_version_id == mv.id,
            Prediction.created_at >= cutoff,
        )
        .all()
    )

    if not preds:
        return {
            "version": version,
            "window_days": days,
            "message": "No predictions found for this model in the given window",
        }

    total = len(preds)
    avg_conf = sum(p.confidence for p in preds) / total
    avg_ms = sum(p.processing_time_ms or 0 for p in preds) / total
    corrections = sum(
        1
        for p in preds
        if p.feedback and p.feedback.feedback_source == "correction"
    )

    return {
        "version": version,
        "window_days": days,
        "total_predictions": total,
        "avg_confidence": round(avg_conf, 4),
        "avg_processing_time_ms": round(avg_ms, 2),
        "corrections": corrections,
        "correction_rate": round(corrections / total, 4),
        "training_metrics": {
            "accuracy": mv.accuracy,
            "precision_score": mv.precision_score,
            "recall_score": mv.recall_score,
            "f1_score": mv.f1_score,
        },
    }


# ===========================================================================
# RETRAINING
# ===========================================================================


@router.post("/retrain")
async def trigger_retraining(
    background_tasks: BackgroundTasks,
    reason: str = Query("manual", description="Trigger reason for audit trail"),
    api_key: str = Depends(verify_api_key),
):
    """
    Queue a model retraining job in the background.

    The job trains a new model on corrected feedback, evaluates it, and
    registers the new version (inactive) in the database.  Use
    POST /admin/models/{version}/activate to promote it.
    """
    from api.retraining_pipeline import RetrainingPipeline

    pipeline = RetrainingPipeline()
    background_tasks.add_task(pipeline.run_retraining, reason)

    return {
        "message": "Retraining job queued",
        "status": "processing",
        "note": "Check GET /admin/retraining/status for progress",
    }


@router.get("/retraining/status")
async def retraining_status(
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key),
):
    """
    Return the status of the most recent retraining job.
    """
    job = (
        db.query(RetrainingJob)
        .order_by(RetrainingJob.start_time.desc())
        .first()
    )

    if job is None:
        return {"message": "No retraining jobs found"}

    return {
        "job_id": job.id,
        "trigger_reason": job.trigger_reason,
        "status": job.status,
        "started_at": job.start_time.isoformat() if job.start_time else None,
        "completed_at": job.end_time.isoformat() if job.end_time else None,
        "new_model_version": job.new_model_version,
        "training_samples": job.training_samples,
        "validation_accuracy": job.validation_accuracy,
        "promoted": job.promoted,
        "error_log": job.error_log,
    }


# ===========================================================================
# A/B TESTING
# ===========================================================================


@router.get("/ab-test/results")
async def ab_test_results(
    days: int = Query(7, ge=1, le=90),
    api_key: str = Depends(verify_api_key),
):
    """
    Compare champion vs challenger model performance over the past N days.

    Returns per-arm stats: prediction count, avg confidence, correction rate.
    Returns 200 with a message if A/B testing is not configured.
    """
    mgr = _get_ab_manager()
    if mgr is None:
        return {
            "message": (
                "A/B testing is not enabled. "
                "Set AB_CHAMPION_VERSION and AB_CHALLENGER_VERSION env vars."
            )
        }

    return mgr.compare_models(days=days)


@router.post("/ab-test/promote")
async def promote_challenger(
    api_key: str = Depends(verify_api_key),
):
    """
    Promote the A/B challenger to production (active) model.

    Deactivates the current champion and activates the challenger.
    Returns an error if A/B testing is not configured.
    """
    mgr = _get_ab_manager()
    if mgr is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "A/B testing is not configured. "
                "Set AB_CHAMPION_VERSION and AB_CHALLENGER_VERSION env vars."
            ),
        )

    result = mgr.promote_challenger()
    return {"message": "Challenger promoted successfully", **result}


# ===========================================================================
# DRIFT MONITORING
# ===========================================================================


@router.get("/drift")
async def check_drift(
    api_key: str = Depends(verify_api_key),
):
    """
    Run on-demand model drift detection.

    Returns chi-square prediction distribution check and KS confidence
    score check results.
    """
    from api.model_monitoring import DriftDetector

    detector = DriftDetector()
    return detector.run_full_check()
