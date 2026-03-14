"""
Scheduled Batch Processor

Uses APScheduler to run background jobs:
  - Every hour: classify any emails that have no prediction yet.
  - Daily 02:00 UTC: purge audit_logs older than 90 days.
  - Daily 08:00 UTC: generate a performance summary log entry.
  - Daily 01:00 UTC: check for model drift and alert if detected.
"""

import logging
import os
from collections import Counter
from datetime import datetime, timedelta

from sqlalchemy import func

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _APSCHEDULER_AVAILABLE = False

from database.connection import DatabaseConnection
from database.models import AuditLog, Email, ModelVersion, Prediction

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Background scheduler wrapper for recurring maintenance jobs."""

    def __init__(self):
        if not _APSCHEDULER_AVAILABLE:
            logger.warning(
                "APScheduler not installed — batch jobs disabled. "
                "Install with: pip install apscheduler"
            )
            self._scheduler = None
            return

        self._scheduler = BackgroundScheduler(timezone="UTC")

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

    @staticmethod
    def process_unclassified_emails() -> None:
        """Classify any emails that have no Prediction record."""
        conn = DatabaseConnection()
        with conn.get_session() as db:
            # Emails with no prediction
            unclassified = (
                db.query(Email)
                .outerjoin(Prediction, Email.id == Prediction.email_id)
                .filter(Prediction.id.is_(None))
                .all()
            )

            if not unclassified:
                logger.debug("Batch: no unclassified emails found")
                return

            active_mv = (
                db.query(ModelVersion).filter(ModelVersion.is_active == True).first()
            )
            if active_mv is None:
                logger.warning("Batch: no active model version — skipping")
                return

            import pickle

            if not active_mv.model_path or not os.path.exists(active_mv.model_path):
                logger.warning("Batch: active model artifact missing — skipping")
                return

            with open(active_mv.model_path, "rb") as fh:
                model = pickle.load(fh)
            with open(active_mv.vectorizer_path, "rb") as fh:
                vectorizer = pickle.load(fh)

            processed = 0
            for email in unclassified:
                try:
                    vec = vectorizer.transform([email.email_text])
                    label = model.predict(vec)[0]
                    confidence = float(model.predict_proba(vec).max())

                    pred = Prediction(
                        email_id=email.id,
                        model_version_id=active_mv.id,
                        predicted_label=label,
                        confidence=confidence,
                        processing_time_ms=0.0,
                        created_at=datetime.utcnow(),
                    )
                    db.add(pred)
                    processed += 1
                except Exception as exc:
                    logger.error(
                        "Batch: error processing email %s: %s", email.id, exc
                    )

            db.commit()
            logger.info("Batch: classified %d emails", processed)

    @staticmethod
    def cleanup_old_audit_logs() -> None:
        """Delete audit_log rows older than 90 days."""
        conn = DatabaseConnection()
        with conn.get_session() as db:
            cutoff = datetime.utcnow() - timedelta(days=90)
            deleted = (
                db.query(AuditLog)
                .filter(AuditLog.timestamp < cutoff)
                .delete(synchronize_session=False)
            )
            db.commit()
            logger.info("Cleanup: deleted %d old audit log entries", deleted)

    @staticmethod
    def generate_daily_report() -> None:
        """Log a human-readable performance summary for the past 24 h."""
        conn = DatabaseConnection()
        with conn.get_session() as db:
            yesterday = datetime.utcnow() - timedelta(days=1)
            preds = (
                db.query(Prediction)
                .filter(Prediction.created_at >= yesterday)
                .all()
            )

            if not preds:
                logger.info("Daily report: no predictions in the last 24 h")
                return

            total = len(preds)
            avg_conf = sum(p.confidence for p in preds) / total
            dist = Counter(p.predicted_label for p in preds)

            lines = [f"  {cat}: {cnt}" for cat, cnt in dist.most_common()]
            report = (
                f"Daily report ({yesterday.strftime('%Y-%m-%d')})\n"
                f"  Total predictions : {total}\n"
                f"  Avg confidence    : {avg_conf:.2%}\n"
                f"  Category breakdown:\n" + "\n".join(lines)
            )
            logger.info(report)

    @staticmethod
    def run_drift_check() -> None:
        """Check for model drift and send an alert if detected."""
        try:
            from api.model_monitoring import DriftDetector
            from api.alerts import send_alert_sync

            detector = DriftDetector()
            result = detector.run_full_check()

            cat_drift = result["prediction_drift"]
            conf_drift = result["confidence_drift"]

            alerts = []
            if cat_drift.get("status") == "drift_detected":
                alerts.append(
                    f"Category drift detected (p={cat_drift['p_value']:.4f}). "
                    f"{cat_drift['recommendation']}"
                )
            if conf_drift.get("status") == "drift_detected":
                alerts.append(
                    f"Confidence drift detected (KS p={conf_drift['p_value']:.4f}). "
                    f"{conf_drift['recommendation']}"
                )

            if alerts:
                body = "\n\n".join(alerts)
                send_alert_sync(
                    subject="Model Drift Detected",
                    body=body,
                    alert_type="warning",
                )
        except Exception as exc:
            logger.error("Drift check job failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register and start all background jobs."""
        if self._scheduler is None:
            return

        self._scheduler.add_job(
            self.process_unclassified_emails,
            "interval",
            hours=1,
            id="batch_classify",
            max_instances=1,
        )
        self._scheduler.add_job(
            self.cleanup_old_audit_logs,
            "cron",
            hour=2,
            minute=0,
            id="audit_cleanup",
        )
        self._scheduler.add_job(
            self.generate_daily_report,
            "cron",
            hour=8,
            minute=0,
            id="daily_report",
        )
        self._scheduler.add_job(
            self.run_drift_check,
            "cron",
            hour=1,
            minute=0,
            id="drift_check",
        )

        self._scheduler.start()
        logger.info("Batch processor scheduler started (4 jobs registered)")

    def shutdown(self) -> None:
        """Gracefully stop the scheduler."""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Batch processor scheduler stopped")
