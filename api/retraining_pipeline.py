"""
Retraining Pipeline

Feedback-driven model retraining.  Pulls corrected predictions from the
database, trains a new scikit-learn model, evaluates it, saves artifacts,
and registers the new version in ModelVersion.

Field mapping (actual ORM columns):
    RetrainingJob : start_time, end_time, status, new_model_version,
                    trigger_reason, training_samples, validation_accuracy,
                    promoted, error_log
    ModelVersion  : version, is_active, model_path, vectorizer_path,
                    accuracy, precision_score, recall_score, f1_score,
                    training_samples
"""

import logging
import os
import pickle
import uuid
from datetime import datetime
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from database.connection import DatabaseConnection
from database.models import Email, Feedback, ModelVersion, Prediction, RetrainingJob

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("RETRAIN_MODEL_DIR", "artifacts/retrained")


class RetrainingPipeline:
    """
    End-to-end retraining pipeline driven by user feedback corrections.

    Args:
        min_feedback_count: Minimum number of corrections before triggering.
        correction_rate_threshold: Trigger retraining when correction rate
            exceeds this fraction (default 0.15 = 15 %).
    """

    def __init__(
        self,
        min_feedback_count: int = 50,
        correction_rate_threshold: float = 0.15,
    ):
        self.min_feedback_count = min_feedback_count
        self.correction_rate_threshold = correction_rate_threshold
        os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Criteria check
    # ------------------------------------------------------------------

    def check_retraining_criteria(self, db: Session) -> dict:
        """Return whether retraining should be triggered and why."""
        correction_count = (
            db.query(Feedback)
            .filter(Feedback.feedback_source == "correction")
            .count()
        )

        total_feedback = db.query(Feedback).count()
        correction_rate = (
            correction_count / total_feedback if total_feedback > 0 else 0.0
        )

        if correction_count < self.min_feedback_count:
            return {
                "should_retrain": False,
                "reason": (
                    f"Insufficient corrections: {correction_count} / "
                    f"{self.min_feedback_count} required"
                ),
                "correction_count": correction_count,
            }

        return {
            "should_retrain": True,
            "reason": (
                f"High correction rate ({correction_rate:.1%})"
                if correction_rate > self.correction_rate_threshold
                else f"Sufficient corrections collected: {correction_count}"
            ),
            "correction_count": correction_count,
            "correction_rate": correction_rate,
        }

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_training_data(self, db: Session) -> Tuple:
        """
        Build (X_train, X_test, y_train, y_test) from feedback corrections.

        Joins Prediction → Feedback → Email to get email text + corrected label.
        """
        rows = (
            db.query(Email.email_text, Feedback.corrected_label)
            .join(Prediction, Email.id == Prediction.email_id)
            .join(Feedback, Prediction.id == Feedback.prediction_id)
            .filter(Feedback.feedback_source == "correction")
            .all()
        )

        if len(rows) < 50:
            raise ValueError(
                f"Only {len(rows)} corrected samples — need at least 50 to retrain."
            )

        texts = [r.email_text for r in rows]
        labels = [r.corrected_label for r in rows]

        return train_test_split(texts, labels, test_size=0.2, random_state=42)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(self, X_train, y_train) -> Tuple:
        """Fit TF-IDF + RandomForest on training data."""
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_vec = vectorizer.fit_transform(X_train)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_vec, y_train)

        return model, vectorizer

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(self, model, vectorizer, X_test, y_test) -> dict:
        """Return accuracy / precision / recall / f1 on held-out set."""
        X_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_vec)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(
        self,
        model,
        vectorizer,
        metrics: dict,
        training_samples: int,
        db: Session,
    ) -> str:
        """Save artifacts to disk and register in database."""
        version_id = f"retrained_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        model_path = os.path.join(MODEL_DIR, f"{version_id}_model.pkl")
        vec_path = os.path.join(MODEL_DIR, f"{version_id}_vectorizer.pkl")

        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)
        with open(vec_path, "wb") as fh:
            pickle.dump(vectorizer, fh)

        mv = ModelVersion(
            version=version_id,
            accuracy=metrics["accuracy"],
            precision_score=metrics["precision"],
            recall_score=metrics["recall"],
            f1_score=metrics["f1_score"],
            is_active=False,  # must be promoted explicitly
            model_path=model_path,
            vectorizer_path=vec_path,
            training_samples=training_samples,
            created_at=datetime.utcnow(),
        )
        db.add(mv)
        db.commit()

        logger.info("New model version saved: %s  accuracy=%.4f", version_id, metrics["accuracy"])
        return version_id

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run_retraining(self, trigger_reason: str = "manual") -> dict:
        """
        Execute the full retraining pipeline.

        Creates a RetrainingJob record at start and updates it on completion
        or failure.  Returns a result dict with success flag and details.
        """
        conn = DatabaseConnection()
        with conn.get_session() as db:
            job = RetrainingJob(
                trigger_reason=trigger_reason,
                status="running",
                start_time=datetime.utcnow(),
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            job_id = job.id
            logger.info("Retraining job %s started (reason: %s)", job_id, trigger_reason)

        try:
            with conn.get_session() as db:
                job = db.query(RetrainingJob).filter(RetrainingJob.id == job_id).first()

                criteria = self.check_retraining_criteria(db)
                if not criteria["should_retrain"]:
                    job.status = "skipped"
                    job.end_time = datetime.utcnow()
                    job.error_log = criteria["reason"]
                    db.commit()
                    return {"success": False, "skipped": True, **criteria}

                X_train, X_test, y_train, y_test = self.prepare_training_data(db)
                model, vectorizer = self.train_model(X_train, y_train)
                metrics = self.evaluate_model(model, vectorizer, X_test, y_test)
                version_id = self.save_model(
                    model, vectorizer, metrics, len(X_train) + len(X_test), db
                )

                job.status = "completed"
                job.end_time = datetime.utcnow()
                job.new_model_version = version_id
                job.training_samples = len(X_train) + len(X_test)
                job.validation_accuracy = metrics["accuracy"]
                db.commit()

                logger.info(
                    "Retraining job %s completed: version=%s  accuracy=%.4f",
                    job_id,
                    version_id,
                    metrics["accuracy"],
                )

                return {
                    "success": True,
                    "job_id": job_id,
                    "version_id": version_id,
                    "metrics": metrics,
                }

        except Exception as exc:
            logger.error("Retraining job %s failed: %s", job_id, exc, exc_info=True)
            with conn.get_session() as db:
                job = db.query(RetrainingJob).filter(RetrainingJob.id == job_id).first()
                if job:
                    job.status = "failed"
                    job.end_time = datetime.utcnow()
                    job.error_log = str(exc)
                    db.commit()
            return {"success": False, "error": str(exc)}
