"""
Database Repository Pattern
Provides clean data access layer for business logic
Works with Neon cloud PostgreSQL
"""
from datetime import datetime, date
from typing import List, Optional, Dict, Any
import logging

from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from .models import Email, Prediction, Feedback, ModelVersion, AuditLog, InferenceMetadata, RetrainingJob

logger = logging.getLogger(__name__)

class EmailRepository:
    """Data access methods for Email table"""
    
    @staticmethod
    def create(session: Session, email_text: str) -> Email:
        """Create new email record"""
        email = Email(email_text=email_text)
        session.add(email)
        session.flush()  # Get ID without committing
        return email

    @staticmethod
    def create_email(
        session: Optional[Session] = None,
        subject: Optional[str] = None,
        body: Optional[str] = None,
        sender: Optional[str] = None
    ) -> Email:
        """Create a new email record.

        This is a convenience wrapper used by higher-level business logic.
        The Email model currently only stores raw text, so only `body` is used.

        Args:
            session: Optional SQLAlchemy session. If not provided, a new session
                will be created.
            subject: Email subject (not stored in current schema).
            body: Email body (stored as email_text).
            sender: Email sender (not stored in current schema).

        Returns:
            Created Email object.
        """
        if session is None:
            from .connection import DatabaseConnection

            db = DatabaseConnection()
            with db.get_session() as sess:
                email = EmailRepository.create(sess, email_text=body or '')
                # Ensure the returned object retains its primary key outside the session
                sess.expunge(email)
                return email

        return EmailRepository.create(session, email_text=body or '')
    
    @staticmethod
    def get_by_id(session: Session, email_id: int) -> Optional[Email]:
        """Retrieve email by ID"""
        return session.query(Email).filter(Email.id == email_id).first()
    
    @staticmethod
    def get_recent(session: Session, limit: int = 100) -> List[Email]:
        """Get recent emails"""
        return session.query(Email).order_by(desc(Email.created_at)).limit(limit).all()

class ModelVersionRepository:
    """Data access methods for ModelVersion table"""
    
    @staticmethod
    def create(
        session: Session,
        version: str,
        accuracy: float,
        precision_score: float,
        recall_score: float,
        f1_score: float,
        model_path: str,
        vectorizer_path: str,
        training_samples: int,
        training_metrics: Dict[str, Any]
    ) -> ModelVersion:
        """Register new model version"""
        model_version = ModelVersion(
            version=version,
            accuracy=accuracy,
            precision_score=precision_score,
            recall_score=recall_score,
            f1_score=f1_score,
            model_path=model_path,
            vectorizer_path=vectorizer_path,
            training_samples=training_samples,
            training_metrics=training_metrics,
            is_active=False
        )
        session.add(model_version)
        session.flush()
        return model_version
    
    @staticmethod
    def get_active(session: Session) -> Optional[ModelVersion]:
        """Get currently active model"""
        return session.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    
    @staticmethod
    def activate(session: Session, version: str) -> bool:
        """
        Activate a model version (deactivates all others)
        Trigger in database ensures only one active model
        """
        model = session.query(ModelVersion).filter(ModelVersion.version == version).first()
        if model:
            model.is_active = True
            model.deployed_at = datetime.now()
            session.flush()
            logger.info(f"Activated model version: {version}")
            return True
        return False
    
    @staticmethod
    def get_all(session: Session, limit: int = 20) -> List[ModelVersion]:
        """Get all model versions ordered by creation date"""
        return session.query(ModelVersion).order_by(desc(ModelVersion.created_at)).limit(limit).all()

class PredictionRepository:
    """Data access methods for Prediction table"""
    
    @staticmethod
    def create(
        session: Session,
        email_id: int,
        model_version_id: int,
        predicted_label: str,
        confidence: float,
        prediction_probabilities: Dict[str, float],
        processing_time_ms: float
    ) -> Prediction:
        """Log new prediction"""
        prediction = Prediction(
            email_id=email_id,
            model_version_id=model_version_id,
            predicted_label=predicted_label,
            confidence=confidence,
            prediction_probabilities=prediction_probabilities,
            processing_time_ms=processing_time_ms
        )
        session.add(prediction)
        session.flush()
        return prediction
    
    @staticmethod
    def get_by_id(session: Session, prediction_id: int) -> Optional[Prediction]:
        """Get prediction by ID"""
        return session.query(Prediction).filter(Prediction.id == prediction_id).first()

    @staticmethod
    def get_by_email(session: Session, email_id: int) -> List[Prediction]:
        """Get predictions for a specific email"""
        return (
            session.query(Prediction)
            .filter(Prediction.email_id == email_id)
            .order_by(desc(Prediction.created_at))
            .all()
        )
    
    @staticmethod
    def get_recent(session: Session, limit: int = 100) -> List[Prediction]:
        """Get recent predictions"""
        return session.query(Prediction).order_by(desc(Prediction.created_at)).limit(limit).all()
    
    @staticmethod
    def get_correction_rate(session: Session, days: int = 7) -> float:
        """Calculate correction rate over last N days"""
        total = session.query(func.count(Prediction.id)).scalar()
        if total == 0:
            return 0.0
        corrected = session.query(func.count(Feedback.id)).scalar()
        return (corrected / total) * 100

class FeedbackRepository:
    """Data access methods for Feedback table"""
    
    @staticmethod
    def create(
        session: Session,
        prediction_id: int,
        corrected_label: str,
        user_id: Optional[str] = None,
        feedback_source: str = 'manual'
    ) -> Feedback:
        """Store user correction"""
        # Check if feedback already exists for this prediction
        existing = session.query(Feedback).filter(Feedback.prediction_id == prediction_id).first()
        if existing:
            raise ValueError(f"Feedback already exists for prediction {prediction_id}")
        
        feedback = Feedback(
            prediction_id=prediction_id,
            corrected_label=corrected_label,
            user_id=user_id,
            feedback_source=feedback_source
        )
        session.add(feedback)
        session.flush()
        return feedback
    
    @staticmethod
    def get_recent(session: Session, limit: int = 100) -> List[Feedback]:
        """Get recent feedback"""
        return session.query(Feedback).order_by(desc(Feedback.created_at)).limit(limit).all()

class AuditLogRepository:
    """Data access methods for AuditLog table"""
    
    @staticmethod
    def create(
        session: Session,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        error_message: Optional[str] = None,
        request_payload: Optional[Dict] = None
    ) -> AuditLog:
        """Log API request"""
        audit_log = AuditLog(
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            user_id=user_id,
            ip_address=ip_address,
            error_message=error_message,
            request_payload=request_payload
        )
        session.add(audit_log)
        session.flush()
        return audit_log


class InferenceMetadataRepository:
    """Data access methods for daily inference metadata aggregation."""

    @staticmethod
    def upsert_prediction_aggregate(
        session: Session,
        prediction_date: date,
        predicted_label: str,
        confidence: float
    ) -> InferenceMetadata:
        """Upsert a daily aggregate row for a new prediction."""
        metadata = (
            session.query(InferenceMetadata)
            .filter(InferenceMetadata.date == prediction_date)
            .first()
        )

        if metadata is None:
            metadata = InferenceMetadata(
                date=prediction_date,
                total_predictions=0,
                category_distribution={},
                avg_confidence=0.0,
                low_confidence_count=0
            )
            session.add(metadata)
            session.flush()

        current_total = metadata.total_predictions or 0
        new_total = current_total + 1
        current_avg = metadata.avg_confidence or 0.0

        metadata.total_predictions = new_total
        metadata.avg_confidence = ((current_avg * current_total) + confidence) / new_total
        metadata.low_confidence_count = (metadata.low_confidence_count or 0) + (1 if confidence < 0.5 else 0)

        distribution = dict(metadata.category_distribution or {})
        distribution[predicted_label] = int(distribution.get(predicted_label, 0)) + 1
        metadata.category_distribution = distribution
        metadata.created_at = datetime.utcnow()

        session.flush()
        return metadata

