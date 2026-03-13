"""
SQLAlchemy ORM models for Email Categorization System
Maps Python classes to Neon PostgreSQL tables
"""
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, 
    DateTime, Date, ForeignKey, CheckConstraint, JSON
)
from sqlalchemy.dialects.postgresql import JSONB, UUID, INET
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

# Base class for all models
Base = declarative_base()

class Email(Base):
    """Raw email text storage"""
    __tablename__ = 'emails'
    
    id = Column(Integer, primary_key=True, index=True)
    email_text = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    predictions = relationship("Prediction", back_populates="email", cascade="all, delete-orphan")

    @property
    def sender(self) -> str:
        """Best-effort sender extraction from structured email_text."""
        if not self.email_text:
            return ""

        first_line = self.email_text.splitlines()[0] if self.email_text.splitlines() else ""
        if first_line.startswith("Sender: "):
            return first_line.replace("Sender: ", "", 1).strip()
        return ""

    @property
    def subject(self) -> str:
        """Best-effort subject extraction from structured email_text."""
        if not self.email_text:
            return ""

        lines = self.email_text.splitlines()
        if len(lines) > 1 and lines[1].startswith("Subject: "):
            return lines[1].replace("Subject: ", "", 1).strip()
        return ""

    @property
    def body(self) -> str:
        """Best-effort body extraction from structured email_text."""
        if not self.email_text:
            return ""

        parts = self.email_text.split("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()
        return self.email_text
    
    def __repr__(self):
        return f"<Email(id={self.id}, created_at={self.created_at})>"

class ModelVersion(Base):
    """ML model version registry"""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, nullable=False, index=True)
    accuracy = Column(Float, CheckConstraint('accuracy >= 0 AND accuracy <= 1'))
    precision_score = Column(Float, CheckConstraint('precision_score >= 0 AND precision_score <= 1'))
    recall_score = Column(Float, CheckConstraint('recall_score >= 0 AND recall_score <= 1'))
    f1_score = Column(Float, CheckConstraint('f1_score >= 0 AND f1_score <= 1'))
    is_active = Column(Boolean, default=False, index=True)
    model_path = Column(Text)
    vectorizer_path = Column(Text)
    training_samples = Column(Integer)
    training_metrics = Column(JSONB)
    deployed_at = Column(DateTime)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    predictions = relationship("Prediction", back_populates="model_version")
    
    def __repr__(self):
        return f"<ModelVersion(version={self.version}, active={self.is_active}, f1={self.f1_score})>"

class Prediction(Base):
    """Prediction logs with confidence and version tracking"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, index=True)
    email_id = Column(Integer, ForeignKey('emails.id', ondelete='CASCADE'), nullable=False, index=True)
    model_version_id = Column(Integer, ForeignKey('model_versions.id'), nullable=False, index=True)
    predicted_label = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, CheckConstraint('confidence >= 0 AND confidence <= 1'), nullable=False, index=True)
    prediction_probabilities = Column(JSONB)
    processing_time_ms = Column(Float)
    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)
    
    # Relationships
    email = relationship("Email", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")
    feedback = relationship("Feedback", back_populates="prediction", uselist=False)
    
    @property
    def predicted_category(self) -> str:
        """Alias for predicted_label to support legacy attribute names."""
        return self.predicted_label

    @property
    def confidence_score(self) -> float:
        """Alias for confidence to support legacy attribute names."""
        return self.confidence

    @property
    def probabilities(self) -> Dict[str, Any]:
        """Alias for prediction_probabilities to support legacy attribute names."""
        return self.prediction_probabilities or {}

    def __repr__(self):
        return f"<Prediction(id={self.id}, label={self.predicted_label}, confidence={self.confidence:.2f})>"

class Feedback(Base):
    """User corrections for continuous learning"""
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey('predictions.id', ondelete='CASCADE'), unique=True, nullable=False)
    corrected_label = Column(String(50), nullable=False, index=True)
    user_id = Column(String(100))
    feedback_source = Column(String(50), default='manual')
    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="feedback")
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, corrected_label={self.corrected_label})>"

class AuditLog(Base):
    """API request audit trail"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(UUID(as_uuid=True), server_default=func.uuid_generate_v4())
    endpoint = Column(String(200), nullable=False, index=True)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, index=True)
    latency_ms = Column(Float)
    user_id = Column(String(100))
    ip_address = Column(INET)
    error_message = Column(Text)
    request_payload = Column(JSONB)
    timestamp = Column(DateTime, nullable=False, server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<AuditLog(endpoint={self.endpoint}, status={self.status_code}, latency={self.latency_ms}ms)>"

class InferenceMetadata(Base):
    """Daily aggregated statistics"""
    __tablename__ = 'inference_metadata'
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False, unique=True, server_default=func.current_date())
    total_predictions = Column(Integer, default=0)
    category_distribution = Column(JSONB)
    avg_confidence = Column(Float)
    low_confidence_count = Column(Integer, default=0)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    def __repr__(self):
        return f"<InferenceMetadata(date={self.date}, total={self.total_predictions})>"

class RetrainingJob(Base):
    """Retraining pipeline execution tracking"""
    __tablename__ = 'retraining_jobs'
    
    id = Column(Integer, primary_key=True, index=True)
    trigger_reason = Column(String(200))
    start_time = Column(DateTime, server_default=func.now())
    end_time = Column(DateTime)
    status = Column(String(50), default='running', index=True)
    new_model_version = Column(String(50))
    training_samples = Column(Integer)
    validation_accuracy = Column(Float)
    promoted = Column(Boolean, default=False)
    error_log = Column(Text)
    created_at = Column(DateTime, nullable=False, server_default=func.now(), index=True)
    
    def __repr__(self):
        return f"<RetrainingJob(id={self.id}, status={self.status}, version={self.new_model_version})>"

