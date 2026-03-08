"""
Database package for Email Categorization System (Neon Cloud)
"""
from .connection import get_db_session, init_database, close_database
from .models import Email, Prediction, Feedback, ModelVersion, AuditLog, InferenceMetadata, RetrainingJob

__all__ = [
    'get_db_session',
    'init_database',
    'close_database',
    'Email',
    'Prediction',
    'Feedback',
    'ModelVersion',
    'AuditLog',
    'InferenceMetadata',
    'RetrainingJob'
]