"""
Inference Service Package

Provides email categorization inference functionality.
"""

from .categorizer import EmailCategorizer
from .prediction_store import PredictionStore
from .batch_processor import BatchProcessor
from .confidence_handler import ConfidenceHandler, ConfidenceStrategy

__all__ = ['EmailCategorizer',
            'PredictionStore', 
            'BatchProcessor',
            'ConfidenceHandler',
            'ConfidenceStrategy']
