"""
Error handling utilities for inference service.
"""

import logging
from typing import Dict, Any, Optional
from functools import wraps


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('inference')


def handle_prediction_errors(func):
    """Decorator for handling prediction errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error in {func.__name__}: {e}")
            return {
                'category': 'error',
                'confidence': 0.0,
                'error': f'Validation error: {str(e)}'
            }
        except FileNotFoundError as e:
            logger.error(f"File error in {func.__name__}: {e}")
            return {
                'category': 'error',
                'confidence': 0.0,
                'error': f'Model file error: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return {
                'category': 'error',
                'confidence': 0.0,
                'error': f'Unexpected error: {str(e)}'
            }
    return wrapper


class InferenceErrorHandler:
    """Handle inference errors with recovery strategies."""
    
    @staticmethod
    def handle_model_load_error(error: Exception) -> Dict[str, str]:
        """Handle model loading errors."""
        if isinstance(error, FileNotFoundError):
            return {
                'error_type': 'model_not_found',
                'message': 'Model file not found. Please train a model first.',
                'recovery': 'Run: python train_model.py --data data/sample_emails.csv --version 1.0 --set-active'
            }
        elif isinstance(error, ValueError):
            return {
                'error_type': 'no_active_model',
                'message': 'No active model in database.',
                'recovery': 'Set a model as active or train a new one with --set-active flag'
            }
        else:
            return {
                'error_type': 'unknown_model_error',
                'message': str(error),
                'recovery': 'Check model files and database connection'
            }
    
    @staticmethod
    def handle_prediction_error(error: Exception, email_text: str) -> Dict[str, Any]:
        """Handle prediction errors."""
        logger.error(f"Prediction error for text: {email_text[:50]}... Error: {error}")
        
        return {
            'category': 'error',
            'confidence': 0.0,
            'error': str(error),
            'error_type': type(error).__name__,
            'recovery': 'Check input text and model status'
        }

