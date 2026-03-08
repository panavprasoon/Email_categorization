"""
Input validation for inference service.
"""

import re
from typing import Dict, Any, List


class EmailValidator:
    """Validate email input before categorization."""
    
    @staticmethod
    def validate_email_text(text: str) -> Dict[str, Any]:
        """
        Validate email text input.
        
        Args:
            text: Email text to validate
            
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str]
            }
        """
        errors = []
        warnings = []
        
        # Check if text exists
        if not text:
            errors.append("Empty text")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check type
        if not isinstance(text, str):
            errors.append(f"Invalid type: {type(text)}, expected str")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check length
        text_stripped = text.strip()
        if not text_stripped:
            errors.append("Text is only whitespace")
        elif len(text_stripped) < 3:
            warnings.append("Text is very short (< 3 characters)")
        elif len(text_stripped) > 10000:
            warnings.append("Text is very long (> 10,000 characters)")
        
        # Check for common issues
        if text_stripped.lower() == 'test':
            warnings.append("Text appears to be test data")
        
        # Check for excessive repeated characters
        if re.search(r'(.)\1{10,}', text):
            warnings.append("Contains excessive repeated characters")
        
        # Check for non-ASCII characters
        if not text.isascii():
            warnings.append("Contains non-ASCII characters")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    @staticmethod
    def sanitize_email_text(text: str, max_length: int = 10000) -> str:
        """
        Sanitize email text.
        
        Args:
            text: Email text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Strip whitespace
        text = text.strip()
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        return text
