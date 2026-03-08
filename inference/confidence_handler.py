# Manage confidence thresholds and uncertainty.

"""
Confidence threshold handling and uncertainty management.

Provides strategies for handling low-confidence predictions.
"""

from typing import Dict, Any, Optional, List
from enum import Enum


class ConfidenceStrategy(Enum):
    """Strategies for handling low-confidence predictions."""
    REJECT = "reject"              # Reject prediction, return 'unknown'
    FLAG = "flag"                  # Accept but flag for review
    SECOND_BEST = "second_best"    # Consider second-best prediction
    HUMAN_REVIEW = "human_review"  # Queue for human review


class ConfidenceHandler:
    """
    Handle confidence thresholds and strategies.
    
    Example:
        >>> handler = ConfidenceHandler(threshold=0.7)
        >>> result = {
        ...     'category': 'incident',
        ...     'confidence': 0.65,
        ...     'all_probabilities': {'incident': 0.65, 'meeting': 0.30, ...}
        ... }
        >>> handled = handler.apply_strategy(result, ConfidenceStrategy.FLAG)
        >>> print(handled['needs_review'])
        True
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        rejection_threshold: float = 0.5
    ):
        """
        Initialize confidence handler.
        
        Args:
            threshold: Confidence threshold for flagging (default: 0.7)
            rejection_threshold: Minimum confidence to accept (default: 0.5)
        """
        self.threshold = threshold
        self.rejection_threshold = rejection_threshold
    
    def apply_strategy(
        self,
        prediction: Dict[str, Any],
        strategy: ConfidenceStrategy = ConfidenceStrategy.FLAG
    ) -> Dict[str, Any]:
        """
        Apply confidence handling strategy.
        
        Args:
            prediction: Prediction dictionary from categorizer
            strategy: Strategy to apply
            
        Returns:
            Modified prediction with strategy applied
        """
        confidence = prediction.get('confidence', 0.0)
        result = prediction.copy()
        
        # Check if below rejection threshold
        if confidence < self.rejection_threshold:
            if strategy == ConfidenceStrategy.REJECT:
                result['category'] = 'unknown'
                result['confidence'] = 0.0
                result['rejection_reason'] = f'Confidence {confidence:.2%} below threshold {self.rejection_threshold:.2%}'
                return result
        
        # Check if below flag threshold
        if confidence < self.threshold:
            if strategy == ConfidenceStrategy.FLAG:
                result['needs_review'] = True
                result['flag_reason'] = f'Low confidence: {confidence:.2%}'
            
            elif strategy == ConfidenceStrategy.SECOND_BEST:
                # Consider second-best prediction
                all_probs = prediction.get('all_probabilities', {})
                if all_probs:
                    sorted_probs = sorted(
                        all_probs.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    if len(sorted_probs) >= 2:
                        second_best = sorted_probs[1]
                        result['alternative_category'] = second_best[0]
                        result['alternative_confidence'] = second_best[1]
                        result['needs_review'] = True
            
            elif strategy == ConfidenceStrategy.HUMAN_REVIEW:
                result['queue_for_review'] = True
                result['review_priority'] = self._calculate_priority(confidence)
        
        return result
    
    def _calculate_priority(self, confidence: float) -> str:
        """Calculate review priority based on confidence."""
        if confidence < 0.3:
            return 'high'
        elif confidence < 0.5:
            return 'medium'
        else:
            return 'low'
    
    def get_confidence_category(self, confidence: float) -> str:
        """
        Categorize confidence level.
        
        Args:
            confidence: Confidence score (0.0-1.0)
            
        Returns:
            Category: 'very_high', 'high', 'medium', 'low', 'very_low'
        """
        if confidence >= 0.9:
            return 'very_high'
        elif confidence >= 0.7:
            return 'high'
        elif confidence >= 0.5:
            return 'medium'
        elif confidence >= 0.3:
            return 'low'
        else:
            return 'very_low'
    
    def should_accept_prediction(self, confidence: float) -> bool:
        """
        Determine if prediction should be accepted.
        
        Args:
            confidence: Confidence score
            
        Returns:
            True if should accept, False otherwise
        """
        return confidence >= self.rejection_threshold
    
    def analyze_prediction_quality(
        self,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze prediction quality with recommendations.
        
        Args:
            prediction: Prediction dictionary
            
        Returns:
            Analysis dictionary with recommendations
        """
        confidence = prediction.get('confidence', 0.0)
        all_probs = prediction.get('all_probabilities', {})
        
        analysis = {
            'confidence': confidence,
            'confidence_category': self.get_confidence_category(confidence),
            'should_accept': self.should_accept_prediction(confidence),
            'needs_review': confidence < self.threshold
        }
        
        # Analyze probability distribution
        if all_probs:
            sorted_probs = sorted(all_probs.values(), reverse=True)
            
            if len(sorted_probs) >= 2:
                # Gap between top 2
                gap = sorted_probs[0] - sorted_probs[1]
                analysis['top_2_gap'] = gap
                
                if gap < 0.1:
                    analysis['warning'] = 'Very close competition between top categories'
                elif gap < 0.2:
                    analysis['warning'] = 'Close competition between top categories'
                
                # Entropy (uncertainty measure)
                import numpy as np
                entropy = -sum(p * np.log(p + 1e-10) for p in all_probs.values())
                analysis['entropy'] = entropy
                
                if entropy > 1.5:
                    analysis['warning'] = 'High uncertainty across multiple categories'
        
        # Recommendations
        if confidence < self.rejection_threshold:
            analysis['recommendation'] = 'Reject prediction'
        elif confidence < self.threshold:
            analysis['recommendation'] = 'Accept with human review'
        else:
            analysis['recommendation'] = 'Accept with confidence'
        
        return analysis
