# Efficient batch processing for multiple emails.

"""
Batch processing for multiple email categorization.

Optimized for processing many emails efficiently with progress tracking
and error handling.
"""

from typing import List, Dict, Any, Optional, Callable
import time
from .categorizer import EmailCategorizer
from .prediction_store import PredictionStore


class BatchProcessor:
    """
    Process multiple emails efficiently.
    
    Example:
        >>> processor = BatchProcessor()
        >>> emails = [
        ...     {'id': 1, 'text': 'urgent meeting'},
        ...     {'id': 2, 'text': 'spam offer'},
        ...     {'id': 3, 'text': 'server down'}
        ... ]
        >>> results = processor.process_emails(emails, save_to_db=True)
        >>> print(f"Processed {len(results)} emails")
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        categorizer: Optional[EmailCategorizer] = None,
        store: Optional[PredictionStore] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of emails to process per batch
            categorizer: EmailCategorizer instance (creates new if None)
            store: PredictionStore instance (creates new if None)
        """
        self.batch_size = batch_size
        self.categorizer = categorizer or EmailCategorizer()
        self.store = store or PredictionStore()
    
    def process_emails(
        self,
        emails: List[Dict[str, Any]],
        save_to_db: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple emails.
        
        Args:
            emails: List of email dictionaries with 'id' and 'text' keys
            save_to_db: Save predictions to database (default: True)
            progress_callback: Optional callback function(current, total)
            
        Returns:
            List of results with predictions
        
        Example:
            >>> emails = [
            ...     {'id': 1, 'text': 'urgent server down'},
            ...     {'id': 2, 'text': 'team meeting'},
            ... ]
            >>> results = processor.process_emails(emails)
            >>> for result in results:
            ...     print(f"Email {result['email_id']}: {result['category']}")
        """
        results = []
        total = len(emails)
        
        print(f"\nProcessing {total} emails in batches of {self.batch_size}...")
        start_time = time.time()
        
        # Process in batches
        for i in range(0, total, self.batch_size):
            batch = emails[i:i + self.batch_size]
            batch_texts = [email['text'] for email in batch]
            
            # Predict batch
            predictions = self.categorizer.predict_batch(batch_texts)
            
            # Combine with email IDs and save
            for email, prediction in zip(batch, predictions):
                result = {
                    'email_id': email['id'],
                    'category': prediction['category'],
                    'confidence': prediction['confidence'],
                    'model_version': prediction['model_version'],
                    'low_confidence': prediction['low_confidence']
                }
                
                # Save to database if requested
                if save_to_db and prediction['category'] not in ['unknown', 'error']:
                    try:
                        prediction_id = self.store.save_prediction(
                            email_id=email['id'],
                            category=prediction['category'],
                            confidence=prediction['confidence'],
                            model_id=prediction['model_id']
                        )
                        result['prediction_id'] = prediction_id
                    except Exception as e:
                        result['save_error'] = str(e)
                
                results.append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback(min(i + self.batch_size, total), total)
            else:
                print(f"  Processed {min(i + self.batch_size, total)}/{total} emails...")
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.2f} seconds")
        print(f"  Average: {elapsed/total*1000:.2f} ms per email")
        
        return results
    
    def get_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary statistics from batch results.
        
        Args:
            results: List of prediction results
            
        Returns:
            Summary statistics dictionary
        """
        if not results:
            return {}
        
        # Count by category
        category_counts = {}
        low_confidence_count = 0
        error_count = 0
        
        for result in results:
            category = result['category']
            category_counts[category] = category_counts.get(category, 0) + 1
            
            if result.get('low_confidence'):
                low_confidence_count += 1
            
            if category in ['unknown', 'error']:
                error_count += 1
        
        # Calculate average confidence
        valid_results = [
            r for r in results
            if r['category'] not in ['unknown', 'error']
        ]
        avg_confidence = (
            sum(r['confidence'] for r in valid_results) / len(valid_results)
            if valid_results else 0.0
        )
        
        return {
            'total_emails': len(results),
            'category_distribution': category_counts,
            'average_confidence': avg_confidence,
            'low_confidence_count': low_confidence_count,
            'error_count': error_count,
            'success_rate': (len(results) - error_count) / len(results)
        }
