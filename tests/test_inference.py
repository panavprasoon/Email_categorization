# Comprehensive tests for inference service.

"""
Test suite for inference service (Step 4).

Tests:
- Email categorization
- Prediction storage
- Batch processing
- Confidence handling
- Error handling
"""

import os
import sys
import unittest
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference import EmailCategorizer, PredictionStore, BatchProcessor
from inference.confidence_handler import ConfidenceHandler, ConfidenceStrategy
from training import EmailClassifierTrainer, ModelRegistry
from feature_pipeline import get_default_config


class TestEmailCategorizer(unittest.TestCase):
    """Test EmailCategorizer functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test model (runs once for all tests)."""
        # Check if active model exists
        registry = ModelRegistry()
        active_model = registry.get_active_model()
        
        # If an active model exists, ensure the file still exists.
        # If it does not, retrain and register a fresh active model.
        needs_training = False
        if not active_model:
            needs_training = True
        else:
            active_path = getattr(active_model, 'model_path', None)
            if active_path and not os.path.exists(active_path):
                print(f"Active model file not found at {active_path}. Re-training test model...")
                needs_training = True

        if needs_training:
            print("\nNo active model found or model file missing. Training test model...")

            # Train a quick test model
            train_texts = [
                'urgent meeting tomorrow',
                'project status update',
                'spam advertisement',
                'critical system down',
                'team meeting friday',
                'please update project',
                'click here to win',
                'server outage detected'
            ]
            train_labels = [
                'meeting', 'update', 'spam', 'incident',
                'meeting', 'update', 'spam', 'incident'
            ]

            config = get_default_config()
            trainer = EmailClassifierTrainer(pipeline_config=config)
            trainer.train(
                train_texts=train_texts,
                train_labels=train_labels,
                model_type='logistic_regression',
                tune_hyperparams=False
            )

            # Save and register
            temp_dir = tempfile.gettempdir()
            model_path = os.path.join(temp_dir, 'test_model_inference.pkl')
            trainer.save_model(model_path)

            # Register as active
            registry.register_model(
                version='test_inference',
                model_type='logistic_regression',
                metrics={'accuracy': 0.9, 'f1_score': 0.89},
                model_path=model_path,
                vectorizer_path=model_path,  # Same file contains both
                set_active=True,
                description='Test model for inference tests'
            )

            print("✓ Test model created and set as active")
    
    def setUp(self):
        """Set up test fixtures."""
        self.categorizer = EmailCategorizer()
    
    def test_predict_single_email(self):
        """Test single email prediction."""
        result = self.categorizer.predict("urgent meeting tomorrow")
        
        self.assertIn('category', result)
        self.assertIn('confidence', result)
        self.assertIn('model_version', result)
        
        self.assertIsInstance(result['category'], str)
        self.assertIsInstance(result['confidence'], float)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_predict_with_details(self):
        """Test prediction with detailed information."""
        result = self.categorizer.predict_with_details("server down critical")
        
        self.assertIn('category', result)
        self.assertIn('confidence', result)
        self.assertIn('all_probabilities', result)
        self.assertIn('top_3_categories', result)
        self.assertIn('prediction_time_ms', result)
        
        # Check all_probabilities
        all_probs = result['all_probabilities']
        self.assertIsInstance(all_probs, dict)
        self.assertGreater(len(all_probs), 0)
        
        # Check top_3
        top_3 = result['top_3_categories']
        self.assertEqual(len(top_3), min(3, len(all_probs)))
    
    def test_predict_empty_input(self):
        """Test prediction with empty input."""
        result = self.categorizer.predict("")
        
        self.assertEqual(result['category'], 'unknown')
        self.assertEqual(result['confidence'], 0.0)
        self.assertIn('error', result)
    
    def test_predict_batch(self):
        """Test batch prediction."""
        emails = [
            "urgent meeting",
            "spam offer",
            "server down"
        ]
        results = self.categorizer.predict_batch(emails)
        
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIn('category', result)
            self.assertIn('confidence', result)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        # Make some predictions
        self.categorizer.predict("test email 1")
        self.categorizer.predict("test email 2")
        
        stats = self.categorizer.get_statistics()
        
        self.assertIn('predictions_made', stats)
        self.assertGreaterEqual(stats['predictions_made'], 2)
        self.assertIn('average_inference_time_ms', stats)


class TestPredictionStore(unittest.TestCase):
    """Test PredictionStore functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.store = PredictionStore()
        
        # Create a test email in database
        from database.repository import EmailRepository
        email_repo = EmailRepository()
        self.test_email = email_repo.create_email(
            subject="Test Email",
            body="Test body",
            sender="test@example.com"
        )
    
    def test_save_prediction(self):
        """Test saving prediction."""
        # Get active model
        registry = ModelRegistry()
        active_model = registry.get_active_model()
        
        prediction_id = self.store.save_prediction(
            email_id=self.test_email.id,
            category='meeting',
            confidence=0.89,
            model_id=active_model.id
        )
        
        self.assertIsInstance(prediction_id, int)
        self.assertGreater(prediction_id, 0)
    
    def test_get_prediction(self):
        """Test retrieving prediction."""
        # Save a prediction first
        registry = ModelRegistry()
        active_model = registry.get_active_model()
        
        prediction_id = self.store.save_prediction(
            email_id=self.test_email.id,
            category='meeting',
            confidence=0.89,
            model_id=active_model.id
        )
        
        # Retrieve it
        prediction = self.store.get_prediction(prediction_id)
        
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.predicted_category, 'meeting')
        self.assertEqual(prediction.confidence_score, 0.89)


class TestBatchProcessor(unittest.TestCase):
    """Test BatchProcessor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = BatchProcessor(batch_size=2)
    
    def test_process_emails(self):
        """Test batch email processing."""
        emails = [
            {'id': 1, 'text': 'urgent meeting'},
            {'id': 2, 'text': 'spam offer'},
            {'id': 3, 'text': 'server down'}
        ]
        
        results = self.processor.process_emails(emails, save_to_db=False)
        
        self.assertEqual(len(results), 3)
        
        for i, result in enumerate(results):
            self.assertEqual(result['email_id'], emails[i]['id'])
            self.assertIn('category', result)
            self.assertIn('confidence', result)
    
    def test_get_summary(self):
        """Test summary statistics."""
        emails = [
            {'id': 1, 'text': 'urgent meeting'},
            {'id': 2, 'text': 'spam offer'}
        ]
        
        results = self.processor.process_emails(emails, save_to_db=False)
        summary = self.processor.get_summary(results)
        
        self.assertIn('total_emails', summary)
        self.assertEqual(summary['total_emails'], 2)
        self.assertIn('category_distribution', summary)
        self.assertIn('average_confidence', summary)


class TestConfidenceHandler(unittest.TestCase):
    """Test ConfidenceHandler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = ConfidenceHandler(threshold=0.7)
    
    def test_apply_flag_strategy(self):
        """Test FLAG strategy."""
        prediction = {
            'category': 'meeting',
            'confidence': 0.65
        }
        
        result = self.handler.apply_strategy(
            prediction,
            ConfidenceStrategy.FLAG
        )
        
        self.assertTrue(result['needs_review'])
        self.assertIn('flag_reason', result)
    
    def test_apply_reject_strategy(self):
        """Test REJECT strategy."""
        prediction = {
            'category': 'meeting',
            'confidence': 0.45
        }
        
        result = self.handler.apply_strategy(
            prediction,
            ConfidenceStrategy.REJECT
        )
        
        self.assertEqual(result['category'], 'unknown')
        self.assertIn('rejection_reason', result)
    
    def test_get_confidence_category(self):
        """Test confidence categorization."""
        self.assertEqual(
            self.handler.get_confidence_category(0.95),
            'very_high'
        )
        self.assertEqual(
            self.handler.get_confidence_category(0.75),
            'high'
        )
        self.assertEqual(
            self.handler.get_confidence_category(0.55),
            'medium'
        )
        self.assertEqual(
            self.handler.get_confidence_category(0.35),
            'low'
        )
    
    def test_should_accept_prediction(self):
        """Test acceptance threshold."""
        self.assertTrue(self.handler.should_accept_prediction(0.6))
        self.assertFalse(self.handler.should_accept_prediction(0.4))


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestEmailCategorizer))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionStore))
    suite.addTests(loader.loadTestsFromTestCase(TestBatchProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestConfidenceHandler))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)