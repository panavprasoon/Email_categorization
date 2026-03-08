"""
Test suite for training service (Step 3).

Tests:
- Data loading and validation
- Model training
- Model evaluation
- Database registration
- Complete pipeline
"""

import os
import sys
import unittest
import tempfile
import shutil
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training import EmailDataLoader, EmailClassifierTrainer, ModelEvaluator, ModelRegistry
from feature_pipeline import get_default_config

class TestEmailDataLoader(unittest.TestCase):
    """Test EmailDataLoader functionality."""
    
    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.temp_dir, 'test_data.csv')
        
        # Create sample CSV
        data = {
            'email_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'text': [
                'urgent meeting tomorrow',
                'project update needed',
                'please review document',
                'spam advertisement',
                'system down critical',
                'weekly team sync',
                'report attached',
                'get rich quick',
                'server maintenance',
                'budget approval'
            ],
            'category': [
                'meeting', 'update', 'report', 'spam', 'incident',
                'meeting', 'report', 'spam', 'incident', 'approval'
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.test_csv, index=False)
        
        self.loader = EmailDataLoader()
    
    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_csv(self):
        """Test CSV loading."""
        df = self.loader.load_csv(self.test_csv)
        self.assertEqual(len(df), 10)
        self.assertIn('text', df.columns)
        self.assertIn('category', df.columns)
    
    def test_validate_data(self):
        """Test data validation."""
        df = self.loader.load_csv(self.test_csv)
        validation = self.loader.validate_data(df)
        
        self.assertEqual(validation['total_records'], 10)
        self.assertEqual(validation['missing_text'], 0)
        self.assertIn('class_distribution', validation)
    
    def test_split_data(self):
        """Test data splitting."""
        df = self.loader.load_csv(self.test_csv)
        splits = self.loader.split_data(df, test_size=0.2, val_size=0.1)
        
        self.assertIn('train', splits)
        self.assertIn('val', splits)
        self.assertIn('test', splits)
        
        X_train, y_train = splits['train']
        X_test, y_test = splits['test']
        
        # Check total samples preserved
        total = len(X_train) + len(splits['val'][0]) + len(X_test)
        self.assertEqual(total, 10)

class TestEmailClassifierTrainer(unittest.TestCase):
    """Test EmailClassifierTrainer functionality."""
    
    def setUp(self):
        """Set up test data and trainer."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.train_texts = [
            'urgent meeting tomorrow',
            'project status update',
            'spam advertisement',
            'critical system down',
            'team meeting friday',
            'please update project',
            'click here to win',
            'server outage detected'
        ]
        self.train_labels = [
            'meeting', 'update', 'spam', 'incident',
            'meeting', 'update', 'spam', 'incident'
        ]
        
        self.config = get_default_config()
        self.trainer = EmailClassifierTrainer(pipeline_config=self.config)
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_train_logistic_regression(self):
        """Test training logistic regression."""
        results = self.trainer.train(
            train_texts=self.train_texts,
            train_labels=self.train_labels,
            model_type='logistic_regression',
            tune_hyperparams=False
        )
        
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.feature_extractor)
        self.assertIn('train_accuracy', results)
    
    def test_predict(self):
        """Test prediction."""
        self.trainer.train(
            train_texts=self.train_texts,
            train_labels=self.train_labels,
            model_type='logistic_regression',
            tune_hyperparams=False
        )
        
        test_texts = ['urgent meeting', 'spam offer']
        predictions = self.trainer.predict(test_texts)
        
        self.assertEqual(len(predictions), 2)
    
    def test_save_load_model(self):
        """Test model save/load."""
        self.trainer.train(
            train_texts=self.train_texts,
            train_labels=self.train_labels,
            model_type='logistic_regression',
            tune_hyperparams=False
        )
        
        model_path = os.path.join(self.temp_dir, 'test_model.pkl')
        self.trainer.save_model(model_path)
        
        # Load model
        loaded_trainer = EmailClassifierTrainer.load_model(model_path)
        
        # Predict with loaded model
        test_texts = ['urgent meeting']
        predictions = loaded_trainer.predict(test_texts)
        
        self.assertEqual(len(predictions), 1)

class TestModelEvaluator(unittest.TestCase):
    """Test ModelEvaluator functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.y_true = np.array(['meeting', 'spam', 'meeting', 'incident', 'spam'])
        self.y_pred = np.array(['meeting', 'meeting', 'meeting', 'incident', 'spam'])
        self.evaluator = ModelEvaluator()
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_evaluate(self):
        """Test metric calculation."""
        metrics = self.evaluator.evaluate(self.y_true, self.y_pred)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
    
    def test_confusion_matrix(self):
        """Test confusion matrix generation."""
        cm, classes = self.evaluator.get_confusion_matrix(self.y_true, self.y_pred)
        
        self.assertEqual(cm.shape[0], len(classes))
        self.assertEqual(cm.shape[1], len(classes))
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        save_path = os.path.join(self.temp_dir, 'cm.png')
        
        self.evaluator.plot_confusion_matrix(
            self.y_true,
            self.y_pred,
            save_path=save_path
        )
        
        self.assertTrue(os.path.exists(save_path))

def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestEmailDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestEmailClassifierTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelEvaluator))
    
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
