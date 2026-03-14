#pipeline tests for preprocessing

"""
Test suite for preprocessing and feature extraction pipeline.

This module tests:
- Text preprocessing functionality
- URL and email removal
- Batch processing
- Feature extraction
- Preprocessing consistency
- Save/load functionality
- Configuration management
- Edge cases

Run with: python -m pytest tests/test_preprocessing.py -v
Or: python tests/test_preprocessing.py
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_pipeline import (
    TextPreprocessor,
    EmailFeatureExtractor,
    PipelineConfig,
    ArtifactManager,
    get_default_config
)

class TestTextPreprocessor(unittest.TestCase):
    """Test TextPreprocessor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
            remove_urls=True,
            remove_emails=True,
            remove_stopwords=True,
            apply_lemmatization=True
        )
    
    def test_basic_preprocessing(self):
        """Test basic text cleaning."""
        text = "Hello World! This is a TEST."
        result = self.preprocessor.clean_text(text)
        
        # Should be lowercase, no punctuation
        self.assertNotIn('!', result)
        self.assertNotIn('.', result)
        self.assertEqual(result.lower(), result)
    
    def test_url_removal(self):
        """Test URL removal."""
        text = "Check this link: https://example.com for more info"
        result = self.preprocessor.clean_text(text)
        
        # URL should be removed
        self.assertNotIn('https', result)
        self.assertNotIn('example.com', result)
    
    def test_email_removal(self):
        """Test email address removal."""
        text = "Contact us at support@example.com for help"
        result = self.preprocessor.clean_text(text)
        
        # Email should be removed
        self.assertNotIn('support@example.com', result)
        self.assertNotIn('@', result)
    
    def test_stopword_removal(self):
        """Test stopword filtering."""
        text = "this is a test of the system"
        result = self.preprocessor.clean_text(text)
        
        # Common stopwords should be removed
        self.assertNotIn('is', result)
        self.assertNotIn('a', result)
        self.assertNotIn('the', result)
    
    def test_batch_processing(self):
        """Test batch text processing."""
        texts = [
            "First email about meetings",
            "Second email about projects",
            "Third email about reports"
        ]
        results = self.preprocessor.clean_batch(texts)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, str)
            self.assertEqual(result.lower(), result)  # All lowercase
    
    def test_empty_text(self):
        """Test handling of empty/None text."""
        self.assertEqual(self.preprocessor.clean_text(""), "")
        self.assertEqual(self.preprocessor.clean_text(None), "")
    
    def test_get_config(self):
        """Test configuration retrieval."""
        config = self.preprocessor.get_config()
        
        self.assertIsInstance(config, dict)
        self.assertTrue(config['lowercase'])
        self.assertTrue(config['remove_punctuation'])
        self.assertTrue(config['remove_urls'])

class TestEmailFeatureExtractor(unittest.TestCase):
    """Test EmailFeatureExtractor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = [
            "urgent meeting scheduled for tomorrow morning",
            "project update needed by end of week",
            "please review the attached document",
            "meeting reminder for team discussion",
            "project status report required"
        ]
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_fit_transform(self):
        """Test fitting and transforming texts."""
        extractor = EmailFeatureExtractor(max_features=100)
        features = extractor.fit_transform(self.sample_texts)
        
        # Check shape
        self.assertEqual(features.shape[0], len(self.sample_texts))
        self.assertLessEqual(features.shape[1], 100)
        
        # Check values are numeric
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_fit_then_transform(self):
        """Test separate fit and transform calls."""
        extractor = EmailFeatureExtractor(max_features=100)
        
        # Fit on training data
        extractor.fit(self.sample_texts)
        
        # Transform new data
        new_texts = ["urgent project meeting tomorrow"]
        features = extractor.transform(new_texts)
        
        self.assertEqual(features.shape[0], 1)
    
    def test_transform_without_fit(self):
        """Test that transform fails without fit."""
        extractor = EmailFeatureExtractor()
        
        with self.assertRaises(ValueError):
            extractor.transform(["test text"])
    
    def test_get_feature_names(self):
        """Test feature name retrieval."""
        extractor = EmailFeatureExtractor(max_features=50)
        extractor.fit(self.sample_texts)
        
        feature_names = extractor.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        self.assertLessEqual(len(feature_names), 50)
    
    def test_pruning_error_message(self):
        """Verify informative error when pruning removes all terms."""
        # Create two totally disjoint documents so min_df=2 leaves no terms
        extractor = EmailFeatureExtractor(min_df=2)
        with self.assertRaises(ValueError) as cm:
            extractor.fit_transform(["foo bar baz", "qux quux corge"])
        msg = str(cm.exception)
        # message may come from sklearn or our custom catch, accept either
        self.assertTrue(
            "No terms remain after pruning" in msg
            or "max_df corresponds to < documents than min_df" in msg,
            f"unexpected error message: {msg}"
        )
    
    def test_vocabulary_size(self):
        """Test vocabulary size retrieval."""
        extractor = EmailFeatureExtractor(max_features=100)
        extractor.fit(self.sample_texts)
        
        vocab_size = extractor.get_vocabulary_size()
        
        self.assertGreater(vocab_size, 0)
        self.assertLessEqual(vocab_size, 100)
    
    def test_save_load(self):
        """Test saving and loading extractor."""
        # Train extractor
        extractor = EmailFeatureExtractor(max_features=100)
        features_before = extractor.fit_transform(self.sample_texts)
        
        # Save
        save_path = os.path.join(self.temp_dir, "test_extractor.pkl")
        extractor.save(save_path)
        
        # Load
        loaded_extractor = EmailFeatureExtractor.load(save_path)
        
        # Transform with loaded extractor
        features_after = loaded_extractor.transform(self.sample_texts)
        
        # Features should be identical
        np.testing.assert_array_almost_equal(features_before, features_after)
    
    def test_preprocessing_consistency(self):
        """Test that preprocessing is consistent across fit and transform."""
        # allow min_df=1 so a single document can be used safely
        extractor = EmailFeatureExtractor(max_features=100, min_df=1, max_df=1.0)
        
        # Fit on lowercase text
        train_texts = ["URGENT MEETING TOMORROW"]
        extractor.fit(train_texts)
        
        # Transform with different case
        test_texts = ["urgent meeting tomorrow"]
        features = extractor.transform(test_texts)
        
        # Should produce valid features (preprocessing handles case)
        self.assertEqual(features.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(features)))

class TestPipelineConfig(unittest.TestCase):
    """Test PipelineConfig functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = get_default_config()
        
        self.assertEqual(config.version, "1.0")
        self.assertIsInstance(config.preprocessing.to_dict(), dict)
        self.assertIsInstance(config.feature_extraction.to_dict(), dict)
    
    def test_save_load_config(self):
        """Test saving and loading configuration."""
        config = get_default_config()
        
        # Save
        config_path = os.path.join(self.temp_dir, "test_config.json")
        config.save(config_path)
        
        # Load
        loaded_config = PipelineConfig.load(config_path)
        
        # Should be identical
        self.assertEqual(config.version, loaded_config.version)
        self.assertEqual(
            config.preprocessing.to_dict(), 
            loaded_config.preprocessing.to_dict()
        )
        self.assertEqual(
            config.feature_extraction.to_dict(),
            loaded_config.feature_extraction.to_dict()
        )
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = get_default_config()
        config_dict = config.to_dict()
        
        self.assertIn('version', config_dict)
        self.assertIn('preprocessing', config_dict)
        self.assertIn('feature_extraction', config_dict)
        self.assertIn('created_at', config_dict)

class TestArtifactManager(unittest.TestCase):
    """Test ArtifactManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ArtifactManager(base_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_directory_creation(self):
        """Test automatic directory creation."""
        self.assertTrue(os.path.exists(self.manager.models_dir))
        self.assertTrue(os.path.exists(self.manager.vectorizers_dir))
        self.assertTrue(os.path.exists(self.manager.configs_dir))
    
    def test_get_paths(self):
        """Test path generation."""
        model_path = self.manager.get_model_path("1.0")
        vectorizer_path = self.manager.get_vectorizer_path("1.0")
        config_path = self.manager.get_config_path("1.0")
        
        self.assertIn("model_v1.0.pkl", model_path)
        self.assertIn("vectorizer_v1.0.pkl", vectorizer_path)
        self.assertIn("config_v1.0.json", config_path)
    
    def test_list_artifacts(self):
        """Test artifact listing."""
        # Create some dummy files
        open(self.manager.get_model_path("1.0"), 'w').close()
        open(self.manager.get_model_path("1.1"), 'w').close()
        
        models = self.manager.list_models()
        
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0][0], "1.0")
        self.assertEqual(models[1][0], "1.1")
    
    def test_get_latest(self):
        """Test getting latest version."""
        # Create dummy files
        open(self.manager.get_model_path("1.0"), 'w').close()
        open(self.manager.get_model_path("1.1"), 'w').close()
        open(self.manager.get_model_path("1.2"), 'w').close()
        
        latest = self.manager.get_latest_model()
        
        self.assertIn("1.2", latest)
    
    def test_generate_version(self):
        """Test version generation."""
        version = self.manager.generate_version()
        
        self.assertIsInstance(version, str)
        self.assertGreater(len(version), 0)
        
        # Test with prefix
        version_with_prefix = self.manager.generate_version("prod")
        self.assertTrue(version_with_prefix.startswith("prod_"))
    
    def test_cleanup_old_versions(self):
        """Test cleaning up old versions."""
        # Create multiple model versions
        for i in range(5):
            open(self.manager.get_model_path(f"1.{i}"), 'w').close()
        
        # Keep only latest 2
        deleted = self.manager.cleanup_old_versions("models", keep_latest=2)
        
        self.assertEqual(deleted, 3)
        
        # Check only 2 remain
        models = self.manager.list_models()
        self.assertEqual(len(models), 2)
    
    def test_get_artifact_info(self):
        """Test artifact information retrieval."""
        # Create some artifacts
        open(self.manager.get_model_path("1.0"), 'w').close()
        open(self.manager.get_vectorizer_path("1.0"), 'w').close()
        open(self.manager.get_config_path("1.0"), 'w').close()
        
        info = self.manager.get_artifact_info()
        
        self.assertEqual(info['models']['count'], 1)
        self.assertEqual(info['vectorizers']['count'], 1)
        self.assertEqual(info['configs']['count'], 1)
        self.assertEqual(info['models']['latest'], "1.0")

class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_texts = [
            "urgent meeting scheduled for tomorrow morning at 10am",
            "project update needed by end of this week",
            "please review the attached quarterly report document",
            "meeting reminder for team discussion on new features",
            "project status report required for management review"
        ]
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_pipeline_workflow(self):
        """Test complete training and inference workflow."""
        # 1. Create configuration
        config = get_default_config()
        config_path = os.path.join(self.temp_dir, "config_v1.0.json")
        config.save(config_path)
        
        # 2. Train feature extractor
        extractor = EmailFeatureExtractor(
            max_features=config.feature_extraction.max_features,
            ngram_range=config.feature_extraction.ngram_range
        )
        train_features = extractor.fit_transform(self.sample_texts)
        
        # 3. Save extractor
        extractor_path = os.path.join(self.temp_dir, "vectorizer_v1.0.pkl")
        extractor.save(extractor_path)
        
        # 4. Load configuration and extractor
        loaded_config = PipelineConfig.load(config_path)
        loaded_extractor = EmailFeatureExtractor.load(extractor_path)
        
        # 5. Transform new data
        new_texts = ["urgent project meeting tomorrow"]
        inference_features = loaded_extractor.transform(new_texts)
        
        # Verify
        self.assertEqual(train_features.shape[1], inference_features.shape[1])
        self.assertTrue(np.all(np.isfinite(inference_features)))
        self.assertEqual(loaded_config.version, config.version)
    
    def test_consistency_across_sessions(self):
        """Test that features are consistent across save/load."""
        # Train and save
        extractor1 = EmailFeatureExtractor(max_features=100)
        features1 = extractor1.fit_transform(self.sample_texts)
        
        extractor_path = os.path.join(self.temp_dir, "extractor.pkl")
        extractor1.save(extractor_path)
        
        # Load and transform
        extractor2 = EmailFeatureExtractor.load(extractor_path)
        features2 = extractor2.transform(self.sample_texts)
        
        # Features should be identical
        np.testing.assert_array_almost_equal(features1, features2, decimal=6)

def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTextPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestEmailFeatureExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestArtifactManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
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
