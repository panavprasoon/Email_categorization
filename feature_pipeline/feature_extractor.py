# TF-IDF vectorization

"""
Feature extraction for email categorization using TF-IDF vectorization.

This module provides the EmailFeatureExtractor class which:
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Integrates with TextPreprocessor for consistent preprocessing
- Supports configurable vocabulary size and n-gram ranges
- Provides save/load functionality for trained vectorizers

CRITICAL: The same vectorizer (fitted on training data) must be used
for both training and inference to ensure feature alignment.
"""

import os
import joblib
import numpy as np
from typing import List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocessing import TextPreprocessor

class EmailFeatureExtractor:
    """
    TF-IDF feature extraction for email text.
    
    This class combines text preprocessing with TF-IDF vectorization to
    convert raw email text into numerical feature vectors suitable for
    machine learning models.
    
    Example:
        >>> # Training phase
        >>> extractor = EmailFeatureExtractor(max_features=5000)
        >>> train_texts = ["urgent meeting tomorrow", "project update needed"]
        >>> train_features = extractor.fit_transform(train_texts)
        >>> extractor.save("artifacts/vectorizers/email_vectorizer_v1.pkl")
        
        >>> # Inference phase
        >>> extractor = EmailFeatureExtractor.load(
        ...     "artifacts/vectorizers/email_vectorizer_v1.pkl"
        ... )
        >>> new_text = ["meeting scheduled for monday"]
        >>> new_features = extractor.transform(new_text)
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        preprocessor_config: Optional[dict] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams (e.g., (1,2) = unigrams + bigrams)
            min_df: Minimum document frequency (ignore rare terms)
            max_df: Maximum document frequency (ignore very common terms)
            preprocessor_config: Configuration dict for TextPreprocessor
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize preprocessor
        if preprocessor_config is None:
            preprocessor_config = {
                'lowercase': True,
                'remove_punctuation': True,
                'remove_numbers': False,
                'remove_urls': True,
                'remove_emails': True,
                'remove_stopwords': True,
                'apply_lemmatization': True,
                'min_word_length': 2
            }
        
        self.preprocessor = TextPreprocessor(**preprocessor_config)
        
        # Initialize TF-IDF vectorizer
        # Use str.split as tokenizer so the extractor remains pickleable
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            strip_accents='unicode',
            lowercase=False,  # Already handled by preprocessor
            tokenizer=str.split,  # simple split; functions are pickleable
            token_pattern=None  # Disable built-in pattern
        )
        
        self._is_fitted = False
    
    def fit(self, texts: List[str]) -> 'EmailFeatureExtractor':
        """
        Fit the vectorizer on training texts.
        
        This learns the vocabulary and IDF weights from the training data.
        Must be called before transform() during training phase.
        
        Args:
            texts: List of raw email texts
            
        Returns:
            self (fitted extractor)
            
        Example:
            >>> extractor = EmailFeatureExtractor()
            >>> train_texts = ["meeting tomorrow", "project update"]
            >>> extractor.fit(train_texts)
        """
        # Preprocess texts
        cleaned_texts = self.preprocessor.clean_batch(texts)
        
        # Fit vectorizer
        try:
            self.vectorizer.fit(cleaned_texts)
        except ValueError as e:
            # sklearn throws a generic ValueError when no terms left after min_df/max_df
            if "After pruning, no terms remain" in str(e):
                raise ValueError(
                    "No terms remain after pruning. "
                    "Try lowering min_df, increasing max_df, or providing more training texts."
                ) from e
            raise
        self._is_fitted = True
        
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors.
        
        Uses the fitted vectorizer to convert texts to TF-IDF features.
        Must call fit() first during training, or load a fitted vectorizer.
        
        Args:
            texts: List of raw email texts
            
        Returns:
            Feature matrix (n_samples, n_features)
            
        Raises:
            ValueError: If vectorizer not fitted
            
        Example:
            >>> extractor.fit(train_texts)  # Must fit first
            >>> features = extractor.transform(test_texts)
            >>> features.shape
            (100, 5000)
        """
        if not self._is_fitted:
            raise ValueError(
                "Vectorizer not fitted. Call fit() first or load a fitted vectorizer."
            )
        
        # Preprocess texts
        cleaned_texts = self.preprocessor.clean_batch(texts)
        
        # Transform to features
        features = self.vectorizer.transform(cleaned_texts)
        
        return features.toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizer and transform texts in one step.
        
        Convenience method combining fit() and transform().
        Use this during training phase.
        
        Args:
            texts: List of raw email texts
            
        Returns:
            Feature matrix (n_samples, n_features)
            
        Example:
            >>> extractor = EmailFeatureExtractor()
            >>> train_features = extractor.fit_transform(train_texts)
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names (vocabulary terms).
        
        Returns:
            List of feature names
            
        Raises:
            ValueError: If vectorizer not fitted
            
        Example:
            >>> extractor.fit(train_texts)
            >>> features = extractor.get_feature_names()
            >>> features[:5]
            ['meeting', 'project', 'update', 'urgent', 'tomorrow']
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_vocabulary_size(self) -> int:
        """
        Get actual vocabulary size (may be less than max_features).
        
        Returns:
            Number of features in vocabulary
            
        Raises:
            ValueError: If vectorizer not fitted
        """
        if not self._is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        return len(self.vectorizer.vocabulary_)
    
    def save(self, filepath: str) -> None:
        """
        Save fitted extractor to disk.
        
        Saves both the vectorizer and preprocessor configuration.
        Creates directory if it doesn't exist.
        
        Args:
            filepath: Path to save the extractor (.pkl extension)
            
        Raises:
            ValueError: If vectorizer not fitted
            
        Example:
            >>> extractor.save("artifacts/vectorizers/email_vectorizer_v1.pkl")
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted vectorizer. Call fit() first.")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save entire extractor
        joblib.dump(self, filepath)
        print(f"Saved EmailFeatureExtractor to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EmailFeatureExtractor':
        """
        Load fitted extractor from disk.
        
        Args:
            filepath: Path to saved extractor file
            
        Returns:
            Loaded EmailFeatureExtractor
            
        Raises:
            FileNotFoundError: If file doesn't exist
            
        Example:
            >>> extractor = EmailFeatureExtractor.load(
            ...     "artifacts/vectorizers/email_vectorizer_v1.pkl"
            ... )
            >>> features = extractor.transform(new_texts)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Extractor file not found: {filepath}")
        
        extractor = joblib.load(filepath)
        print(f"Loaded EmailFeatureExtractor from {filepath}")
        return extractor
    
    def get_config(self) -> dict:
        """
        Get extractor configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'preprocessor_config': self.preprocessor.get_config(),
            'vocabulary_size': self.get_vocabulary_size() if self._is_fitted else None,
            'is_fitted': self._is_fitted
        }
    
    def __repr__(self) -> str:
        """String representation of extractor."""
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"EmailFeatureExtractor(max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, {status})"
        )
