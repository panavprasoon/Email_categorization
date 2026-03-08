# package initialization

"""
Feature Pipeline Package

This package provides text preprocessing and feature extraction for
email categorization.

Main components:
- TextPreprocessor: Text cleaning and normalization
- EmailFeatureExtractor: TF-IDF vectorization
- PipelineConfig: Configuration management
- ArtifactManager: Artifact organization

Example usage:
    >>> from feature_pipeline import EmailFeatureExtractor, PipelineConfig
    >>> 
    >>> # Create configuration
    >>> config = PipelineConfig()
    >>> config.save("artifacts/configs/config_v1.0.json")
    >>> 
    >>> # Train feature extractor
    >>> extractor = EmailFeatureExtractor(
    ...     max_features=config.feature_extraction.max_features
    ... )
    >>> features = extractor.fit_transform(train_texts)
    >>> extractor.save("artifacts/vectorizers/vectorizer_v1.0.pkl")
    >>> 
    >>> # Load for inference
    >>> extractor = EmailFeatureExtractor.load(
    ...     "artifacts/vectorizers/vectorizer_v1.0.pkl"
    ... )
    >>> new_features = extractor.transform(new_texts)
"""

from .preprocessing import TextPreprocessor
from .feature_extractor import EmailFeatureExtractor
from .pipeline_config import (
    PreprocessingConfig,
    FeatureExtractionConfig,
    PipelineConfig,
    get_default_config,
    get_aggressive_config,
    get_minimal_config
)
from .artifact_manager import ArtifactManager

__version__ = "1.0.0"

__all__ = [
    'TextPreprocessor',
    'EmailFeatureExtractor',
    'PreprocessingConfig',
    'FeatureExtractionConfig',
    'PipelineConfig',
    'get_default_config',
    'get_aggressive_config',
    'get_minimal_config',
    'ArtifactManager'
]
