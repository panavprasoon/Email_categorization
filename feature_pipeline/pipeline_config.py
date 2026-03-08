# configuration

"""
Pipeline configuration management for email categorization.

This module provides dataclasses for managing preprocessing and feature
extraction configurations. Configurations can be serialized to JSON for
versioning and reproducibility.

CRITICAL: Save configuration with each trained model to ensure
identical preprocessing during inference.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

@dataclass
class PreprocessingConfig:
    """
    Configuration for text preprocessing.
    
    Attributes:
        lowercase: Convert text to lowercase
        remove_punctuation: Remove punctuation characters
        remove_numbers: Remove numeric digits
        remove_urls: Remove HTTP/HTTPS URLs
        remove_emails: Remove email addresses
        remove_stopwords: Remove common stopwords
        apply_lemmatization: Apply word lemmatization
        min_word_length: Minimum word length to keep
        max_word_length: Maximum word length (None = no limit)
    """
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_urls: bool = True
    remove_emails: bool = True
    remove_stopwords: bool = True
    apply_lemmatization: bool = True
    min_word_length: int = 2
    max_word_length: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PreprocessingConfig':
        """Create from dictionary."""
        return cls(**config_dict)

@dataclass
class FeatureExtractionConfig:
    """
    Configuration for TF-IDF feature extraction.
    
    Attributes:
        max_features: Maximum vocabulary size
        ngram_range: N-gram range (min, max)
        min_df: Minimum document frequency
        max_df: Maximum document frequency
    """
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeatureExtractionConfig':
        """Create from dictionary."""
        # Convert ngram_range from list to tuple if needed
        if 'ngram_range' in config_dict and isinstance(config_dict['ngram_range'], list):
            config_dict['ngram_range'] = tuple(config_dict['ngram_range'])
        return cls(**config_dict)

@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration including preprocessing and feature extraction.
    
    This class combines all configuration settings and provides methods for
    saving/loading configurations as JSON files.
    
    Attributes:
        version: Configuration version identifier
        created_at: Timestamp of creation
        preprocessing: Preprocessing configuration
        feature_extraction: Feature extraction configuration
        description: Optional description of this configuration
    """
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    description: str = "Default pipeline configuration"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            'version': self.version,
            'created_at': self.created_at,
            'preprocessing': self.preprocessing.to_dict(),
            'feature_extraction': self.feature_extraction.to_dict(),
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create from dictionary."""
        preprocessing = PreprocessingConfig.from_dict(config_dict['preprocessing'])
        feature_extraction = FeatureExtractionConfig.from_dict(config_dict['feature_extraction'])
        
        return cls(
            version=config_dict.get('version', '1.0'),
            created_at=config_dict.get('created_at', datetime.now().isoformat()),
            preprocessing=preprocessing,
            feature_extraction=feature_extraction,
            description=config_dict.get('description', '')
        )
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save configuration (.json extension)
            
        Example:
            >>> config = PipelineConfig()
            >>> config.save("artifacts/configs/pipeline_config_v1.json")
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"Saved pipeline configuration to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PipelineConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Loaded PipelineConfig
            
        Example:
            >>> config = PipelineConfig.load(
            ...     "artifacts/configs/pipeline_config_v1.json"
            ... )
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls.from_dict(config_dict)
        print(f"Loaded pipeline configuration from {filepath}")
        return config
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PipelineConfig(version={self.version}, "
            f"max_features={self.feature_extraction.max_features})"
        )

# Predefined configurations for common use cases

def get_default_config() -> PipelineConfig:
    """
    Get default pipeline configuration.
    
    Returns:
        Default PipelineConfig with standard settings
    """
    return PipelineConfig(
        version="1.0",
        description="Default configuration with standard preprocessing and 5000 features"
    )

def get_aggressive_config() -> PipelineConfig:
    """
    Get aggressive preprocessing configuration.
    
    More aggressive cleaning: removes numbers, more stopwords, stricter length filters.
    
    Returns:
        PipelineConfig with aggressive preprocessing
    """
    preprocessing = PreprocessingConfig(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=True,  # Remove numbers
        remove_urls=True,
        remove_emails=True,
        remove_stopwords=True,
        apply_lemmatization=True,
        min_word_length=3,  # Longer minimum
        max_word_length=15   # Set maximum length
    )
    
    feature_extraction = FeatureExtractionConfig(
        max_features=3000,  # Smaller vocabulary
        ngram_range=(1, 2),
        min_df=3,           # Higher minimum frequency
        max_df=0.90         # Lower maximum frequency
    )
    
    return PipelineConfig(
        version="1.0-aggressive",
        preprocessing=preprocessing,
        feature_extraction=feature_extraction,
        description="Aggressive preprocessing with stricter filtering"
    )

def get_minimal_config() -> PipelineConfig:
    """
    Get minimal preprocessing configuration.
    
    Lighter preprocessing: keeps numbers, no lemmatization, larger vocabulary.
    
    Returns:
        PipelineConfig with minimal preprocessing
    """
    preprocessing = PreprocessingConfig(
        lowercase=True,
        remove_punctuation=True,
        remove_numbers=False,  # Keep numbers
        remove_urls=True,
        remove_emails=True,
        remove_stopwords=False,  # Keep stopwords
        apply_lemmatization=False,  # No lemmatization
        min_word_length=1,
        max_word_length=None
    )
    
    feature_extraction = FeatureExtractionConfig(
        max_features=10000,  # Larger vocabulary
        ngram_range=(1, 3),  # Include trigrams
        min_df=1,
        max_df=1.0
    )
    
    return PipelineConfig(
        version="1.0-minimal",
        preprocessing=preprocessing,
        feature_extraction=feature_extraction,
        description="Minimal preprocessing with larger vocabulary"
    )

