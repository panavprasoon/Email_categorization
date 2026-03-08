# save/load utilities for feature pipeline artifacts (e.g. vectorizers, encoders, etc.)

"""
Artifact management for models, vectorizers, and configurations.

This module provides utilities for organizing and managing ML artifacts
including trained models, fitted vectorizers, and pipeline configurations.

Artifact organization:
artifacts/
├── models/               # Trained classification models
│   └── model_v1.0.pkl
├── vectorizers/          # Fitted TF-IDF vectorizers
│   └── vectorizer_v1.0.pkl
└── configs/              # Pipeline configurations
    └── config_v1.0.json
"""

import os
import re
from typing import Optional, List, Tuple
from datetime import datetime

class ArtifactManager:
    """
    Manager for ML artifacts (models, vectorizers, configs).
    
    Provides methods for:
    - Generating versioned artifact paths
    - Listing available artifacts
    - Finding latest versions
    - Cleaning up old artifacts
    
    Example:
        >>> manager = ArtifactManager()
        >>> model_path = manager.get_model_path("1.0")
        >>> manager.save_model(model, model_path)
        >>> latest = manager.get_latest_model()
    """
    
    def __init__(self, base_dir: str = "artifacts"):
        """
        Initialize artifact manager.
        
        Args:
            base_dir: Base directory for artifacts
        """
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.vectorizers_dir = os.path.join(base_dir, "vectorizers")
        self.configs_dir = os.path.join(base_dir, "configs")
        
        # Create directories if they don't exist
        for dir_path in [self.models_dir, self.vectorizers_dir, self.configs_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_model_path(self, version: str, create_dir: bool = True) -> str:
        """
        Get path for model artifact.
        
        Args:
            version: Model version identifier
            create_dir: Create directory if it doesn't exist
            
        Returns:
            Full path to model file
            
        Example:
            >>> manager.get_model_path("1.0")
            'artifacts/models/model_v1.0.pkl'
        """
        if create_dir:
            os.makedirs(self.models_dir, exist_ok=True)
        return os.path.join(self.models_dir, f"model_v{version}.pkl")
    
    def get_vectorizer_path(self, version: str, create_dir: bool = True) -> str:
        """
        Get path for vectorizer artifact.
        
        Args:
            version: Vectorizer version identifier
            create_dir: Create directory if it doesn't exist
            
        Returns:
            Full path to vectorizer file
            
        Example:
            >>> manager.get_vectorizer_path("1.0")
            'artifacts/vectorizers/vectorizer_v1.0.pkl'
        """
        if create_dir:
            os.makedirs(self.vectorizers_dir, exist_ok=True)
        return os.path.join(self.vectorizers_dir, f"vectorizer_v{version}.pkl")
    
    def get_config_path(self, version: str, create_dir: bool = True) -> str:
        """
        Get path for configuration artifact.
        
        Args:
            version: Configuration version identifier
            create_dir: Create directory if it doesn't exist
            
        Returns:
            Full path to config file
            
        Example:
            >>> manager.get_config_path("1.0")
            'artifacts/configs/config_v1.0.json'
        """
        if create_dir:
            os.makedirs(self.configs_dir, exist_ok=True)
        return os.path.join(self.configs_dir, f"config_v{version}.json")
    
    def list_models(self) -> List[Tuple[str, str]]:
        """
        List all available model artifacts.
        
        Returns:
            List of (version, filepath) tuples
            
        Example:
            >>> manager.list_models()
            [('1.0', 'artifacts/models/model_v1.0.pkl'),
             ('1.1', 'artifacts/models/model_v1.1.pkl')]
        """
        return self._list_artifacts(self.models_dir, "model_v", ".pkl")
    
    def list_vectorizers(self) -> List[Tuple[str, str]]:
        """
        List all available vectorizer artifacts.
        
        Returns:
            List of (version, filepath) tuples
        """
        return self._list_artifacts(self.vectorizers_dir, "vectorizer_v", ".pkl")
    
    def list_configs(self) -> List[Tuple[str, str]]:
        """
        List all available configuration artifacts.
        
        Returns:
            List of (version, filepath) tuples
        """
        return self._list_artifacts(self.configs_dir, "config_v", ".json")
    
    def _list_artifacts(
        self, 
        directory: str, 
        prefix: str, 
        extension: str
    ) -> List[Tuple[str, str]]:
        """
        List artifacts in a directory matching pattern.
        
        Args:
            directory: Directory to search
            prefix: Filename prefix (e.g., "model_v")
            extension: File extension (e.g., ".pkl")
            
        Returns:
            List of (version, filepath) tuples sorted by version
        """
        if not os.path.exists(directory):
            return []
        
        pattern = re.compile(f"^{prefix}(.+){re.escape(extension)}$")
        artifacts = []
        
        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                version = match.group(1)
                filepath = os.path.join(directory, filename)
                artifacts.append((version, filepath))
        
        # Sort by version (simple string sort works for semantic versioning)
        artifacts.sort(key=lambda x: x[0])
        return artifacts
    
    def get_latest_model(self) -> Optional[str]:
        """
        Get path to the latest model version.
        
        Returns:
            Path to latest model, or None if no models exist
            
        Example:
            >>> manager.get_latest_model()
            'artifacts/models/model_v1.2.pkl'
        """
        models = self.list_models()
        return models[-1][1] if models else None
    
    def get_latest_vectorizer(self) -> Optional[str]:
        """
        Get path to the latest vectorizer version.
        
        Returns:
            Path to latest vectorizer, or None if no vectorizers exist
        """
        vectorizers = self.list_vectorizers()
        return vectorizers[-1][1] if vectorizers else None
    
    def get_latest_config(self) -> Optional[str]:
        """
        Get path to the latest configuration version.
        
        Returns:
            Path to latest config, or None if no configs exist
        """
        configs = self.list_configs()
        return configs[-1][1] if configs else None
    
    def generate_version(self, prefix: str = "") -> str:
        """
        Generate new version identifier based on timestamp.
        
        Args:
            prefix: Optional prefix for version (e.g., "prod")
            
        Returns:
            Version string in format: prefix_YYYYMMDD_HHMMSS
            
        Example:
            >>> manager.generate_version("prod")
            'prod_20260302_143022'
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if prefix:
            return f"{prefix}_{timestamp}"
        return timestamp
    
    def delete_artifact(self, filepath: str) -> bool:
        """
        Delete an artifact file.
        
        Args:
            filepath: Path to artifact file
            
        Returns:
            True if deleted successfully, False otherwise
            
        Example:
            >>> manager.delete_artifact("artifacts/models/model_v1.0.pkl")
            True
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Deleted artifact: {filepath}")
                return True
            else:
                print(f"Artifact not found: {filepath}")
                return False
        except Exception as e:
            print(f"Error deleting artifact {filepath}: {e}")
            return False
    
    def cleanup_old_versions(
        self, 
        artifact_type: str = "all", 
        keep_latest: int = 3
    ) -> int:
        """
        Delete old artifact versions, keeping only the latest N versions.
        
        Args:
            artifact_type: Type of artifacts to clean ("models", "vectorizers", 
                          "configs", or "all")
            keep_latest: Number of latest versions to keep
            
        Returns:
            Number of artifacts deleted
            
        Example:
            >>> # Keep only latest 3 models
            >>> manager.cleanup_old_versions("models", keep_latest=3)
            2
        """
        deleted_count = 0
        
        artifact_types = {
            'models': self.list_models,
            'vectorizers': self.list_vectorizers,
            'configs': self.list_configs
        }
        
        if artifact_type == "all":
            types_to_clean = artifact_types.keys()
        else:
            types_to_clean = [artifact_type]
        
        for atype in types_to_clean:
            if atype not in artifact_types:
                continue
            
            artifacts = artifact_types[atype]()
            
            # Keep only latest N versions
            if len(artifacts) > keep_latest:
                to_delete = artifacts[:-keep_latest]
                for version, filepath in to_delete:
                    if self.delete_artifact(filepath):
                        deleted_count += 1
        
        print(f"Cleaned up {deleted_count} old artifact(s)")
        return deleted_count
    
    def get_artifact_info(self) -> dict:
        """
        Get summary information about all artifacts.
        
        Returns:
            Dictionary with artifact counts and latest versions
            
        Example:
            >>> manager.get_artifact_info()
            {
                'models': {'count': 3, 'latest': '1.2'},
                'vectorizers': {'count': 3, 'latest': '1.2'},
                'configs': {'count': 3, 'latest': '1.2'}
            }
        """
        models = self.list_models()
        vectorizers = self.list_vectorizers()
        configs = self.list_configs()
        
        return {
            'models': {
                'count': len(models),
                'latest': models[-1][0] if models else None
            },
            'vectorizers': {
                'count': len(vectorizers),
                'latest': vectorizers[-1][0] if vectorizers else None
            },
            'configs': {
                'count': len(configs),
                'latest': configs[-1][0] if configs else None
            }
        }
