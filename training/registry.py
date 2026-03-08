"""
Model registry for database integration.

Registers trained models in the model_versions table with:
- Version identifier
- Performance metrics
- Configuration reference
- Artifact paths
"""

import os
from datetime import datetime
from typing import Dict, Optional
from sqlalchemy import text

# Import database components from Step 1
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.connection import DatabaseConnection
from database.models import ModelVersion
from database.repository import ModelVersionRepository

class ModelRegistry:
    """
    Register trained models in database.
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.register_model(
        ...     version='1.0',
        ...     model_type='logistic_regression',
        ...     metrics={'accuracy': 0.89, 'f1_score': 0.87},
        ...     model_path='artifacts/models/model_v1.0.pkl',
        ...     vectorizer_path='artifacts/vectorizers/vectorizer_v1.0.pkl'
        ... )
    """
    
    def __init__(self):
        """Initialize registry with database connection."""
        self.db = DatabaseConnection()
        self.repo = ModelVersionRepository()
    
    def register_model(
        self,
        version: str,
        model_type: str,
        metrics: Dict[str, float],
        model_path: str,
        vectorizer_path: str,
        config_path: Optional[str] = None,
        description: Optional[str] = None,
        set_active: bool = False
    ) -> int:
        """
        Register model in database.
        
        Args:
            version: Model version identifier
            model_type: Algorithm type
            metrics: Performance metrics dictionary
            model_path: Path to saved model file
            vectorizer_path: Path to saved vectorizer
            config_path: Path to config file (optional)
            description: Model description (optional)
            set_active: Set this model as active (default: False)
            
        Returns:
            Model ID from database
        """
        print("\n" + "=" * 70)
        print("REGISTERING MODEL IN DATABASE")
        print("=" * 70)
        
        # Prepare model version data
        model_data = {
            'version': version,
            'model_type': model_type,
            'model_path': model_path,
            'vectorizer_path': vectorizer_path,
            'config_path': config_path,
            'accuracy': metrics.get('accuracy'),
            'precision_score': metrics.get('precision'),
            'recall_score': metrics.get('recall'),
            'f1_score': metrics.get('f1_score'),
            'description': description or f"{model_type} v{version}",
            'is_active': set_active,
            'trained_at': datetime.now()
        }
        
        print(f"  Version: {version}")
        print(f"  Model type: {model_type}")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
        
        # If setting as active, deactivate other models
        if set_active:
            print(f"  Setting as active model...")
            with self.db.get_session() as session:
                # Deactivate all models
                session.execute(
                    text("UPDATE model_versions SET is_active = FALSE")
                )
                session.commit()
        
        # Create model version record
        with self.db.get_session() as session:
            model_version = self.repo.create(
                session,
                version=model_data['version'],
                accuracy=model_data.get('accuracy'),
                precision_score=model_data.get('precision_score'),
                recall_score=model_data.get('recall_score'),
                f1_score=model_data.get('f1_score'),
                model_path=model_data.get('model_path'),
                vectorizer_path=model_data.get('vectorizer_path'),
                # Store minimal training metadata; extended metrics can be stored here
                training_samples=model_data.get('training_samples', 0),
                training_metrics=model_data.get('training_metrics', model_data.get('metrics', {}))
            )
            model_id = model_version.id

        print(f"  ✓ Registered as model ID: {model_id}")
        print("=" * 70 + "\n")
        
        return model_id
    
    def get_active_model(self) -> Optional[ModelVersion]:
        """
        Get currently active model.
        
        Returns:
            Active ModelVersion or None
        """
        with self.db.get_session() as session:
            model = self.repo.get_active(session)
            if model:
                # Detach from session so it can be accessed after session closes
                session.expunge(model)
            return model
    
    def get_model_by_version(self, version: str) -> Optional[ModelVersion]:
        """
        Get model by version string.
        
        Args:
            version: Version identifier
            
        Returns:
            ModelVersion or None
        """
        with self.db.get_session() as session:
            models = self.repo.get_all(session)
            for model in models:
                if model.version == version:
                    return model
        return None
    
    def list_all_models(self) -> list:
        """
        List all registered models.
        
        Returns:
            List of ModelVersion objects
        """
        with self.db.get_session() as session:
            return self.repo.get_all(session)
    
    def set_model_active(self, model_id: int) -> None:
        """
        Set a model as active.
        
        Args:
            model_id: Database ID of model to activate
        """
        with self.db.get_session() as session:
            # Deactivate all
            session.execute(
                text("UPDATE model_versions SET is_active = FALSE")
            )
            # Activate target
            session.execute(
                text("UPDATE model_versions SET is_active = TRUE WHERE id = :id"),
                {"id": model_id}
            )
            session.commit()
        
        print(f"✓ Set model {model_id} as active")
    
    def compare_models(self) -> None:
        """Print comparison of all registered models."""
        with self.db.get_session() as session:
            models = self.repo.get_all(session)

            if not models:
                print("No models registered yet.")
                return

            print("\n" + "=" * 100)
            print("MODEL COMPARISON")
            print("=" * 100)
            print(f"{'ID':<5} {'Version':<10} {'Type':<20} {'Accuracy':<10} {'F1':<10} {'Active':<8} {'Trained'}")
            print("-" * 100)

            for model in models:
                accuracy = f"{model.accuracy:.4f}" if model.accuracy else "N/A"
                f1 = f"{model.f1_score:.4f}" if model.f1_score else "N/A"
                active = "Yes" if model.is_active else "No"
                trained = "N/A"
                if getattr(model, 'deployed_at', None):
                    trained = model.deployed_at.strftime("%Y-%m-%d %H:%M")
                elif getattr(model, 'created_at', None):
                    trained = model.created_at.strftime("%Y-%m-%d %H:%M")

                model_type = getattr(model, 'model_type', 'N/A')
                print(f"{model.id:<5} {model.version:<10} {model_type:<20} {accuracy:<10} {f1:<10} {active:<8} {trained}")

            print("=" * 100 + "\n")

