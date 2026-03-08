"""
Email classification model training service.

Supports multiple algorithms:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes

Includes hyperparameter tuning with GridSearchCV.
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from feature_pipeline import EmailFeatureExtractor, PipelineConfig

class EmailClassifierTrainer:
    """
    Train email classification models with multiple algorithms.
    
    Example:
        >>> trainer = EmailClassifierTrainer()
        >>> trainer.train(
        ...     train_texts=['urgent meeting', 'project update'],
        ...     train_labels=['meeting', 'update'],
        ...     model_type='logistic_regression'
        ... )
        >>> trainer.save_model('artifacts/models/model_v1.0.pkl')
    """
    
    # Hyperparameter grids for tuning
    PARAM_GRIDS = {
        'logistic_regression': {
            'C': [0.1, 1.0, 10.0],
            'max_iter': [1000],
            'class_weight': ['balanced', None]
        },
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', None]
        },
        'svm': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'class_weight': ['balanced', None]
        },
        'naive_bayes': {
            'alpha': [0.1, 0.5, 1.0]
        }
    }
    
    def __init__(self, pipeline_config: Optional[PipelineConfig] = None):
        """
        Initialize trainer.
        
        Args:
            pipeline_config: Configuration for preprocessing pipeline
        """
        if pipeline_config is None:
            from feature_pipeline import get_default_config
            pipeline_config = get_default_config()
        
        self.config = pipeline_config
        self.feature_extractor = None
        self.model = None
        self.model_type = None
        self.best_params = None
        self.training_history = {}
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[str],
        model_type: str = 'logistic_regression',
        tune_hyperparams: bool = True,
        cv_folds: int = 5,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train classification model.
        
        Args:
            train_texts: Training email texts
            train_labels: Training labels
            model_type: Algorithm ('logistic_regression', 'random_forest', 
                       'svm', 'naive_bayes')
            tune_hyperparams: Perform hyperparameter tuning
            cv_folds: Number of cross-validation folds
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            
        Returns:
            Dictionary with training results and metrics
        """
        print("\n" + "=" * 70)
        print(f"TRAINING {model_type.upper().replace('_', ' ')} MODEL")
        print("=" * 70)
        
        self.model_type = model_type
        
        # Step 1: Feature extraction
        print("\n[1/4] Extracting features...")
        self.feature_extractor = EmailFeatureExtractor(
            max_features=self.config.feature_extraction.max_features,
            ngram_range=self.config.feature_extraction.ngram_range,
            min_df=self.config.feature_extraction.min_df,
            max_df=self.config.feature_extraction.max_df,
            preprocessor_config=self.config.preprocessing.to_dict()
        )
        
        X_train = self.feature_extractor.fit_transform(train_texts)
        y_train = np.array(train_labels)
        
        print(f"  ✓ Training features: {X_train.shape}")
        print(f"  ✓ Vocabulary size: {self.feature_extractor.get_vocabulary_size()}")
        
        # Validation features if provided
        X_val, y_val = None, None
        if val_texts is not None and val_labels is not None:
            X_val = self.feature_extractor.transform(val_texts)
            y_val = np.array(val_labels)
            print(f"  ✓ Validation features: {X_val.shape}")
        
        # Step 2: Initialize model
        print(f"\n[2/4] Initializing {model_type} classifier...")
        base_model = self._get_base_model(model_type)
        
        # Step 3: Hyperparameter tuning or direct training
        if tune_hyperparams and model_type in self.PARAM_GRIDS:
            print(f"\n[3/4] Tuning hyperparameters ({cv_folds}-fold CV)...")
            self.model, self.best_params = self._tune_hyperparameters(
                base_model, X_train, y_train, model_type, cv_folds
            )
        else:
            print(f"\n[3/4] Training with default parameters...")
            self.model = base_model
            self.model.fit(X_train, y_train)
            self.best_params = None
        
        # Step 4: Evaluate
        print(f"\n[4/4] Evaluating model...")
        results = self._evaluate_training(
            X_train, y_train, X_val, y_val
        )
        
        # Store training history
        self.training_history = {
            'model_type': model_type,
            'training_samples': len(train_texts),
            'validation_samples': len(val_texts) if val_texts else 0,
            'num_classes': len(np.unique(train_labels)),
            'vocabulary_size': self.feature_extractor.get_vocabulary_size(),
            'best_params': self.best_params,
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE")
        print("=" * 70 + "\n")
        
        return results
    
    def _get_base_model(self, model_type: str):
        """Get base model by type."""
        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': MultinomialNB()
        }
        
        if model_type not in models:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported: {list(models.keys())}"
            )
        
        return models[model_type]
    
    def _tune_hyperparameters(
        self,
        base_model,
        X_train,
        y_train,
        model_type: str,
        cv_folds: int
    ) -> Tuple[Any, Dict]:
        """Perform hyperparameter tuning with GridSearchCV."""
        param_grid = self.PARAM_GRIDS[model_type]
        
        print(f"  Testing {self._count_combinations(param_grid)} parameter combinations...")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\n  ✓ Best parameters: {grid_search.best_params_}")
        print(f"  ✓ Best CV F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def _count_combinations(self, param_grid: Dict) -> int:
        """Count total parameter combinations."""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count
    
    def _evaluate_training(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None
    ) -> Dict[str, Any]:
        """Evaluate trained model."""
        results = {}
        
        # Training set performance
        train_pred = self.model.predict(X_train)
        train_accuracy = (train_pred == y_train).mean()
        results['train_accuracy'] = train_accuracy
        print(f"  Training accuracy: {train_accuracy:.4f}")
        
        # Validation set performance
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = (val_pred == y_val).mean()
            results['val_accuracy'] = val_accuracy
            print(f"  Validation accuracy: {val_accuracy:.4f}")
            
            # Check overfitting
            if train_accuracy - val_accuracy > 0.1:
                print(f"  ⚠ Potential overfitting detected (gap: {train_accuracy - val_accuracy:.4f})")
        
        return results
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict categories for new emails.
        
        Args:
            texts: Email texts to classify
            
        Returns:
            Array of predicted categories
        """
        if self.model is None or self.feature_extractor is None:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self.feature_extractor.transform(texts)
        predictions = self.model.predict(features)
        return predictions
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probability distributions.
        
        Args:
            texts: Email texts to classify
            
        Returns:
            Array of probability distributions (n_samples, n_classes)
        """
        if self.model is None or self.feature_extractor is None:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self.feature_extractor.transform(texts)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)
        else:
            # Fallback for models without predict_proba
            predictions = self.model.predict(features)
            probabilities = np.zeros((len(predictions), len(self.model.classes_)))
            for i, pred in enumerate(predictions):
                class_idx = np.where(self.model.classes_ == pred)[0][0]
                probabilities[i, class_idx] = 1.0
        
        return probabilities
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get most important features per class.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary mapping class to list of (feature, importance) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        feature_names = self.feature_extractor.get_feature_names()
        importance_per_class = {}
        
        if isinstance(self.model, LogisticRegression):
            # Coefficients for each class
            for idx, class_name in enumerate(self.model.classes_):
                if len(self.model.classes_) == 2 and self.model.coef_.shape[0] == 1:
                    # Binary classification
                    coef = self.model.coef_[0] if idx == 1 else -self.model.coef_[0]
                else:
                    coef = self.model.coef_[idx]
                
                top_indices = np.argsort(np.abs(coef))[-top_n:][::-1]
                top_features = [(feature_names[i], coef[i]) for i in top_indices]
                importance_per_class[class_name] = top_features
        
        elif isinstance(self.model, RandomForestClassifier):
            # Feature importance from Random Forest
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]
            top_features = [(feature_names[i], importances[i]) for i in top_indices]
            importance_per_class['overall'] = top_features
        
        return importance_per_class
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model and feature extractor.
        
        Args:
            filepath: Path to save model (.pkl extension)
        """
        if self.model is None or self.feature_extractor is None:
            raise ValueError("Model not trained. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Package model + extractor + config
        package = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'config': self.config,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'training_history': self.training_history,
            'classes': self.model.classes_.tolist()
        }
        
        joblib.dump(package, filepath)
        print(f"✓ Saved model to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EmailClassifierTrainer':
        """
        Load trained model.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded EmailClassifierTrainer instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        package = joblib.load(filepath)
        
        trainer = cls(pipeline_config=package['config'])
        trainer.model = package['model']
        trainer.feature_extractor = package['feature_extractor']
        trainer.model_type = package['model_type']
        trainer.best_params = package.get('best_params')
        trainer.training_history = package.get('training_history', {})
        
        print(f"✓ Loaded model from {filepath}")
        return trainer

