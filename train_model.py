"""
Complete training pipeline for email classification.

Usage:
    python train_model.py --data data/sample_emails.csv --version 1.0 --model logistic_regression

This script:
1. Loads and validates data
2. Splits into train/val/test sets
3. Trains model with hyperparameter tuning
4. Evaluates performance
5. Saves model artifacts
6. Registers in database
"""

import argparse
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training import EmailDataLoader, EmailClassifierTrainer, ModelEvaluator, ModelRegistry
from feature_pipeline import PipelineConfig, ArtifactManager, get_default_config

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train email classification model')
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to training data CSV file'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Model version (default: auto-generate from timestamp)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='logistic_regression',
        choices=['logistic_regression', 'random_forest', 'svm', 'naive_bayes'],
        help='Model algorithm (default: logistic_regression)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set proportion (default: 0.2)'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Validation set proportion (default: 0.1)'
    )
    
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Disable hyperparameter tuning (faster but may reduce performance)'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Maximum TF-IDF features (default: 5000)'
    )
    
    parser.add_argument(
        '--set-active',
        action='store_true',
        help='Set this model as active in database'
    )
    
    parser.add_argument(
        '--description',
        type=str,
        default=None,
        help='Model description'
    )
    
    return parser.parse_args()

def main():
    """Main training pipeline."""
    args = parse_arguments()
    
    print("\n" + "=" * 70)
    print("EMAIL CLASSIFICATION MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Version: {args.version or 'auto-generate'}")
    print("=" * 70 + "\n")
    
    # Initialize components
    artifact_manager = ArtifactManager()
    
    # Generate version if not provided
    if args.version is None:
        args.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Auto-generated version: {args.version}\n")
    
    # STEP 1: Load and validate data
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING & VALIDATION")
    print("=" * 70)
    
    loader = EmailDataLoader()
    df = loader.load_csv(args.data)
    validation_results = loader.validate_data(df)
    
    # Clean data if issues found
    if validation_results['status'] == 'warning':
        print("\nCleaning data...")
        df = loader.clean_data(df)
        loader.validate_data(df)
    
    # STEP 2: Split data
    print("\n" + "=" * 70)
    print("STEP 2: DATA SPLITTING")
    print("=" * 70)
    
    splits = loader.split_data(
        df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=42,
        stratify=True
    )
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    # STEP 3: Create pipeline configuration
    print("\n" + "=" * 70)
    print("STEP 3: PIPELINE CONFIGURATION")
    print("=" * 70)
    
    config = get_default_config()
    config.version = args.version
    config.feature_extraction.max_features = args.max_features
    config.description = args.description or f"{args.model} model v{args.version}"
    
    config_path = artifact_manager.get_config_path(args.version)
    config.save(config_path)
    print(f"✓ Saved configuration to {config_path}")
    
    # STEP 4: Train model
    print("\n" + "=" * 70)
    print("STEP 4: MODEL TRAINING")
    print("=" * 70)
    
    trainer = EmailClassifierTrainer(pipeline_config=config)
    
    training_results = trainer.train(
        train_texts=X_train.tolist(),
        train_labels=y_train.tolist(),
        model_type=args.model,
        tune_hyperparams=not args.no_tune,
        cv_folds=5,
        val_texts=X_val.tolist() if len(X_val) > 0 else None,
        val_labels=y_val.tolist() if len(y_val) > 0 else None
    )
    
    # STEP 5: Evaluate on test set
    print("\n" + "=" * 70)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 70)
    
    evaluator = ModelEvaluator()
    
    # Predictions
    y_pred = trainer.predict(X_test.tolist())
    y_pred_proba = trainer.predict_proba(X_test.tolist())
    
    # Metrics
    test_metrics = evaluator.evaluate(y_test.values, y_pred, y_pred_proba)
    
    print("\nTest Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(evaluator.get_classification_report(y_test.values, y_pred))
    
    # Generate and save evaluation report
    report_path = f"artifacts/reports/evaluation_report_v{args.version}.txt"
    evaluator.generate_evaluation_report(
        y_test.values,
        y_pred,
        model_name=f"{args.model} v{args.version}",
        save_path=report_path
    )
    
    # Plot and save confusion matrix
    cm_path = f"artifacts/reports/confusion_matrix_v{args.version}.png"
    evaluator.plot_confusion_matrix(
        y_test.values,
        y_pred,
        save_path=cm_path,
        normalize=True
    )
    
    # STEP 6: Save model artifacts
    print("\n" + "=" * 70)
    print("STEP 6: SAVING ARTIFACTS")
    print("=" * 70)
    
    model_path = artifact_manager.get_model_path(args.version)
    trainer.save_model(model_path)
    
    # Vectorizer is saved inside the model package, but also save separately
    vectorizer_path = artifact_manager.get_vectorizer_path(args.version)
    trainer.feature_extractor.save(vectorizer_path)
    
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Vectorizer saved to: {vectorizer_path}")
    print(f"✓ Config saved to: {config_path}")
    print(f"✓ Report saved to: {report_path}")
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    # STEP 7: Register in database
    print("\n" + "=" * 70)
    print("STEP 7: DATABASE REGISTRATION")
    print("=" * 70)
    
    registry = ModelRegistry()
    model_id = registry.register_model(
        version=args.version,
        model_type=args.model,
        metrics=test_metrics,
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        config_path=config_path,
        description=config.description,
        set_active=args.set_active
    )
    
    print(f"✓ Registered in database with ID: {model_id}")
    
    # STEP 8: Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Model Version: {args.version}")
    print(f"Model Type: {args.model}")
    print(f"Database ID: {model_id}")
    print(f"\nPerformance:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"\nArtifacts:")
    print(f"  Model: {model_path}")
    print(f"  Vectorizer: {vectorizer_path}")
    print(f"  Config: {config_path}")
    print(f"  Report: {report_path}")
    print(f"  Confusion Matrix: {cm_path}")
    
    if args.set_active:
        print(f"\n✓ This model is now ACTIVE in production")
    else:
        print(f"\n  Note: Use --set-active flag to make this model active")
    
    print("=" * 70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
