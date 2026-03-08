"""
Model evaluation and performance metrics.

Calculates:
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- Per-class metrics
- Classification report
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

class ModelEvaluator:
    """
    Evaluate email classification model performance.
    
    Example:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.evaluate(
        ...     y_true=['meeting', 'spam', 'meeting'],
        ...     y_pred=['meeting', 'meeting', 'meeting']
        ... )
        >>> evaluator.plot_confusion_matrix(
        ...     y_true, y_pred, 
        ...     save_path='artifacts/reports/confusion_matrix.png'
        ... )
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 (weighted average)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        # Macro averages (unweighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_score_macro'] = f1_macro
        
        return metrics
    
    def evaluate_per_class(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            DataFrame with per-class metrics
        """
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, zero_division=0
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'class': classes,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        })
        
        return df
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate sklearn classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        return classification_report(y_true, y_pred, zero_division=0)
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization ('true', 'pred', 'all', or None)
            
        Returns:
            Tuple of (confusion matrix, class labels)
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=classes, normalize=normalize)
        return cm, classes.tolist()
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = False,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save plot (optional)
            normalize: Normalize by true labels
            figsize: Figure size
        """
        cm, classes = self.get_confusion_matrix(
            y_true, y_pred,
            normalize='true' if normalize else None
        )
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved confusion matrix to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of model
            save_path: Path to save report (optional)
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append(f"{model_name.upper()} - EVALUATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Overall metrics
        metrics = self.evaluate(y_true, y_pred)
        report_lines.append("OVERALL METRICS:")
        report_lines.append(f"  Accuracy:  {metrics['accuracy']:.4f}")
        report_lines.append(f"  Precision: {metrics['precision']:.4f} (weighted)")
        report_lines.append(f"  Recall:    {metrics['recall']:.4f} (weighted)")
        report_lines.append(f"  F1-Score:  {metrics['f1_score']:.4f} (weighted)")
        report_lines.append("")
        
        # Per-class metrics
        report_lines.append("PER-CLASS METRICS:")
        report_lines.append(self.get_classification_report(y_true, y_pred))
        report_lines.append("")
        
        # Confusion matrix summary
        cm, classes = self.get_confusion_matrix(y_true, y_pred)
        report_lines.append("CONFUSION MATRIX:")
        report_lines.append(f"  Classes: {classes}")
        report_lines.append(f"  Total predictions: {len(y_pred)}")
        report_lines.append(f"  Correct predictions: {np.trace(cm)}")
        report_lines.append(f"  Incorrect predictions: {len(y_pred) - np.trace(cm)}")
        report_lines.append("")
        
        report_lines.append("=" * 70)
        
        report = "\n".join(report_lines)
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"✓ Saved evaluation report to {save_path}")
        
        return report
