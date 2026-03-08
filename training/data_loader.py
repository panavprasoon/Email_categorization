"""
Data loading and validation utilities for email classification training.

This module provides:
- CSV data loading
- Data validation (missing values, format checks)
- Train/validation/test splitting
- Class distribution analysis
- Data sampling and balancing
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from collections import Counter

class EmailDataLoader:
    """
    Load and validate email datasets for training.
    
    Example:
        >>> loader = EmailDataLoader()
        >>> df = loader.load_csv("data/sample_emails.csv")
        >>> loader.validate_data(df)
        >>> splits = loader.split_data(df, test_size=0.2, val_size=0.1)
    """
    
    def __init__(self, text_column: str = 'text', label_column: str = 'category'):
        """
        Initialize data loader.
        
        Args:
            text_column: Name of column containing email text
            label_column: Name of column containing category labels
        """
        self.text_column = text_column
        self.label_column = label_column
    
    def load_csv(self, filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load email data from CSV file.
        
        Args:
            filepath: Path to CSV file
            encoding: File encoding (default: utf-8)
            
        Returns:
            DataFrame with email data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns missing
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Load CSV
        df = pd.read_csv(filepath, encoding=encoding)
        
        print(f"Loaded {len(df)} emails from {filepath}")
        
        # Validate required columns
        required_cols = [self.text_column, self.label_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality and report issues.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        print("\n" + "=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        
        validation = {}
        
        # Check total records
        total_records = len(df)
        validation['total_records'] = total_records
        print(f"Total records: {total_records}")
        
        # Check missing values
        missing_text = df[self.text_column].isna().sum()
        missing_labels = df[self.label_column].isna().sum()
        validation['missing_text'] = missing_text
        validation['missing_labels'] = missing_labels
        
        print(f"\nMissing values:")
        print(f"  - Text: {missing_text} ({missing_text/total_records*100:.1f}%)")
        print(f"  - Labels: {missing_labels} ({missing_labels/total_records*100:.1f}%)")
        
        # Check empty strings
        empty_text = (df[self.text_column].str.strip() == '').sum()
        validation['empty_text'] = empty_text
        print(f"  - Empty text: {empty_text}")
        
        # Check duplicates
        duplicates = df.duplicated(subset=[self.text_column]).sum()
        validation['duplicates'] = duplicates
        print(f"\nDuplicate emails: {duplicates}")
        
        # Check class distribution
        class_counts = df[self.label_column].value_counts()
        validation['class_distribution'] = class_counts.to_dict()
        
        print(f"\nClass distribution:")
        for category, count in class_counts.items():
            percentage = count / total_records * 100
            print(f"  - {category}: {count} ({percentage:.1f}%)")
        
        # Check class balance
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        validation['imbalance_ratio'] = imbalance_ratio
        
        print(f"\nClass balance:")
        print(f"  - Most common: {class_counts.index[0]} ({max_count} samples)")
        print(f"  - Least common: {class_counts.index[-1]} ({min_count} samples)")
        print(f"  - Imbalance ratio: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio > 3:
            print(f"  ⚠ WARNING: Significant class imbalance detected!")
            print(f"    Consider using SMOTE or class weights.")
        
        # Check text lengths
        text_lengths = df[self.text_column].str.len()
        validation['avg_text_length'] = text_lengths.mean()
        validation['min_text_length'] = text_lengths.min()
        validation['max_text_length'] = text_lengths.max()
        
        print(f"\nText statistics:")
        print(f"  - Average length: {text_lengths.mean():.0f} characters")
        print(f"  - Min length: {text_lengths.min()}")
        print(f"  - Max length: {text_lengths.max()}")
        
        # Overall health
        print("\n" + "=" * 60)
        issues = []
        if missing_text > 0 or missing_labels > 0:
            issues.append("Missing values detected")
        if empty_text > 0:
            issues.append("Empty text fields detected")
        if duplicates > total_records * 0.1:
            issues.append("High duplicate rate")
        if imbalance_ratio > 5:
            issues.append("Severe class imbalance")
        
        if issues:
            print("⚠ ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
            validation['status'] = 'warning'
        else:
            print("✓ DATA VALIDATION PASSED")
            validation['status'] = 'ok'
        
        print("=" * 60 + "\n")
        
        return validation
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing invalid records.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        original_count = len(df)
        
        # Remove missing values
        df = df.dropna(subset=[self.text_column, self.label_column])
        
        # Remove empty strings
        df = df[df[self.text_column].str.strip() != '']
        
        # Remove duplicates
        df = df.drop_duplicates(subset=[self.text_column])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        removed = original_count - len(df)
        if removed > 0:
            print(f"Cleaned data: removed {removed} invalid records")
            print(f"Remaining records: {len(df)}")
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Dict[str, Tuple[pd.Series, pd.Series]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            test_size: Proportion for test set (0.2 = 20%)
            val_size: Proportion for validation set (0.1 = 10%)
            random_state: Random seed for reproducibility
            stratify: Maintain class distribution in splits
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
            Each split is tuple of (texts, labels)
        """
        X = df[self.text_column]
        y = df[self.label_column]
        
        stratify_labels = y if stratify else None

        # If there are any classes with fewer than 2 samples, stratified splitting is not possible.
        if stratify_labels is not None:
            label_counts = y.value_counts()
            if label_counts.min() < 2:
                print("⚠ WARNING: Stratified split disabled because at least one class has fewer than 2 samples.")
                stratify_labels = None

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels
        )

        # Second split: separate validation from training
        # Adjust val_size relative to remaining data
        val_size_adjusted = val_size / (1 - test_size)

        stratify_temp = y_temp if stratify and stratify_labels is not None else None
        if stratify_temp is not None:
            temp_label_counts = y_temp.value_counts()
            if temp_label_counts.min() < 2:
                print("⚠ WARNING: Stratified split disabled for validation set because at least one class has fewer than 2 samples.")
                stratify_temp = None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_temp
        )
        
        print("\n" + "=" * 60)
        print("DATA SPLIT SUMMARY")
        print("=" * 60)
        print(f"Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
        print(f"Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
        print("=" * 60 + "\n")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def get_class_weights(self, labels: pd.Series) -> Dict[str, float]:
        """
        Calculate class weights for imbalanced datasets.
        
        Args:
            labels: Series of category labels
            
        Returns:
            Dictionary mapping category to weight
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        
        class_weights = dict(zip(classes, weights))
        
        print("Class weights (for imbalanced data):")
        for category, weight in class_weights.items():
            print(f"  {category}: {weight:.3f}")
        
        return class_weights
