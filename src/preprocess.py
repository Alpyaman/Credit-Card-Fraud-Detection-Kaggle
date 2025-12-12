"""
Data preprocessing module for fraud detection.
Handles outlier detection, scaling, and feature engineering.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudPreprocessor:
    """Handles all preprocessing steps for fraud detection data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.outlier_features = ['V11', 'V3', 'V17', 'V10', 'V16']
        self.fitted = False
        
    def create_outlier_flags(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Create outlier flags for specified features using IQR method.
        
        Args:
            df: Input DataFrame
            features: List of feature names to flag outliers
            
        Returns:
            DataFrame with outlier flag columns added
        """
        df = df.copy()
        
        for feature in features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in DataFrame")
                continue
                
            q1 = df[feature].quantile(0.25)
            q3 = df[feature].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            df[f'{feature}_outlier'] = (
                (df[feature] < lower_bound) | (df[feature] > upper_bound)
            ).astype(int)
            
        return df
    
    def fit(self, df: pd.DataFrame) -> 'FraudPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self for method chaining
        """
        if 'Amount' in df.columns:
            self.scaler.fit(df[['Amount']])
            self.fitted = True
            logger.info("Preprocessor fitted successfully")
        else:
            logger.warning("Amount column not found. Scaler not fitted.")
            
        return self
    
    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Transform the data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data (affects target column)
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        
        # Create outlier flags
        df = self.create_outlier_flags(df, self.outlier_features)
        
        # Drop Time column if present
        if 'Time' in df.columns:
            df.drop(columns=['Time'], inplace=True)
        
        # Scale Amount if present
        if 'Amount' in df.columns:
            if not self.fitted:
                raise ValueError("Preprocessor must be fitted before transform")
            df['scaled_amount'] = self.scaler.transform(df[['Amount']])
            df.drop(columns=['Amount'], inplace=True)
        
        logger.info(f"Data transformed. Shape: {df.shape}")
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df, is_training=True)
    
    def save(self, filepath: str):
        """Save the preprocessor to disk."""
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FraudPreprocessor':
        """Load a preprocessor from disk."""
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    Preprocesses data without requiring fit/transform pattern.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = FraudPreprocessor()
    return preprocessor.fit_transform(df)


if __name__ == "__main__":
    # Test the preprocessor
    test_data = pd.DataFrame({
        'Time': [0, 1, 2],
        'Amount': [100, 200, 300],
        'V11': [1.5, 2.0, 10.0],
        'V3': [1.0, 2.0, 3.0],
        'V17': [0.5, 1.0, 1.5],
        'V10': [0.1, 0.2, 0.3],
        'V16': [0.01, 0.02, 0.03],
        'Class': [0, 0, 1]
    })
    
    preprocessor = FraudPreprocessor()
    transformed = preprocessor.fit_transform(test_data)
    print("Transformed data shape:", transformed.shape)
    print(transformed.head())
