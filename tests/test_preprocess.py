"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocess import FraudPreprocessor, preprocess_data


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Time': [0, 1, 2, 3, 4],
        'Amount': [100, 200, 300, 400, 500],
        'V1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'V3': [0.5, 1.0, 1.5, 2.0, 2.5],
        'V10': [0.1, 0.2, 0.3, 0.4, 0.5],
        'V11': [1.5, 2.0, 10.0, 3.0, 2.5],  # V11 has outlier
        'V16': [0.01, 0.02, 0.03, 0.04, 0.05],
        'V17': [0.5, 1.0, 1.5, 2.0, 2.5],
        'Class': [0, 0, 1, 0, 0]
    })


class TestFraudPreprocessor:
    """Test suite for FraudPreprocessor class."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = FraudPreprocessor()
        assert preprocessor.fitted is False
        assert len(preprocessor.outlier_features) == 5
        assert preprocessor.scaler is not None
    
    def test_fit(self, sample_data):
        """Test fitting the preprocessor."""
        preprocessor = FraudPreprocessor()
        preprocessor.fit(sample_data)
        assert preprocessor.fitted is True
    
    def test_create_outlier_flags(self, sample_data):
        """Test outlier flag creation."""
        preprocessor = FraudPreprocessor()
        result = preprocessor.create_outlier_flags(sample_data, ['V11'])
        
        assert 'V11_outlier' in result.columns
        assert result['V11_outlier'].dtype == int
        # V11 value of 10.0 should be flagged as outlier
        assert result.loc[2, 'V11_outlier'] == 1
    
    def test_transform_drops_time(self, sample_data):
        """Test that Time column is dropped during transformation."""
        preprocessor = FraudPreprocessor()
        preprocessor.fit(sample_data)
        result = preprocessor.transform(sample_data)
        
        assert 'Time' not in result.columns
    
    def test_transform_scales_amount(self, sample_data):
        """Test that Amount is scaled and original is dropped."""
        preprocessor = FraudPreprocessor()
        preprocessor.fit(sample_data)
        result = preprocessor.transform(sample_data)
        
        assert 'Amount' not in result.columns
        assert 'scaled_amount' in result.columns
        # Scaled values should have mean close to 0
        assert abs(result['scaled_amount'].mean()) < 1e-10
    
    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform without fit raises error."""
        preprocessor = FraudPreprocessor()
        
        with pytest.raises(ValueError, match="Preprocessor must be fitted"):
            preprocessor.transform(sample_data)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        preprocessor = FraudPreprocessor()
        result = preprocessor.fit_transform(sample_data)
        
        assert preprocessor.fitted is True
        assert 'scaled_amount' in result.columns
        assert 'Time' not in result.columns
    
    def test_output_shape(self, sample_data):
        """Test that output shape is correct."""
        preprocessor = FraudPreprocessor()
        result = preprocessor.fit_transform(sample_data)
        
        # Original columns + outlier flags - Time - Amount + scaled_amount
        expected_cols = len(sample_data.columns) - 2 + 1 + len(preprocessor.outlier_features)
        assert result.shape[1] == expected_cols
        assert result.shape[0] == sample_data.shape[0]
    
    def test_legacy_preprocess_data_function(self, sample_data):
        """Test legacy preprocess_data function."""
        result = preprocess_data(sample_data)
        
        assert 'scaled_amount' in result.columns
        assert 'Time' not in result.columns
        assert result.shape[0] == sample_data.shape[0]
    
    def test_missing_features_warning(self, sample_data):
        """Test handling of missing features."""
        preprocessor = FraudPreprocessor()
        # Remove a feature
        data_missing = sample_data.drop(columns=['V11'])
        preprocessor.fit(data_missing)
        
        # Should not raise error, just warning
        result = preprocessor.transform(data_missing)
        assert 'V11_outlier' not in result.columns


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        preprocessor = FraudPreprocessor()
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        preprocessor.fit(empty_df)
        assert preprocessor.fitted is False
    
    def test_single_row(self):
        """Test with single row DataFrame."""
        preprocessor = FraudPreprocessor()
        single_row = pd.DataFrame({
            'Time': [0],
            'Amount': [100],
            'V1': [1.0],
            'V3': [0.5],
            'V10': [0.1],
            'V11': [1.5],
            'V16': [0.01],
            'V17': [0.5],
            'Class': [0]
        })
        
        result = preprocessor.fit_transform(single_row)
        assert result.shape[0] == 1
    
    def test_all_same_values(self):
        """Test with all same values (no variance)."""
        preprocessor = FraudPreprocessor()
        same_values = pd.DataFrame({
            'Time': [0, 0, 0],
            'Amount': [100, 100, 100],
            'V11': [1.0, 1.0, 1.0],
            'V3': [1.0, 1.0, 1.0],
            'V10': [1.0, 1.0, 1.0],
            'V16': [1.0, 1.0, 1.0],
            'V17': [1.0, 1.0, 1.0],
            'Class': [0, 0, 0]
        })
        
        result = preprocessor.fit_transform(same_values)
        # Should handle without error
        assert result.shape[0] == 3
    
    def test_extreme_outliers(self):
        """Test with extreme outlier values."""
        preprocessor = FraudPreprocessor()
        extreme_data = pd.DataFrame({
            'Time': [0, 1, 2],
            'Amount': [100, 200, 300],
            'V11': [1.0, 2.0, 1000.0],  # Extreme outlier
            'V3': [1.0, 2.0, 3.0],
            'V10': [1.0, 2.0, 3.0],
            'V16': [1.0, 2.0, 3.0],
            'V17': [1.0, 2.0, 3.0],
            'Class': [0, 0, 1]
        })
        
        result = preprocessor.fit_transform(extreme_data)
        # Extreme value should be flagged
        assert result.loc[2, 'V11_outlier'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
