"""
Unit tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
import pandas as pd
import numpy as np


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_transaction():
    """Create a sample transaction for testing."""
    return {
        "Time": 0.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62
    }


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert "endpoints" in data
        assert data["status"] == "running"


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestModelInfo:
    """Test model info endpoint."""
    
    def test_model_info(self, client):
        """Test model info endpoint returns model details."""
        response = client.get("/model/info")
        
        # Should return 200 if model loaded, 503 if not
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_type" in data


class TestPredictionEndpoint:
    """Test single prediction endpoint."""
    
    def test_predict_with_valid_transaction(self, client, sample_transaction):
        """Test prediction with valid transaction data."""
        response = client.post("/predict", json=sample_transaction)
        
        # May return 503 if model not loaded in test environment
        if response.status_code == 200:
            data = response.json()
            assert "is_fraud" in data
            assert "fraud_probability" in data
            assert "transaction_id" in data
            assert "timestamp" in data
            assert isinstance(data["is_fraud"], bool)
            assert 0 <= data["fraud_probability"] <= 1
    
    def test_predict_with_missing_fields(self, client):
        """Test prediction fails with missing required fields."""
        incomplete_transaction = {
            "Time": 0.0,
            "V1": 1.0,
            "Amount": 100.0
            # Missing many required fields
        }
        
        response = client.post("/predict", json=incomplete_transaction)
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_invalid_types(self, client, sample_transaction):
        """Test prediction fails with invalid data types."""
        invalid_transaction = sample_transaction.copy()
        invalid_transaction["Amount"] = "not_a_number"
        
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Test batch prediction endpoint."""
    
    def test_batch_predict_with_valid_transactions(self, client, sample_transaction):
        """Test batch prediction with valid transactions."""
        batch_request = {
            "transactions": [
                sample_transaction,
                sample_transaction.copy()
            ]
        }
        
        response = client.post("/predict/batch", json=batch_request)
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_transactions" in data
            assert "fraud_count" in data
            assert "timestamp" in data
            assert len(data["predictions"]) == 2
            assert data["total_transactions"] == 2
    
    def test_batch_predict_with_empty_list(self, client):
        """Test batch prediction with empty transaction list."""
        batch_request = {"transactions": []}
        
        response = client.post("/predict/batch", json=batch_request)
        # Should handle gracefully - can be 200, 422, or 503 if model not loaded
        assert response.status_code in [200, 422, 503]
    
    def test_batch_predict_with_large_batch(self, client, sample_transaction):
        """Test batch prediction with large number of transactions."""
        batch_request = {
            "transactions": [sample_transaction.copy() for _ in range(100)]
        }
        
        response = client.post("/predict/batch", json=batch_request)
        
        if response.status_code == 200:
            data = response.json()
            assert data["total_transactions"] == 100


class TestErrorHandling:
    """Test error handling."""
    
    def test_nonexistent_endpoint(self, client):
        """Test request to non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_wrong_http_method(self, client, sample_transaction):
        """Test using wrong HTTP method."""
        # GET doesn't support json parameter, use params or no data
        response = client.get("/predict")
        assert response.status_code == 405  # Method not allowed


class TestDataValidation:
    """Test input data validation."""
    
    def test_negative_amount(self, client, sample_transaction):
        """Test prediction with negative amount."""
        transaction = sample_transaction.copy()
        transaction["Amount"] = -100.0
        
        response = client.post("/predict", json=transaction)
        # Should still process (some refunds might be negative)
        assert response.status_code in [200, 503]
    
    def test_extreme_values(self, client, sample_transaction):
        """Test prediction with extreme values."""
        transaction = sample_transaction.copy()
        transaction["Amount"] = 1e10
        transaction["V1"] = 1e6
        
        response = client.post("/predict", json=transaction)
        assert response.status_code in [200, 503]
    
    def test_zero_amount(self, client, sample_transaction):
        """Test prediction with zero amount."""
        transaction = sample_transaction.copy()
        transaction["Amount"] = 0.0
        
        response = client.post("/predict", json=transaction)
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
