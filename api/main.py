"""
FastAPI application for fraud detection model serving.
Provides REST API endpoints for predictions and health checks.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
import logging
from typing import List
from datetime import datetime
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Production-ready ML API for detecting fraudulent credit card transactions",
    version="1.0.0"
)

# Global variables for model and preprocessor
MODEL = None
PREPROCESSOR = None
MODEL_METADATA = {}


class Transaction(BaseModel):
    """Single transaction data model."""
    Time: float = Field(..., description="Seconds elapsed between this transaction and first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")
    
    class Config:
        schema_extra = {
            "example": {
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
        }


class BatchTransactions(BaseModel):
    """Batch of transactions for prediction."""
    transactions: List[Transaction]


class PredictionResponse(BaseModel):
    """Prediction response model."""
    is_fraud: bool
    fraud_probability: float
    transaction_id: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_count: int
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load model and preprocessor on startup."""
    global MODEL, PREPROCESSOR, MODEL_METADATA
    
    try:
        model_path = Path("models/fraud_model.pkl")
        preprocessor_path = Path("models/preprocessor.pkl")
        
        if not model_path.exists():
            logger.warning("Model file not found, using fallback path")
            model_path = Path("best_model.pkl")
        
        MODEL = joblib.load(model_path)
        logger.info(f"✅ Model loaded from {model_path}")
        
        if preprocessor_path.exists():
            PREPROCESSOR = joblib.load(preprocessor_path)
            logger.info(f"✅ Preprocessor loaded from {preprocessor_path}")
        else:
            logger.warning("Preprocessor not found, will use basic preprocessing")
        
        # Load metadata if available
        metadata_path = Path("models/model_metadata.json")
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                MODEL_METADATA = json.load(f)
            logger.info("✅ Model metadata loaded")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Single transaction prediction",
            "/predict/batch": "Batch transaction prediction",
            "/model/info": "Model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = MODEL is not None
    preprocessor_loaded = PREPROCESSOR is not None
    
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def model_info():
    """Get model information and metadata."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "LightGBM Classifier",
        "features": MODEL.feature_name_ if hasattr(MODEL, 'feature_name_') else "N/A",
        "metadata": MODEL_METADATA,
        "preprocessor_loaded": PREPROCESSOR is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(transaction: Transaction):
    """
    Predict whether a single transaction is fraudulent.
    
    Args:
        transaction: Transaction data
        
    Returns:
        Prediction response with fraud probability
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # Preprocess if preprocessor available
        if PREPROCESSOR is not None:
            df_processed = PREPROCESSOR.transform(df)
        else:
            # Basic preprocessing fallback
            if 'Time' in df.columns:
                df.drop(columns=['Time'], inplace=True)
            df_processed = df
        
        # Make prediction
        prediction = MODEL.predict(df_processed)[0]
        probability = MODEL.predict_proba(df_processed)[0, 1]
        
        return PredictionResponse(
            is_fraud=bool(prediction),
            fraud_probability=float(probability),
            transaction_id=f"txn_{datetime.now().timestamp()}",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchTransactions):
    """
    Predict fraud for a batch of transactions.
    
    Args:
        batch: Batch of transactions
        
    Returns:
        Batch prediction response
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([t.dict() for t in batch.transactions])
        
        # Preprocess
        if PREPROCESSOR is not None:
            df_processed = PREPROCESSOR.transform(df)
        else:
            if 'Time' in df.columns:
                df.drop(columns=['Time'], inplace=True)
            df_processed = df
        
        # Make predictions
        predictions = MODEL.predict(df_processed)
        probabilities = MODEL.predict_proba(df_processed)[:, 1]
        
        # Create response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append(
                PredictionResponse(
                    is_fraud=bool(pred),
                    fraud_probability=float(prob),
                    transaction_id=f"txn_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now().isoformat()
                )
            )
        
        return BatchPredictionResponse(
            predictions=results,
            total_transactions=len(results),
            fraud_count=int(sum(predictions)),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
