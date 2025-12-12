"""
Script to create reference and production data for model monitoring.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_monitoring_data():
    """Create reference and sample production data for monitoring."""
    
    # Load the full dataset
    data_path = Path("data/creditcard.csv")
    
    if not data_path.exists():
        logger.error("creditcard.csv not found. Please download the dataset first.")
        return
    
    logger.info("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Create reference data (sample from training set)
    # Use 10,000 samples for reference
    reference_data = df.sample(n=10000, random_state=42)
    reference_path = Path("data/reference_data.csv")
    reference_data.to_csv(reference_path, index=False)
    logger.info(f"✅ Reference data saved: {reference_path}")
    logger.info(f"   - Shape: {reference_data.shape}")
    logger.info(f"   - Fraud rate: {reference_data['Class'].mean()*100:.3f}%")
    
    # Create sample production predictions
    # Simulate production data with potential drift
    production_sample = df.sample(n=1000, random_state=123).copy()
    
    # Add prediction columns (simulated)
    production_sample['prediction'] = (
        production_sample['Class'].values  # Perfect predictions for demo
    )
    production_sample['prediction_proba'] = np.where(
        production_sample['Class'] == 1,
        np.random.uniform(0.7, 0.99, len(production_sample)),
        np.random.uniform(0.01, 0.3, len(production_sample))
    )
    production_sample['timestamp'] = pd.date_range(
        start='2025-01-01', 
        periods=len(production_sample), 
        freq='5min'
    )
    
    production_path = Path("data/production_predictions.csv")
    production_sample.to_csv(production_path, index=False)
    logger.info(f"✅ Production data saved: {production_path}")
    logger.info(f"   - Shape: {production_sample.shape}")
    logger.info(f"   - Fraud rate: {production_sample['Class'].mean()*100:.3f}%")
    
    # Create a smaller test set for quick testing
    test_data = df.sample(n=100, random_state=456)
    test_path = Path("data/test_sample.csv")
    test_data.to_csv(test_path, index=False)
    logger.info(f"✅ Test sample saved: {test_path}")


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Creating Monitoring Data")
    logger.info("="*60)
    create_monitoring_data()
    logger.info("\n✅ All monitoring data files created successfully!")
