"""
Script to download the Credit Card Fraud Detection dataset from Kaggle.
Requires Kaggle API credentials.
"""

import os
from pathlib import Path
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_kaggle_dataset():
    """Download dataset using Kaggle API."""
    try:
        import kaggle
        
        # Create data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        logger.info("Downloading Credit Card Fraud Detection dataset from Kaggle...")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path=str(data_dir),
            unzip=True
        )
        
        logger.info(f"✅ Dataset downloaded successfully to {data_dir}")
        
        # Check if file exists
        csv_file = data_dir / "creditcard.csv"
        if csv_file.exists():
            logger.info(f"✅ Found creditcard.csv ({csv_file.stat().st_size / 1024 / 1024:.2f} MB)")
            return True
        else:
            logger.error("❌ creditcard.csv not found after download")
            return False
            
    except ImportError:
        logger.error("❌ Kaggle API not installed. Install with: pip install kaggle")
        logger.info("\nAlternative: Manual download instructions:")
        logger.info("1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        logger.info("2. Download the dataset manually")
        logger.info("3. Extract creditcard.csv to the 'data' folder")
        return False
    
    except Exception as e:
        logger.error(f"❌ Error downloading dataset: {e}")
        logger.info("\nTo use Kaggle API:")
        logger.info("1. Create a Kaggle account at https://www.kaggle.com")
        logger.info("2. Go to Account settings -> API -> Create New API Token")
        logger.info("3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
        logger.info("\nOr download manually from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return False


def create_sample_data():
    """Create a small sample dataset for testing if full dataset unavailable."""
    import pandas as pd
    import numpy as np
    
    logger.info("Creating sample dataset for testing...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate synthetic data with similar structure
    n_samples = 10000
    np.random.seed(42)
    
    data = {
        'Time': np.arange(n_samples),
        'Amount': np.random.uniform(1, 1000, n_samples)
    }
    
    # Generate V1-V28 features (PCA components)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    # Create imbalanced target (0.2% fraud)
    data['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    sample_file = data_dir / "creditcard_sample.csv"
    df.to_csv(sample_file, index=False)
    
    logger.info(f"✅ Sample dataset created: {sample_file}")
    logger.info(f"   - Shape: {df.shape}")
    logger.info(f"   - Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    
    return sample_file


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Credit Card Fraud Detection - Data Download")
    logger.info("="*60)
    
    # Try to download real dataset
    success = download_kaggle_dataset()
    
    if not success:
        logger.info("\n" + "="*60)
        logger.info("Creating sample dataset for testing...")
        logger.info("="*60)
        create_sample_data()
        logger.info("\n Using sample data. For production, download the full dataset.")
