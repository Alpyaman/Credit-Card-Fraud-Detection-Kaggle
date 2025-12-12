"""
Model training module for fraud detection.
Handles model training, hyperparameter tuning, and evaluation.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score
)
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from pathlib import Path
import json
from datetime import datetime
from preprocess import FraudPreprocessor

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionTrainer:
    """Handles training and evaluation of fraud detection models."""
    
    def __init__(self, model_dir: str = "models", random_state: int = 42):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.metrics = {}
        
    def prepare_data(self, data_path: str, test_size: float = 0.2) -> tuple:
        """
        Load and prepare data for training.
        
        Args:
            data_path: Path to CSV file
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, preprocessor)
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Initialize and fit preprocessor
        preprocessor = FraudPreprocessor()
        df_processed = preprocessor.fit_transform(df)
        
        # Save preprocessor
        preprocessor_path = self.model_dir / "preprocessor.pkl"
        preprocessor.save(str(preprocessor_path))
        
        # Split features and target
        X = df_processed.drop(columns=["Class"])
        y = df_processed["Class"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )
        
        logger.info(f"Data prepared: Train shape {X_train.shape}, Test shape {X_test.shape}")
        logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test, preprocessor
    
    def train_with_grid_search(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        param_grid: dict = None
    ):
        """
        Train model using GridSearchCV for hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Hyperparameter grid (optional)
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [5, -1],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0]
            }
        
        base_model = LGBMClassifier(random_state=self.random_state, verbose=-1)
        
        logger.info("Starting GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        logger.info(f"✅ Best Parameters: {self.best_params}")
        logger.info(f"✅ Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "timestamp": datetime.now().isoformat()
        }
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("EVALUATION METRICS")
        logger.info("="*50)
        logger.info(f"ROC-AUC Score: {self.metrics['roc_auc']:.4f}")
        logger.info(f"Precision: {self.metrics['precision']:.4f}")
        logger.info(f"Recall: {self.metrics['recall']:.4f}")
        logger.info(f"F1-Score: {self.metrics['f1_score']:.4f}")
        
        print("\n[Confusion Matrix]")
        print(confusion_matrix(y_test, y_pred))
        
        print("\n[Classification Report]")
        print(classification_report(y_test, y_pred, digits=4))
        
        # Save metrics
        metrics_path = self.model_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")
        
        return self.metrics
    
    def plot_roc_curve(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Plot and save ROC curve."""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            fpr, tpr, 
            label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})",
            linewidth=2
        )
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curve - Fraud Detection Model", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        roc_path = self.model_dir / "roc_curve.png"
        plt.savefig(roc_path, dpi=300)
        logger.info(f"ROC curve saved to {roc_path}")
        plt.close()
    
    def plot_precision_recall_curve(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Plot and save Precision-Recall curve."""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, linewidth=2, label="PR Curve")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title("Precision-Recall Curve - Fraud Detection Model", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        pr_path = self.model_dir / "pr_curve.png"
        plt.savefig(pr_path, dpi=300)
        logger.info(f"PR curve saved to {pr_path}")
        plt.close()
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance."""
        if self.model is None:
            raise ValueError("Model must be trained before plotting feature importance")
        
        importance = pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance, x='importance', y='feature', palette='viridis')
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.title(f"Top {top_n} Feature Importances", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        importance_path = self.model_dir / "feature_importance.png"
        plt.savefig(importance_path, dpi=300)
        logger.info(f"Feature importance plot saved to {importance_path}")
        plt.close()
    
    def save_model(self, model_name: str = "fraud_model.pkl"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = self.model_dir / model_name
        joblib.dump(self.model, model_path)
        logger.info(f"✅ Model saved to {model_path}")
        
        # Also save metadata
        metadata = {
            "model_name": model_name,
            "best_params": self.best_params,
            "metrics": self.metrics,
            "trained_at": datetime.now().isoformat()
        }
        
        metadata_path = self.model_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Model metadata saved to {metadata_path}")
    
    @staticmethod
    def load_model(model_path: str):
        """Load a trained model."""
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model


def main():
    """Main training pipeline."""
    # Configuration
    DATA_PATH = "C:/Users/alpyaman/Desktop/Projects/Credit-Card-Fraud-Detection-Kaggle/data/creditcard.csv"
    MODEL_DIR = "models"
    
    # Initialize trainer
    trainer = FraudDetectionTrainer(model_dir=MODEL_DIR)
    
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor = trainer.prepare_data(DATA_PATH)
    
    # Train model
    trainer.train_with_grid_search(X_train, y_train)
    
    # Evaluate
    trainer.evaluate(X_test, y_test)
    
    # Create visualizations
    trainer.plot_roc_curve(X_test, y_test)
    trainer.plot_precision_recall_curve(X_test, y_test)
    trainer.plot_feature_importance()
    
    # Save model
    trainer.save_model()
    
    logger.info("\n🎉 Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
