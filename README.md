# 🛡️ Credit Card Fraud Detection

This project tackles the problem of **fraud detection** using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset is highly imbalanced, with only 0.17% of transactions being fraudulent.

We implement a complete machine learning pipeline including:
- Exploratory Data Analysis (EDA)
- Outlier inspection
- Data preprocessing
- Feature engineering
- Model training with `GridSearchCV`
- Evaluation using ROC & Precision-Recall curves
- Final Kaggle submission file generation

---

## 📁 Project Structure
```
cc_fraud_detection/
│
├── preprocess.py # Data cleaning and transformation logic
├── train_model.py # Model training and evaluation pipeline
├── submission.csv # Final predictions in Kaggle format
├── best_model.pkl # Trained model saved using joblib
├── eda_for_cc_fraud.ipynb # EDA for data understanding
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## 🧪 Model & Performance

We use **LightGBM** with `GridSearchCV` to tune hyperparameters.

| Metric     | Value     |
|------------|-----------|
| Accuracy   | 99.93%    |
| Precision (Fraud) | 85.37% |
| Recall (Fraud)    | 71.43% |
| ROC AUC    | ~0.99     |

🔍 ROC and Precision-Recall curves are saved as `.png` files in the repo.

---

## 🛠️ How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
2. 
