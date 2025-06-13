# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import warnings
from preprocess import preprocess_data
warnings.filterwarnings('ignore')
# 1. Load + preprocess
df = pd.read_csv("C:/Users/alpya/Documents/cc_fraud_detection/data/creditcard.csv")
df = preprocess_data(df)

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Define model and hyperparameter grid
model = LGBMClassifier(random_state=42, verbose=-1)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, -1],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=1
)

# 3. Fit with GridSearchCV
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\n✅ Best Params:", grid_search.best_params_)
print("✅ Best AUC Score (CV):", grid_search.best_score_)

# 4. Evaluate on test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\n[Confusion Matrix]")
print(confusion_matrix(y_test, y_pred))
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=4))

# 5. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label="ROC Curve (AUC = %.4f)" % roc_auc_score(y_test, y_pred_proba))
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# 6. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("pr_curve.png")
plt.show()

# 7. Save model
joblib.dump(best_model, "best_model.pkl")
print("\n✅ Model saved as best_model.pkl")

# 8. Create Kaggle submission
test_ids = df.loc[X_test.index, :].copy()
submission = pd.DataFrame({
    "TransactionID": test_ids.index,
    "Class": y_pred
})
submission.to_csv("submission.csv", index=False)
print("✅ Kaggle submission saved as submission.csv")
