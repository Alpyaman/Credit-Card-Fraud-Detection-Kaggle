# 🛡️ Production-Ready Credit Card Fraud Detection System

![MLOps Pipeline](https://img.shields.io/badge/MLOps-Production--Ready-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-yellow)
![Cloud](https://img.shields.io/badge/Cloud-AWS%20%7C%20GCP-orange)

A complete end-to-end MLOps pipeline for credit card fraud detection, featuring containerization, automated CI/CD, model monitoring, and cloud deployment capabilities.

---

## 🎯 Project Overview

This project transforms a traditional ML model into a **production-grade MLOps system** with:

- ✅ **Containerized API** with Docker
- ✅ **REST API** for real-time predictions
- ✅ **CI/CD Pipeline** with GitHub Actions
- ✅ **Model Monitoring** with Evidently AI
- ✅ **Cloud Deployment** (AWS ECS, GCP Cloud Run)
- ✅ **Automated Retraining** workflows
- ✅ **Comprehensive Testing** suite

---

## 📁 Project Structure

```
fraud-detection-mlops/
│
├── api/                          # FastAPI application
│   └── main.py                   # API endpoints and model serving
│
├── src/                          # Source code
│   ├── preprocess.py             # Data preprocessing module
│   └── train.py                  # Model training pipeline
│
├── monitoring/                   # Monitoring dashboard
│   ├── dashboard.py              # Evidently AI dashboard
│   └── Dockerfile                # Monitoring container
│
├── tests/                        # Test suite
│   ├── test_preprocess.py        # Preprocessing tests
│   └── test_api.py               # API tests
│
├── cloud/                        # Cloud deployment configs
│   ├── aws-ecs.tf                # AWS ECS Terraform
│   └── gcp-cloudrun.tf           # GCP Cloud Run Terraform
│
├── .github/workflows/            # CI/CD pipelines
│   └── ci-cd.yml                 # GitHub Actions workflow
│
├── models/                       # Trained models
│   ├── fraud_model.pkl           # Trained model
│   ├── preprocessor.pkl          # Fitted preprocessor
│   └── model_metadata.json       # Model metadata
│
├── data/                         # Data directory
│
├── Dockerfile                    # Main application container
├── docker-compose.yml            # Multi-container orchestration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Credit-Card-Fraud-Detection-Kaggle
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train the Model

```bash
# Make sure you have the data file in the correct location
python src/train.py
```

### 4. Run the API Locally

```bash
# Start the FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive API documentation.

---

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t fraud-detection-api .

# Run the container
docker run -p 8000:8000 fraud-detection-api
```

### Using Docker Compose

```bash
# Start all services (API + Monitoring)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Access:
- **API**: http://localhost:8000
- **Monitoring Dashboard**: http://localhost:8501

---

## 📊 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "V1": -1.359807,
    "V2": -0.072781,
    ...
    "Amount": 149.62
  }'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {...transaction1...},
      {...transaction2...}
    ]
  }'
```

### Python Client Example

```python
import requests

# Prepare transaction data
transaction = {
    "Time": 0.0,
    "V1": -1.359807,
    # ... other features ...
    "Amount": 149.62
}

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json=transaction
)

result = response.json()
print(f"Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.2%}")
```

---

## 🔄 CI/CD Pipeline

The project includes a complete GitHub Actions workflow that:

1. **Runs Tests** - Unit tests and code quality checks
2. **Builds Docker Image** - Containerizes the application
3. **Retrains Model** - When triggered or new data added
4. **Deploys to Cloud** - AWS ECS or GCP Cloud Run

### Trigger Manual Retrain

```bash
# Commit with retrain flag
git commit -m "[retrain] Update model with new data"
git push
```

### Required GitHub Secrets

Add these secrets to your GitHub repository:

```
# Docker Hub
DOCKER_USERNAME
DOCKER_PASSWORD

# AWS
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY

# GCP
GCP_SA_KEY
GCP_PROJECT_ID
```

---

## ☁️ Cloud Deployment

### AWS ECS Deployment

1. **Setup Infrastructure**

```bash
cd cloud
terraform init
terraform plan
terraform apply
```

2. **Push Image to ECR**

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag fraud-detection-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection-api:latest
```

### GCP Cloud Run Deployment

1. **Build and Deploy**

```bash
gcloud builds submit --tag gcr.io/<project-id>/fraud-detection-api
gcloud run deploy fraud-detection-api \
  --image gcr.io/<project-id>/fraud-detection-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 📈 Model Monitoring

### Start Monitoring Dashboard

```bash
streamlit run monitoring/dashboard.py
```

The dashboard provides:

- 📊 **Data Drift Detection** - Tracks feature distribution changes
- 🎯 **Model Performance Metrics** - ROC-AUC, Precision, Recall
- 🔍 **Prediction Analysis** - Recent predictions and patterns
- ⚠️ **Alerts** - Automated alerts for performance degradation

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=src --cov=api --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_api.py -v
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.9900 |
| **Precision (Fraud)** | 85.37% |
| **Recall (Fraud)** | 71.43% |
| **F1-Score** | 77.78% |
| **Accuracy** | 99.93% |

---

## 🛠️ Development Workflow

1. **Create Feature Branch**
```bash
git checkout -b feature/your-feature
```

2. **Make Changes and Test**
```bash
pytest tests/ -v
flake8 src/ api/
```

3. **Commit and Push**
```bash
git add .
git commit -m "Description of changes"
git push origin feature/your-feature
```

4. **Create Pull Request** - CI/CD will run automatically

---

## 🔐 Security Considerations

- ✅ API authentication (implement as needed)
- ✅ HTTPS/TLS encryption in production
- ✅ Environment variable management
- ✅ Secrets management (AWS Secrets Manager, GCP Secret Manager)
- ✅ Container security scanning
- ✅ Regular dependency updates

---

## 📝 Environment Variables

Create a `.env` file for local development:

```bash
MODEL_PATH=models/fraud_model.pkl
PREPROCESSOR_PATH=models/preprocessor.pkl
LOG_LEVEL=INFO
API_KEY=your-secret-key  # If implementing authentication
```

---

## 🎓 Learning Resources

This project demonstrates:

- **MLOps Best Practices** - Production ML workflows
- **Docker & Containerization** - Application packaging
- **REST API Design** - FastAPI implementation
- **CI/CD Pipelines** - Automated testing and deployment
- **Cloud Platforms** - AWS and GCP services
- **Model Monitoring** - Drift detection and alerting

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Alp Yaman**  
Intermediate Data Scientist - AI Enthusiast

- GitHub: [@alpyaman](https://github.com/alpyaman)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Monitoring: [Evidently AI](https://www.evidentlyai.com/)
- Framework: [FastAPI](https://fastapi.tiangolo.com/)

---

## 📞 Support

For issues and questions:
- Create an [Issue](https://github.com/your-username/repo/issues)
- Email: your.email@example.com

---

**Ready to deploy to production! 🚀**
