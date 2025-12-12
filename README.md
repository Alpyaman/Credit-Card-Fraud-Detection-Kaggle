# 🛡️ Production-Ready Credit Card Fraud Detection System

![MLOps Pipeline](https://img.shields.io/badge/MLOps-Production--Ready-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-yellow)
![Cloud](https://img.shields.io/badge/Cloud-AWS%20%7C%20GCP-orange)
![Tests](https://img.shields.io/badge/Tests-28%2F28%20Passing-success)

A complete end-to-end MLOps pipeline for credit card fraud detection, featuring containerization, automated CI/CD, model monitoring, and cloud deployment capabilities.

---

## 🎯 Project Overview

This project demonstrates a **production-grade MLOps system** with:

- ✅ **Trained Model** (ROC-AUC: 0.9508)
- ✅ **FastAPI REST API** for real-time predictions
- ✅ **Containerization** with Docker
- ✅ **CI/CD Pipeline** with GitHub Actions
- ✅ **Model Monitoring** dashboard (Python 3.11 compatible)
- ✅ **Cloud Deployment** configs (AWS ECS, GCP Cloud Run)
- ✅ **Comprehensive Testing** (28/28 tests passing)
- ✅ **Complete Documentation**

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection-Kaggle/
│
├── api/                          # FastAPI application
│   ├── __init__.py
│   └── main.py                   # API endpoints and model serving
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── preprocess.py             # Data preprocessing module
│   ├── train.py                  # Model training pipeline
|   ├── download_data.py          # Dataset download script
|   └── create_monitoring_data.py # Create monitoring data

│
├── monitoring/                   # Monitoring dashboards
│   ├── dashboard.py              # Evidently AI dashboard
│   ├── dashboard_simple.py       # Simplified dashboard (Python 3.11)
│   └── Dockerfile                # Monitoring container
│
├── tests/                        # Test suite (28 tests)
│   ├── __init__.py
│   ├── test_preprocess.py        # Preprocessing tests
│   ├── test_api.py               # API tests
|   └── test_api_local.py         # API testing script

│
├── cloud/                        # Cloud deployment configs
│   ├── aws-ecs.tf                # AWS ECS Terraform
│   └── gcp-cloudrun.tf           # GCP Cloud Run Terraform
│
├── .github/workflows/            # CI/CD pipelines
│   └── ci-cd.yml                 # GitHub Actions workflow
│
├── models/                       # Trained models
│   ├── fraud_model.pkl           # Trained LightGBM model
│   ├── preprocessor.pkl          # Fitted preprocessor
│   ├── model_metadata.json       # Model metadata & metrics
│   └── *.png                     # Visualization plots
│
├── data/                         # Data directory
│   ├── creditcard.csv            # Main dataset (143MB)
│   ├── reference_data.csv        # Reference data for monitoring
│   ├── production_predictions.csv # Sample production data
│   └── test_sample.csv           # Test samples
│
├── Dockerfile                    # Main application container
├── docker-compose.yml            # Multi-container orchestration
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
│
├── README.md                     # This file
├── QUICKSTART.md                 # Quick start guide
└── WHATS_NEXT.md                 # Learning roadmap
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Alpyaman/Credit-Card-Fraud-Detection-Kaggle.git
cd Credit-Card-Fraud-Detection-Kaggle
```

### 2. Setup Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### 3. Download Data and Train Model

```bash
# Download the dataset
python download_data.py

# Train the model
python src/train.py

# Create monitoring data
python create_monitoring_data.py
```

**Output:**
- `models/fraud_model.pkl` - Trained model (ROC-AUC: 0.9508)
- `models/preprocessor.pkl` - Data preprocessor
- `models/model_metadata.json` - Metrics and metadata
- Visualization plots (ROC curve, PR curve, feature importance)

### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Expected: All 28 tests pass ✅
```

### 5. Start the API

```bash
# Start FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- http://localhost:8000 - Root
- http://localhost:8000/docs - Interactive API docs
- http://localhost:8000/health - Health check
- http://localhost:8000/predict - Single prediction
- http://localhost:8000/predict/batch - Batch predictions
- http://localhost:8000/model/info - Model information

### 6. Test the API

```bash
# In another terminal, run the test script
python test_api_local.py

# Expected: All 5 API tests pass ✅
```

### 7. Run Monitoring Dashboard

```bash
# Start the monitoring dashboard
streamlit run monitoring/dashboard_simple.py
```

Visit: http://localhost:8501

**Dashboard Features:**
- Overview: Model metrics and data summary
- Data Drift: Drift detection with statistical tests
- Model Performance: Performance metrics and trends
- Predictions: Recent prediction analysis

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step getting started guide
- **[WHATS_NEXT.md](WHATS_NEXT.md)** - Next steps and learning path

---

## 🛠️ Technology Stack

**Machine Learning:**
- LightGBM - Gradient boosting framework
- Scikit-learn - Preprocessing and metrics
- Pandas, NumPy - Data manipulation

**API & Web:**
- FastAPI - REST API framework
- Uvicorn - ASGI server
- Pydantic - Data validation
- Streamlit - Monitoring dashboard

**DevOps & MLOps:**
- Docker - Containerization
- Docker Compose - Multi-container orchestration
- GitHub Actions - CI/CD
- Pytest - Testing framework

**Cloud & Infrastructure:**
- Terraform - Infrastructure as Code
- AWS ECS - Container orchestration
- GCP Cloud Run - Serverless containers

**Monitoring:**
- Plotly - Interactive visualizations
- SciPy - Statistical analysis
- Custom drift detection

---

## 📈 Project Highlights

✅ **End-to-End Pipeline**: Complete flow from data → model → API → deployment  
✅ **Production Ready**: Tests, monitoring, documentation all in place  
✅ **Cloud Native**: Containerized and ready for any cloud platform  
✅ **Automated**: CI/CD handles testing, building, and deployment  
✅ **Maintainable**: Well-structured, tested, and documented code  
✅ **Scalable**: Can handle production workloads with proper deployment  

---

## 🎓 Learning Outcomes

This project demonstrates:

- ✅ Machine Learning model development and evaluation
- ✅ REST API design and implementation
- ✅ Containerization with Docker
- ✅ CI/CD pipeline setup with GitHub Actions
- ✅ Model monitoring and drift detection
- ✅ Cloud deployment (AWS & GCP)
- ✅ Infrastructure as Code with Terraform
- ✅ Comprehensive testing strategies
- ✅ Production-ready MLOps practices

### Option 1: Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - Monitoring: http://localhost:8501

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 2: Docker Only

```bash
# Build the image
docker build -t fraud-detection-api:latest .

# Run the container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name fraud-detection \
  fraud-detection-api:latest

# Check logs
docker logs fraud-detection

# Stop container
docker stop fraud-detection
```

---

## 🧪 Testing

### Run All Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=api --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

**Test Coverage:**
- ✅ 14 API tests (endpoints, validation, error handling)
- ✅ 14 preprocessing tests (transformations, edge cases)
- ✅ All 28 tests passing

---

## 📊 Model Performance

**Training Results:**
- **ROC-AUC Score**: 0.9508
- **Precision**: 0.8537
- **Recall**: 0.7143
- **F1-Score**: 0.7778

**Model Details:**
- Algorithm: LightGBM Classifier
- Features: 34 (28 original + 5 outlier flags + scaled amount)
- Best Parameters:
  - learning_rate: 0.05
  - max_depth: 5
  - n_estimators: 100
  - subsample: 0.8

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is for educational purposes as part of MLOps learning.

---

## 📦 Support

For questions or issues:
- Check the documentation files (QUICKSTART.md, etc.)
- Open an issue on GitHub
- Review [WHATS_NEXT.md](WHATS_NEXT.md) for common scenarios

---

## 🎉 Acknowledgments

- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle
- Inspired by production MLOps best practices

---

**Built with ❤️ for learning MLOps**

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

---

## 🙏 Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Monitoring: [Evidently AI](https://www.evidentlyai.com/)
- Framework: [FastAPI](https://fastapi.tiangolo.com/)

---

## 📞 Support

For issues and questions:
- Create an [Issue](https://github.com/Alpyaman/Credit-Card-Fraud-Detection-Kaggle/issues)
- Email: alpyaman3@gmail.com

---

**Ready to deploy to production! 🚀**
