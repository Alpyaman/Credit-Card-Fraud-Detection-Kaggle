# 🚀 Quick Start Guide - Credit Card Fraud Detection MLOps

This guide will help you get the fraud detection system running locally in minutes.

## Prerequisites

- Python 3.10 or higher
- Docker (optional, for containerization)
- Git

## 📦 Step 1: Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/Alpyaman/Credit-Card-Fraud-Detection-Kaggle.git
cd Credit-Card-Fraud-Detection-Kaggle

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

## 📊 Step 2: Download Data and Train Model

```bash
# Download the dataset (requires Kaggle API or manual download)
python src/download_data.py

# Train the model
python src/train.py

# Create monitoring reference data
python src/create_monitoring_data.py
```

**Model Training Output:**
- `models/fraud_model.pkl` - Trained LightGBM model
- `models/preprocessor.pkl` - Data preprocessor
- `models/model_metadata.json` - Model metrics and metadata
- `models/*.png` - Visualization plots

## 🧪 Step 3: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=api --cov-report=html
```

Expected: All 28 tests should pass ✅

## 🔥 Step 4: Start the API

```bash
# Start the FastAPI server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @- << EOF
{
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
EOF
```

## 📊 Step 5: Run Monitoring Dashboard

```bash
# Start the Evidently monitoring dashboard
streamlit run monitoring/dashboard.py --server.port 8501
```

Dashboard available at: http://localhost:8501

## 🐳 Step 6: Docker Deployment (Optional)

### Build and Run with Docker

```bash
# Build the Docker image
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

### Or use Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

This starts:
- **API** on port 8000
- **Monitoring Dashboard** on port 8501

## 🌥️ Step 7: Cloud Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed cloud deployment instructions:

- **AWS ECS**: Terraform configuration in `cloud/aws-ecs.tf`
- **GCP Cloud Run**: Terraform configuration in `cloud/gcp-cloudrun.tf`

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline (`.github/workflows/ci-cd.yml`) automatically:

1. **Tests**: Runs pytest on every push
2. **Linting**: Checks code quality with flake8
3. **Docker Build**: Builds and pushes Docker images
4. **Model Retraining**: Triggered by `[retrain]` in commit message
5. **Deployment**: Deploys to cloud platforms

### Setup GitHub Secrets

Add these secrets to your GitHub repository:

```
DOCKER_USERNAME
DOCKER_PASSWORD
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
GCP_SERVICE_ACCOUNT_KEY
```

See [GITHUB_SECRETS.md](GITHUB_SECRETS.md) for details.

## 📈 Next Steps

1. **Customize the Model**: Modify `src/train.py` to tune hyperparameters
2. **Add Features**: Extend preprocessing in `src/preprocess.py`
3. **Enhance API**: Add new endpoints in `api/main.py`
4. **Monitor Performance**: Use the monitoring dashboard to track drift
5. **Scale**: Deploy to cloud platforms for production use

## 🆘 Troubleshooting

### Model not loading
```bash
# Check if model files exist
ls models/

# Retrain if needed
python src/train.py
```

### Tests failing
```bash
# Reinstall dependencies
pip install -r requirements.txt
pip install -e .

# Run specific test
pytest tests/test_api.py::TestHealthCheck -v
```

### Docker build fails
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t fraud-detection-api:latest .
```

### API not responding
```bash
# Check if port is in use
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/Mac

# Kill process using port
taskkill /PID <PID> /F  # Windows
kill -9 <PID>  # Linux/Mac
```

## 📚 Additional Resources

- [Full Documentation](README_MLOPS.md)
- [Deployment Guide](DEPLOYMENT.md)
- [API Documentation](http://localhost:8000/docs) (when API is running)
- [Original Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

**🎉 Congratulations!** You now have a production-ready fraud detection system with MLOps best practices!
