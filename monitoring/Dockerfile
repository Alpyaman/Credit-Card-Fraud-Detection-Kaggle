# Monitoring Dashboard Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    streamlit==1.28.0 \
    evidently==0.4.0 \
    pandas==2.3.0 \
    plotly==5.18.0 \
    joblib==1.5.1 \
    numpy==2.3.0

# Copy monitoring scripts
COPY dashboard.py .

# Create necessary directories
RUN mkdir -p /app/data /app/models

# Expose Streamlit port
EXPOSE 8501

# Run dashboard
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
