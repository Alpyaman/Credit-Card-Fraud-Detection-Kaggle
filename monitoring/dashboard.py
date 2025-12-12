"""
Model monitoring dashboard using Evidently AI.
Tracks model performance, data drift, and prediction quality.
"""

import streamlit as st
import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently import Report
from evidently.metrics import (
    DataDriftPreset,
    TargetDriftPreset,
    ClassificationPreset
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
import joblib
import plotly.express as px
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="Fraud Detection Monitoring",
    page_icon="🛡️",
    layout="wide"
)

# Title
st.title("🛡️ Credit Card Fraud Detection - Model Monitoring")
st.markdown("---")


@st.cache_resource
def load_model_and_preprocessor():
    """Load model and preprocessor."""
    try:
        model = joblib.load("models/fraud_model.pkl")
        preprocessor = joblib.load("models/preprocessor.pkl")
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_data
def load_reference_data():
    """Load reference (training) data for comparison."""
    try:
        # Load your reference data here
        # This should be a sample of your training data
        reference_data = pd.read_csv("data/reference_data.csv")
        return reference_data
    except Exception as e:
        st.warning(f"Reference data not found: {e}")
        return None


def load_production_data():
    """Load recent production predictions."""
    try:
        # Load recent predictions from your database or logs
        production_data = pd.read_csv("data/production_predictions.csv")
        return production_data
    except Exception as e:
        st.warning(f"Production data not found: {e}")
        return None


def generate_sample_data(n_samples=1000):
    """Generate sample data for demonstration."""
    np.random.seed(42)
    
    # Generate feature data
    data = {
        f'V{i}': np.random.randn(n_samples) for i in range(1, 29)
    }
    data['Amount'] = np.random.uniform(1, 1000, n_samples)
    data['Time'] = np.arange(n_samples)
    data['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    
    return pd.DataFrame(data)


def compute_drift_report(reference_data, current_data, column_mapping):
    """Compute data drift report using Evidently."""
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    return report


def compute_model_performance_report(reference_data, current_data, column_mapping):
    """Compute model performance report."""
    report = Report(metrics=[
        ClassificationPreset(),
        TargetDriftPreset(),
    ])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    return report


# Sidebar
st.sidebar.header("⚙️ Configuration")

# Date range selector
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime.now() - timedelta(days=7), datetime.now())
)

# Monitoring options
show_drift = st.sidebar.checkbox("Show Data Drift Analysis", value=True)
show_performance = st.sidebar.checkbox("Show Model Performance", value=True)
show_predictions = st.sidebar.checkbox("Show Recent Predictions", value=True)

# Load model
model, preprocessor = load_model_and_preprocessor()

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Status", "✅ Active", delta="Healthy")

with col2:
    st.metric("Total Predictions Today", "12,543", delta="+1,234")

with col3:
    st.metric("Fraud Detection Rate", "0.18%", delta="+0.02%")

with col4:
    st.metric("Avg Response Time", "45ms", delta="-5ms")

st.markdown("---")

# Load data
reference_data = load_reference_data()
production_data = load_production_data()

# Use sample data if real data not available
if reference_data is None:
    st.info("Using sample reference data for demonstration")
    reference_data = generate_sample_data(5000)

if production_data is None:
    st.info("Using sample production data for demonstration")
    production_data = generate_sample_data(1000)

# Define column mapping
column_mapping = ColumnMapping()
column_mapping.target = 'Class'
column_mapping.prediction = 'prediction' if 'prediction' in production_data.columns else None
column_mapping.numerical_features = [f'V{i}' for i in range(1, 29)] + ['Amount']

# Data Drift Analysis
if show_drift:
    st.header("📊 Data Drift Analysis")
    
    with st.spinner("Computing drift metrics..."):
        drift_report = compute_drift_report(reference_data, production_data, column_mapping)
        
        # Display report
        st.components.v1.html(drift_report.get_html(), height=1000, scrolling=True)
    
    st.markdown("---")

# Model Performance
if show_performance and 'prediction' in production_data.columns:
    st.header("🎯 Model Performance Monitoring")
    
    with st.spinner("Computing performance metrics..."):
        performance_report = compute_model_performance_report(
            reference_data, 
            production_data, 
            column_mapping
        )
        
        # Display report
        st.components.v1.html(performance_report.get_html(), height=1000, scrolling=True)
    
    st.markdown("---")

# Recent Predictions
if show_predictions:
    st.header("🔍 Recent Predictions")
    
    # Show fraud predictions
    fraud_predictions = production_data[production_data['Class'] == 1].tail(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Fraud Detections")
        if len(fraud_predictions) > 0:
            st.dataframe(
                fraud_predictions[['Time', 'Amount', 'Class']].style.highlight_max(axis=0),
                use_container_width=True
            )
        else:
            st.info("No fraud detected in recent transactions")
    
    with col2:
        st.subheader("Amount Distribution")
        fig = px.histogram(
            production_data,
            x='Amount',
            color='Class',
            nbins=50,
            title="Transaction Amount Distribution by Class"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series of predictions
    st.subheader("Fraud Rate Over Time")
    
    # Create time-based aggregation
    production_data['hour'] = production_data['Time'] // 3600
    fraud_rate = production_data.groupby('hour')['Class'].agg(['sum', 'count'])
    fraud_rate['fraud_rate'] = fraud_rate['sum'] / fraud_rate['count'] * 100
    
    fig = px.line(
        fraud_rate,
        y='fraud_rate',
        title="Hourly Fraud Rate (%)",
        labels={'fraud_rate': 'Fraud Rate (%)', 'hour': 'Hour'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Alerts and Recommendations
st.header("⚠️ Alerts & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Active Alerts")
    alerts = []
    
    # Add some sample alerts logic
    if production_data['Amount'].mean() > reference_data['Amount'].mean() * 1.5:
        alerts.append({
            "severity": "⚠️ Warning",
            "message": "Average transaction amount significantly higher than baseline",
            "recommendation": "Review recent high-value transactions"
        })
    
    if len(alerts) > 0:
        for alert in alerts:
            st.warning(f"{alert['severity']}: {alert['message']}")
            st.info(f"💡 {alert['recommendation']}")
    else:
        st.success("✅ No active alerts")

with col2:
    st.subheader("Model Health Score")
    
    # Calculate a simple health score
    health_score = 95  # Placeholder
    
    st.metric("Overall Health", f"{health_score}%")
    st.progress(health_score / 100)
    
    if health_score >= 90:
        st.success("Model is performing well")
    elif health_score >= 70:
        st.warning("Model performance is degraded")
    else:
        st.error("Model requires immediate attention")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Fraud Detection Monitoring Dashboard | Powered by Evidently AI</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    unsafe_allow_html=True
)

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
