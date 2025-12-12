"""
Model monitoring dashboard for fraud detection.
Simplified version without Evidently AI (Python 3.11 compatible).
Tracks model performance, data drift, and prediction quality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from scipy import stats

st.set_page_config(
    page_title="Fraud Detection Monitoring",
    page_icon="🛡️",
    layout="wide"
)

# Title
st.title("🛡️ Credit Card Fraud Detection - Model Monitoring")
st.markdown("---")


@st.cache_resource
def load_model_and_metadata():
    """Load model and metadata."""
    try:
        model = joblib.load("models/fraud_model.pkl")
        
        metadata = {}
        metadata_path = Path("models/model_metadata.json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, {}


@st.cache_data
def load_reference_data():
    """Load reference (training) data for comparison."""
    try:
        reference_data = pd.read_csv("data/reference_data.csv")
        return reference_data
    except Exception as e:
        st.warning(f"Reference data not found: {e}")
        return None


@st.cache_data
def load_production_data():
    """Load recent production predictions."""
    try:
        production_data = pd.read_csv("data/production_predictions.csv")
        return production_data
    except Exception as e:
        st.warning(f"Production data not found: {e}")
        return None


def calculate_drift_score(reference_col, current_col):
    """Calculate drift score using Kolmogorov-Smirnov test."""
    try:
        statistic, p_value = stats.ks_2samp(reference_col, current_col)
        return statistic, p_value
    except:
        return None, None


def plot_distribution_comparison(reference_data, current_data, column, title):
    """Plot distribution comparison between reference and current data."""
    fig = go.Figure()
    
    # Reference distribution
    fig.add_trace(go.Histogram(
        x=reference_data[column],
        name='Reference',
        opacity=0.7,
        nbinsx=50
    ))
    
    # Current distribution
    fig.add_trace(go.Histogram(
        x=current_data[column],
        name='Current',
        opacity=0.7,
        nbinsx=50
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=column,
        yaxis_title='Count',
        barmode='overlay',
        height=300
    )
    
    return fig


def main():
    # Sidebar
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Data Drift", "Model Performance", "Predictions"]
    )
    
    # Load data
    model, metadata = load_model_and_metadata()
    reference_data = load_reference_data()
    production_data = load_production_data()
    
    if page == "Overview":
        st.header("📈 Model Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Type", "LightGBM")
        
        with col2:
            if metadata and 'metrics' in metadata:
                roc_auc = metadata['metrics'].get('roc_auc', 0)
                st.metric("ROC-AUC", f"{roc_auc:.4f}")
        
        with col3:
            if metadata and 'metrics' in metadata:
                precision = metadata['metrics'].get('precision', 0)
                st.metric("Precision", f"{precision:.4f}")
        
        with col4:
            if metadata and 'metrics' in metadata:
                recall = metadata['metrics'].get('recall', 0)
                st.metric("Recall", f"{recall:.4f}")
        
        st.markdown("---")
        
        # Model Information
        if metadata:
            st.subheader("🔧 Model Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'best_params' in metadata:
                    st.json(metadata['best_params'])
            
            with col2:
                if 'metrics' in metadata:
                    st.json(metadata['metrics'])
        
        # Data Summary
        st.markdown("---")
        st.subheader("📊 Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if reference_data is not None:
                st.write("**Reference Data**")
                st.write(f"- Shape: {reference_data.shape}")
                st.write(f"- Fraud Rate: {reference_data['Class'].mean()*100:.3f}%")
                st.write(f"- Total Samples: {len(reference_data):,}")
        
        with col2:
            if production_data is not None:
                st.write("**Production Data**")
                st.write(f"- Shape: {production_data.shape}")
                st.write(f"- Fraud Rate: {production_data['Class'].mean()*100:.3f}%")
                st.write(f"- Total Samples: {len(production_data):,}")
    
    elif page == "Data Drift":
        st.header("🌊 Data Drift Analysis")
        
        if reference_data is None or production_data is None:
            st.warning("Please ensure both reference and production data are available.")
            return
        
        st.write("Analyzing drift between reference and production data...")
        
        # Select features to analyze
        feature_cols = [col for col in reference_data.columns if col.startswith('V') or col == 'Amount']
        
        # Calculate drift for each feature
        drift_results = []
        for col in feature_cols[:10]:  # Analyze first 10 features for performance
            if col in production_data.columns:
                statistic, p_value = calculate_drift_score(
                    reference_data[col].dropna(),
                    production_data[col].dropna()
                )
                if statistic is not None:
                    drift_results.append({
                        'Feature': col,
                        'KS Statistic': statistic,
                        'P-Value': p_value,
                        'Drift Detected': 'Yes' if p_value < 0.05 else 'No'
                    })
        
        drift_df = pd.DataFrame(drift_results)
        
        # Display drift summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_features = len(drift_df)
            st.metric("Features Analyzed", total_features)
        
        with col2:
            drift_detected = (drift_df['Drift Detected'] == 'Yes').sum()
            st.metric("Drift Detected", drift_detected)
        
        with col3:
            drift_pct = (drift_detected / total_features * 100) if total_features > 0 else 0
            st.metric("Drift Percentage", f"{drift_pct:.1f}%")
        
        st.markdown("---")
        
        # Drift table
        st.subheader("📋 Drift Detection Results")
        st.dataframe(drift_df, use_container_width=True)
        
        st.markdown("---")
        
        # Feature distributions
        st.subheader("📊 Feature Distribution Comparison")
        
        selected_feature = st.selectbox(
            "Select feature to visualize",
            feature_cols[:10]
        )
        
        if selected_feature:
            fig = plot_distribution_comparison(
                reference_data,
                production_data,
                selected_feature,
                f"Distribution Comparison: {selected_feature}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Performance":
        st.header("📊 Model Performance Metrics")
        
        if metadata and 'metrics' in metadata:
            metrics = metadata['metrics']
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
            
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
            
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
            
            st.markdown("---")
            
            # Performance over time (simulated)
            st.subheader("📈 Performance Trend")
            
            # Generate simulated trend data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            trend_data = pd.DataFrame({
                'Date': dates,
                'ROC-AUC': np.random.normal(metrics.get('roc_auc', 0.95), 0.01, 30),
                'Precision': np.random.normal(metrics.get('precision', 0.85), 0.02, 30),
                'Recall': np.random.normal(metrics.get('recall', 0.71), 0.02, 30)
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend_data['Date'], y=trend_data['ROC-AUC'], 
                                    mode='lines+markers', name='ROC-AUC'))
            fig.add_trace(go.Scatter(x=trend_data['Date'], y=trend_data['Precision'], 
                                    mode='lines+markers', name='Precision'))
            fig.add_trace(go.Scatter(x=trend_data['Date'], y=trend_data['Recall'], 
                                    mode='lines+markers', name='Recall'))
            
            fig.update_layout(
                title="Model Performance Over Time (Last 30 Days)",
                xaxis_title="Date",
                yaxis_title="Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Model metadata not available.")
    
    elif page == "Predictions":
        st.header("🔮 Recent Predictions")
        
        if production_data is not None:
            st.subheader("📊 Prediction Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Predictions", len(production_data))
            
            with col2:
                fraud_count = production_data['Class'].sum()
                st.metric("Fraud Detected", int(fraud_count))
            
            with col3:
                fraud_rate = production_data['Class'].mean() * 100
                st.metric("Fraud Rate", f"{fraud_rate:.3f}%")
            
            with col4:
                if 'prediction_proba' in production_data.columns:
                    avg_confidence = production_data['prediction_proba'].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            st.markdown("---")
            
            # Prediction distribution
            st.subheader("📈 Prediction Probability Distribution")
            
            if 'prediction_proba' in production_data.columns:
                fig = px.histogram(
                    production_data,
                    x='prediction_proba',
                    nbins=50,
                    title="Distribution of Fraud Probabilities",
                    labels={'prediction_proba': 'Fraud Probability'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Recent predictions table
            st.subheader("📋 Recent Predictions")
            display_cols = ['Class', 'prediction', 'prediction_proba', 'timestamp'] if 'timestamp' in production_data.columns else ['Class', 'prediction', 'prediction_proba']
            available_cols = [col for col in display_cols if col in production_data.columns]
            
            if available_cols:
                st.dataframe(
                    production_data[available_cols].head(50),
                    use_container_width=True
                )
            else:
                st.dataframe(production_data.head(50), use_container_width=True)
        else:
            st.warning("Production data not available.")


if __name__ == "__main__":
    main()
