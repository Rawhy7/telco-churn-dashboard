# pages/model_insights.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from utils.preprocessing import load_data, get_preprocessing_pipeline
from utils.visualization import plot_feature_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff

st.set_page_config(
    page_title="Model Insights - Telco Churn",
    page_icon="📈",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model():
    try:
        with open('models/churn_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please train and save the model first.")
        return None

@st.cache_resource
def load_pipeline():
    try:
        # Import the pickle fix before loading
        import utils.pickle_fix
        
        with open('models/churn_pipeline.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Pipeline file not found. Please train and save the model pipeline first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def get_data():
    return load_data()

pipeline_data = load_pipeline()

# Extract components if successfully loaded
if pipeline_data is not None:
    model = pipeline_data['model']
    preprocessing_pipeline = pipeline_data['preprocessing']
    feature_names = pipeline_data['feature_names']
else:
    st.stop()

df = get_data()

st.title("Telco Churn Model Insights")

# Check if model is loaded successfully
if model is None:
    st.stop()

# Model information
st.subheader("Model Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Type", "Logistic Regression")
    
with col2:
    st.metric("Regularization", "L2 (Ridge)")
    
with col3:
    st.metric("Regularization Strength", "C=0.01")

# For demo purposes, we'll calculate metrics on a subset of data
# In a real application, you'd use proper train/test splits
# Prepare data for model evaluation
X = df.drop('Churn', axis=1)
if 'customerID' in X.columns:
    X = X.drop('customerID', axis=1)
y = df['Churn']

# Get preprocessing pipeline
pipeline = get_preprocessing_pipeline()
X_processed = pipeline.fit_transform(X)

# Get predictions
y_pred = model.predict(X_processed)
y_pred_proba = model.predict_proba(X_processed)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Display metrics
st.subheader("Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{accuracy:.4f}")
    
with col2:
    st.metric("Precision", f"{precision:.4f}")
    
with col3:
    st.metric("Recall", f"{recall:.4f}")
    
with col4:
    st.metric("F1 Score", f"{f1:.4f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_pred)
conf_matrix_fig = ff.create_annotated_heatmap(
    z=cm, 
    x=['Predicted Stay', 'Predicted Churn'],
    y=['Actual Stay', 'Actual Churn'], 
    colorscale='Blues'
)
conf_matrix_fig.update_layout(title="Confusion Matrix")
st.plotly_chart(conf_matrix_fig)

# Feature importance
st.subheader("Feature Importance")
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
feature_importance_fig = plot_feature_importance(model, feature_names)
if feature_importance_fig:
    st.plotly_chart(feature_importance_fig, use_container_width=True)

# ROC Curve would go here in a complete implementation

# Additional Insights
st.subheader("Prediction Distribution")
prediction_df = pd.DataFrame({
    'Actual': y,
    'Predicted Probability': y_pred_proba
})

fig = px.histogram(prediction_df, x='Predicted Probability', color='Actual',
                  barmode='overlay', nbins=50,
                  labels={'Actual': 'Actual Churn'},
                  color_discrete_map={0: '#2E86C1', 1: '#E74C3C'})
fig.update_layout(title='Distribution of Churn Probabilities')
st.plotly_chart(fig, use_container_width=True)