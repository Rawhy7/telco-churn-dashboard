# pages/data_explorer.py
import streamlit as st
import pandas as pd
from utils.preprocessing import load_data
from utils.visualization import (
    plot_churn_distribution, 
    plot_numerical_features, 
    plot_categorical_features,
    correlation_heatmap
)

st.set_page_config(
    page_title="Data Explorer - Telco Churn",
    page_icon="🔍",
    layout="wide"
)

# Load data
@st.cache_data
def get_data():
    return load_data()

df = get_data()

st.title("Telco Customer Data Explorer")

# Display churn distribution
st.subheader("Customer Churn Distribution")
churn_fig = plot_churn_distribution(df)
st.plotly_chart(churn_fig)

# Numerical features analysis
st.subheader("Numerical Features Analysis")
num_features_fig = plot_numerical_features(df)
st.plotly_chart(num_features_fig, use_container_width=True)

# Correlation heatmap
st.subheader("Feature Correlation Analysis")
corr_fig = correlation_heatmap(df)
st.plotly_chart(corr_fig, use_container_width=True)

# Categorical features analysis
st.subheader("Categorical Features Analysis")
cat_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Let user select which categorical feature to view
selected_cat = st.selectbox("Select categorical feature to explore:", cat_cols)
cat_fig = plot_categorical_features(df, selected_cat)
st.plotly_chart(cat_fig, use_container_width=True)

# Summary statistics
st.subheader("Summary Statistics")
st.dataframe(df.describe())

# Data quality information
st.subheader("Data Quality Information")
col1, col2 = st.columns(2)
with col1:
    st.write("Missing values:")
    st.write(df.isnull().sum())
with col2:
    st.write("Data types:")
    st.write(df.dtypes)