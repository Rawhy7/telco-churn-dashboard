# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from utils.preprocessing import load_data

# Set page config
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def get_data():
    return load_data()

df = get_data()

# Main page header
st.title("Telco Customer Churn Analysis Dashboard")

# Overview metrics
st.subheader("Overview Metrics")
col1, col2, col3, col4 = st.columns(4)

# Calculate metrics
total_customers = len(df)
churn_rate = df['Churn'].mean() * 100
avg_tenure = df['tenure'].mean()
avg_monthly = df['MonthlyCharges'].mean()

with col1:
    st.metric("Total Customers", f"{total_customers:,}")

with col2:
    st.metric("Churn Rate", f"{churn_rate:.2f}%")
    
with col3:
    st.metric("Avg. Tenure (months)", f"{avg_tenure:.1f}")
    
with col4:
    st.metric("Avg. Monthly Charges", f"${avg_monthly:.2f}")
    
# Brief description about the project
st.markdown("""
This dashboard provides insights about customer churn in a telecommunications company. 
Use the navigation menu to explore different aspects of the data and model.

- **Data Explorer**: Explore the dataset and visualize relationships
- **Model Insights**: Understand model performance and feature importance
- **Prediction Tool**: Predict churn probability for specific customers
""")

# Display dataset sample
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Footer
st.markdown("---")
st.markdown("© 2025 Telco Churn Analysis Dashboard")