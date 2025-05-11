# utils/visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

def plot_churn_distribution(df):
    """Plot the distribution of churn"""
    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Churn', 'Count']
    churn_counts['Churn'] = churn_counts['Churn'].map({1: 'Churned', 0: 'Stayed'})
    churn_counts['Percentage'] = churn_counts['Count'] / churn_counts['Count'].sum() * 100
    
    fig = px.pie(churn_counts, values='Count', names='Churn', 
                 title='Customer Churn Distribution',
                 color='Churn',
                 color_discrete_map={'Stayed': '#2E86C1', 'Churned': '#E74C3C'})
    
    return fig

def plot_numerical_features(df):
    """Plot histograms of numerical features by churn status"""
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    fig = px.histogram(df, x=numerical_cols, color='Churn', 
                       barmode='overlay', 
                       color_discrete_map={0: '#2E86C1', 1: '#E74C3C'},
                       facet_col='variable', 
                       labels={'value': 'Value', 'variable': 'Feature'},
                       title='Distribution of Numerical Features by Churn Status')
    fig.update_layout(height=400)
    return fig

def plot_categorical_features(df, column):
    """Plot the relationship between a categorical feature and churn"""
    df_plot = df.copy()
    df_plot['Churn Status'] = df_plot['Churn'].map({1: 'Churned', 0: 'Stayed'})
    
    # Count by category and churn status
    count_df = df_plot.groupby([column, 'Churn Status']).size().reset_index(name='Count')
    
    # Calculate percentage within each category
    total_counts = count_df.groupby(column)['Count'].transform('sum')
    count_df['Percentage'] = count_df['Count'] / total_counts * 100
    
    fig = px.bar(count_df, x=column, y='Percentage', color='Churn Status',
                 barmode='group',
                 color_discrete_map={'Stayed': '#2E86C1', 'Churned': '#E74C3C'},
                 title=f'Churn Rate by {column}')
    
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance for the model"""
    if hasattr(model, 'coef_'):  # For linear models like logistic regression
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):  # For tree-based models
        importance = model.feature_importances_
    else:
        st.warning("Feature importance not available for this model")
        return None
    
    # Create dataframe for plotting
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(15)
    
    fig = px.bar(feat_imp, x='Importance', y='Feature', 
                 orientation='h', title='Top 15 Feature Importance')
    
    return fig

def correlation_heatmap(df):
    """Plot correlation heatmap for numerical features"""
    # Select numeric columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_df = df.select_dtypes(include=numerics)
    
    # Calculate correlation
    corr = numeric_df.corr()
    
    # Plot using Plotly
    fig = px.imshow(corr, text_auto=True, aspect="auto", 
                    title="Feature Correlation Heatmap")
    return fig