# pages/prediction.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from utils.preprocessing import get_preprocessing_pipeline

st.set_page_config(
    page_title="Churn Prediction - Telco Churn",
    page_icon="🔮",
    layout="wide"
)

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

# Load the complete pipeline instead of just the model
pipeline_data = load_pipeline()

# Check if pipeline is loaded successfully
if pipeline_data is None:
    st.stop()

# Extract components
model = pipeline_data['model']
preprocessing_pipeline = pipeline_data['preprocessing']

st.title("Customer Churn Prediction Tool")

st.write("""
Use this tool to predict whether a customer is likely to churn based on their characteristics.
Fill in the form below with customer information to get a prediction.
""")

# Create two columns for the form
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 65)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure))

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Predict button
if st.button("Predict Churn"):
    # Apply preprocessing
    input_processed = preprocessing_pipeline.transform(input_data)
    
    # Make prediction
    prediction_proba = model.predict_proba(input_processed)[0, 1]
    prediction_binary = model.predict(input_processed)[0]
    
    # Display results
    st.subheader("Prediction Results")
    
    # Gauge chart for churn probability
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction_proba * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    st.plotly_chart(fig)
    
    # Display recommendation
    st.subheader("Recommendation")
    if prediction_binary == 1:
        st.markdown("""
        ⚠️ **This customer is likely to churn.**
        
        Consider these retention strategies:
        * Offer a promotional discount
        * Reach out for feedback
        * Propose a loyalty program
        * Offer service upgrades
        """)
    else:
        st.markdown("""
        ✅ **This customer is likely to stay.**
        
        Recommended actions:
        * Continue providing quality service
        * Consider up-selling additional services
        * Enroll in loyalty program if not already
        """)
        
    # Show key risk/retention factors
    st.subheader("Key Factors")
    
    factors = []
    
    # These are simplified heuristics based on common patterns
    if contract == "Month-to-month":
        factors.append("⚠️ Month-to-month contract (higher churn risk)")
    else:
        factors.append("✅ Long-term contract (retention positive)")
        
    if tenure < 12:
        factors.append("⚠️ New customer (< 12 months)")
    elif tenure > 36:
        factors.append("✅ Loyal customer (> 36 months)")
        
    if payment_method == "Electronic check":
        factors.append("⚠️ Electronic check payment method (associated with higher churn)")
    
    if internet_service == "Fiber optic" and (online_security == "No" or tech_support == "No"):
        factors.append("⚠️ Fiber optic without security/support (churn risk)")
    
    if internet_service == "Fiber optic" and online_security == "Yes" and tech_support == "Yes":
        factors.append("✅ Protected fiber service (retention positive)")
        
    for factor in factors:
        st.write(factor)