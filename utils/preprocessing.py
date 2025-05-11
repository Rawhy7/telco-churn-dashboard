# utils/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data():
    """Load the telco customer churn dataset"""
    df = pd.read_csv('data/Telco Customer Churn.csv')
    
    # Convert Churn to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Convert TotalCharges to numeric and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    
    return df

class FeatureEngineering:
    def transform(self, df, y=None):
        """Add engineered features to the dataset"""
        df = df.copy()  # To avoid SettingWithCopyWarning
        
        # Customer lifetime value (CLV) = TotalCharges / tenure
        df['CLV'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], 0)
        
        # Average monthly charges
        df['AvgMonthlyCharges'] = np.where(df['tenure'] > 0, 
                                          df['TotalCharges'] / df['tenure'], 
                                          df['MonthlyCharges'])
        
        # Has Phone Service (binary)
        df['HasPhoneService'] = np.where(df['PhoneService'] == 'Yes', 1, 0)
        
        # Has Internet Service (binary)
        df['HasInternetService'] = np.where(df['InternetService'] != 'No', 1, 0)
        
        # Number of services subscribed
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['NumServices'] = 0
        for service in services:
            df['NumServices'] += np.where((df[service] == 'Yes'), 1, 0)
        
        # Is new customer (tenure <= 6 months)
        df['IsNewCustomer'] = np.where(df['tenure'] <= 6, 1, 0)
        
        # Payment digitalization
        df['DigitalPayment'] = np.where(
            (df['PaymentMethod'] == 'Electronic check') |
            (df['PaymentMethod'] == 'Credit card (automatic)') |
            (df['PaymentMethod'] == 'Bank transfer (automatic)'),
            1, 0)
        
        # Contract type as ordered category
        df['ContractType_Coded'] = df['Contract'].map({'Month-to-month': 0, 
                                                      'One year': 1, 
                                                      'Two year': 2})
        return df
    
    def fit(self, X, y=None):
        return self

def get_preprocessing_pipeline():
    """Return preprocessing pipeline used in model training"""
    
    # Define column types
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'OnlineSecurity', 
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                       'StreamingTV', 'StreamingMovies', 'Contract', 
                       'PaperlessBilling', 'PaymentMethod']
    
    numerical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Transformers
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Full preprocessing pipeline
    pipeline = Pipeline([
        ('feature_engineering', FeatureEngineering()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline