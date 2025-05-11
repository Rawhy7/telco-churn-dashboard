import sys
import numpy as np

# This class needs to match EXACTLY the one used in the notebook
class FeatureEngineering:
    def transform(self, X, y=None):
        return add_features(X)

    def fit(self, X, y=None):
        return self

def add_features(df):
    #  Customer lifetime value (CLV) = TotalCharges / tenure (if tenure > 0)
    df['CLV'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], 0)
    
    #  Average monthly charges
    df['AvgMonthlyCharges'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])
    
    #  Has Phone Service (binary)
    df['HasPhoneService'] = np.where(df['PhoneService'] == 'Yes', 1, 0)
    
    #  Has Internet Service (binary)
    df['HasInternetService'] = np.where(df['InternetService'] != 'No', 1, 0)
    
    #  Number of services subscribed
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['NumServices'] = 0
    for service in services:
        df['NumServices'] += np.where((df[service] == 'Yes'), 1, 0)
    
    #  Is new customer (tenure <= 6 months)
    df['IsNewCustomer'] = np.where(df['tenure'] <= 6, 1, 0)
    
    # Payment digitalization (electronic methods vs. others)
    df['DigitalPayment'] = np.where(
        (df['PaymentMethod'] == 'Electronic check') |
        (df['PaymentMethod'] == 'Credit card (automatic)') |
        (df['PaymentMethod'] == 'Bank transfer (automatic)'),
        1, 0)
    
    # Contract type as ordered category
    df['ContractType_Coded'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    return df

# Add to module lookup path - this allows pickle to find the class
sys.modules['__main__'] = sys.modules[__name__]