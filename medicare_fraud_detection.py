
# medicare_fraud_detection.py

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib


# Load Datasets
train_bene_df = pd.read_csv("Train_Beneficiarydata-1542865627584.csv")
train_ip_df = pd.read_csv("Train_Inpatientdata-1542865627584.csv")
train_op_df = pd.read_csv("Train_Outpatientdata-1542865627584.csv")
train_labels = pd.read_csv("Train-1542865627584.csv")
    

# Data Preprocessing
train_ip_df['AdmissionDt'] = pd.to_datetime(train_ip_df['AdmissionDt'], format='%Y-%m-%d')
train_ip_df['DischargeDt'] = pd.to_datetime(train_ip_df['DischargeDt'], format='%Y-%m-%d')
train_ip_df['Admit_Duration'] = (train_ip_df['DischargeDt'] - train_ip_df['AdmissionDt']).dt.days

train_op_df['ClaimStartDt'] = pd.to_datetime(train_op_df['ClaimStartDt'], format='%Y-%m-%d')
train_op_df['ClaimEndDt'] = pd.to_datetime(train_op_df['ClaimEndDt'], format='%Y-%m-%d')
train_op_df['Claim_Duration'] = (train_op_df['ClaimEndDt'] - train_op_df['ClaimStartDt']).dt.days

train_bene_df['DOB'] = pd.to_datetime(train_bene_df['DOB'], format='%Y-%m-%d')
train_bene_df['DOD'] = pd.to_datetime(train_bene_df['DOD'], format='%Y-%m-%d')
train_bene_df['Age'] = train_bene_df['DOB'].apply(lambda x: 2024 - x.year)
    

# Feature Engineering and Merging
ip_data = train_ip_df.groupby('Provider').agg({'Admit_Duration': ['mean', 'count', 'sum']})
ip_data.columns = ['IP_mean_admit', 'IP_claim_count', 'IP_total_admit']
ip_data.reset_index(inplace=True)

op_data = train_op_df.groupby('Provider').agg({'Claim_Duration': ['mean', 'count', 'sum']})
op_data.columns = ['OP_mean_claim', 'OP_claim_count', 'OP_total_claim']
op_data.reset_index(inplace=True)

labels = train_labels.copy()

data = pd.merge(ip_data, op_data, on='Provider', how='outer')
data = pd.merge(data, labels, on='Provider', how='left')
    

# Encoding and Model Training
data.fillna(0, inplace=True)

X = data.drop(['Provider', 'PotentialFraud'], axis=1)
y = data['PotentialFraud']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred))
    
# Save Model
joblib.dump(model, 'fraud_detection_model.pkl')
    