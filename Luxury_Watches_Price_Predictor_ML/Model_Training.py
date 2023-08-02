import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import explained_variance_score as evs

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import Ridge, RANSACRegressor
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor

import warnings 
warnings.filterwarnings('ignore')

# Importing Data 
df = pd.read_csv('Luxury watch.csv')


# EDA & Preprocessing
print(df.info())

for c in df.columns:
    if df[c].dtype == 'object':
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
    df[c] = df[c].astype('float')

print(df.info())


# Data Split
features = df.drop('Price (USD)', axis= 1)
target = df['Price (USD)']

print(features, target)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, shuffle= True, random_state= 42, test_size= 0.23)


# Model Training
models = [DecisionTreeRegressor(), RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor(), Ridge(), RANSACRegressor(), SVR(), LinearSVR(), XGBRegressor()]

for m in models:
    print(m)

    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Training accuracy is : {evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy is : {evs(Y_test, pred_test)}\n')