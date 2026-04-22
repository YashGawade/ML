#  -*- coding: utf-8 -*-
# .Predict the price of the Uber ride from a given pickup point to the agreed drop-off location. Perform following tasks: 
# 1. Pre-process the dataset. 
# 2. Identify outliers.
# 3. Check the correlation. 
# 4. Implement linear regression and ridge, Lasso regression models.
#  5. Evaluate the models and compare their respective scores like R2, RMSE, etc.
#   Dataset link: https://www.kaggle.com/datasets/yasserh/uber-fares-dataset



**Assignment no-5**
"""

from google.colab import files
uploaded = files.upload()

"""**Import Required Libraries**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

"""Load the dataset"""

import pandas as pd

df = pd.read_csv('uber.csv')
df.head()

"""Dataset Pre-processing"""

df.isnull().sum()
df.dropna(inplace=True)
df = df[(df['fare_amount'] > 0) & (df['passenger_count'] > 0)]

"""Outlier Detection & Removal"""

Q1 = df['fare_amount'].quantile(0.25)
Q3 = df['fare_amount'].quantile(0.75)
IQR = Q3 - Q1

df = df[
    (df['fare_amount'] >= Q1 - 1.5 * IQR) &
    (df['fare_amount'] <= Q3 + 1.5 * IQR)
]

"""Feature Engineering (Distance Calculation)"""

from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

df['distance_km'] = df.apply(lambda row: haversine(
    row['pickup_latitude'], row['pickup_longitude'],
    row['dropoff_latitude'], row['dropoff_longitude']
), axis=1)

"""Correlation Analysis"""

plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=[np.number])

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

"""Prepare Data for Modeling"""

X = df[['distance_km', 'passenger_count']]
y = df['fare_amount']

"""Train–Test Split"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""Apply Regression Models
Linear Regression
"""

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

"""Ridge Regression"""

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

"""Lasso Regression"""

from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

"""Model Evaluation"""

from sklearn.metrics import r2_score, mean_squared_error

def evaluate(name, y_test, y_pred):
    print(name)
    print("R2 Score:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("-" * 30)

evaluate("Linear Regression", y_test, y_pred_lr)
evaluate("Ridge Regression", y_test, y_pred_ridge)
evaluate("Lasso Regression", y_test, y_pred_lasso)