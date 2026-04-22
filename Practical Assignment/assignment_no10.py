# -*- coding: utf-8 -*-
"""Assignment_no10.ipynb

10. Assignment on Regression technique. 

Download temperature data from the link below.

https://www.kaggle.com/venky73/temperaturesof-india?select=temperatures.csv

This data consists of temperatures of INDIA averaging the temperatures of all places month wise. Temperatures values are recorded in CELSIUS 

Apply Linear Regression using a suitable library function and predict the Month-wise temperature.
Assess the performance of regression models using MSE, MAE and R-Square   
metrics
Visualize a simple regression model.

**Import Libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

"""**Load Dataset**"""

df = pd.read_csv('/content/temperatures.csv')
df.head()

"""**Understand Dataset**"""

print(df.columns)
print(df.info())

"""**Convert Data**
**We convert data into Month vs Temperature format**


"""

# Check columns
print(df.columns)

# Convert 'YEAR' column to numeric if needed
df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')

# Drop missing values
df = df.dropna()

# Create a month-wise dataset
# Example: predicting January temperature vs Year

X = df[['YEAR']]
y = df['APR']   # You can change to FEB, MAR, etc.

# Reshape if needed
X = X.values
y = y.values

"""**Convert Month Names to Numbers**

**Remove Missing Values**
"""

df_long = df_long.dropna()

"""**Define Features & Target**"""

X = df_long[['Month']]   # Input
y = df_long['Temperature']   # Output

"""**Train-Test Split**

**Apply Linear Regression**
"""

model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

"""**Predictions**"""

y_pred = model.predict(X_test)

"""**Evaluation Metrics**"""

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R2 Score:", r2)

"""**Visualization**"""

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

plt.xlabel("Year")
plt.ylabel("Temperature (Jan)")
plt.title("Linear Regression - Temperature Prediction")
plt.legend()

plt.show()

"""**Predict Temperature for Any Month**"""

# Example: Predict for May (5)
pred = model.predict([[5]])
print("Predicted Temperature for Month 5:", pred[0])