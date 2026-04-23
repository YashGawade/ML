# 10. Assignment on Regression technique. 

# Download temperature data from the link below.

# https://www.kaggle.com/venky73/temperaturesof-india?select=temperatures.csv

# This data consists of temperatures of INDIA averaging the temperatures of all places month wise. 
# Temperatures values are recorded in CELSIUS 

# Apply Linear Regression using a suitable library function and predict the Month-wise temperature.
# Assess the performance of regression models using MSE, MAE and R-Square   
# metrics
# Visualize a simple regression model.


# =========================================
# Import Libraries
# =========================================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================================
# Load Dataset
# =========================================
df = pd.read_csv("temperatures.csv")


# =========================================
# Convert Data (Wide → Long)
# =========================================
df_long = df.melt(
    id_vars=['YEAR'],
    var_name='Month',
    value_name='Temperature'
)


# =========================================
# Convert Month Names to Numbers
# =========================================
month_map = {
    'JAN':1, 'FEB':2, 'MAR':3, 'APR':4,
    'MAY':5, 'JUN':6, 'JUL':7, 'AUG':8,
    'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12
}

df_long['Month'] = df_long['Month'].map(month_map)


# =========================================
# Remove Missing Values
# =========================================
df_long = df_long.dropna()


# =========================================
# Define Features & Target
# =========================================
X = df_long[['Month']]        # Input
y = df_long['Temperature']    # Output


# =========================================
# Train-Test Split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================
# Apply Linear Regression
# =========================================
model = LinearRegression()
model.fit(X_train, y_train)


# =========================================
# Predictions
# =========================================
y_pred = model.predict(X_test)


# =========================================
# Evaluation Metrics
# =========================================
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R2 Score:", r2)


# =========================================
# Visualization
# =========================================
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, color='red', label="Regression Line")

plt.xlabel("Month")
plt.ylabel("Temperature")
plt.title("Month vs Temperature")

plt.legend()
plt.show()


# =========================================
# Prediction Example
# =========================================
pred = model.predict([[5]])   # Month = May
print("Predicted Temperature for Month 5:", pred[0])