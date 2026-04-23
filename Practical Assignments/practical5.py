# 5.Predict the price of the Uber ride from a given pickup point 
# to the agreed drop-off location. Perform following tasks: 
# 1. Pre-process the dataset. 
# 2. Identify outliers. 
# 3. Check the correlation.
# 4. Implement linear regression and ridge, Lasso regression models.
# 5. Evaluate the models
# and compare their respective scores like R2, RMSE, etc.
   
#  Dataset link: https://www.kaggle.com/datasets/yasserh/uber-fares-dataset

# =========================================
# Import Libraries
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error


# =========================================
# Load Dataset
# =========================================
df = pd.read_csv("uber.csv")


# =========================================
# 1. Data Preprocessing
# =========================================

# Remove unnecessary columns
df.drop(columns=["Unnamed: 0", "key"], inplace=True, errors='ignore')

# Remove missing values
df.dropna(inplace=True)

# Convert datetime
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

# Extract features
df["hour"] = df["pickup_datetime"].dt.hour
df["month"] = df["pickup_datetime"].dt.month


# =========================================
# Create Distance Feature
# =========================================
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

df["distance_km"] = haversine(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)


# =========================================
# 2. Outlier Removal
# =========================================
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 200)]
df = df[(df["distance_km"] > 0) & (df["distance_km"] < 100)]
df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]


# =========================================
# 3. Correlation
# =========================================
corr = df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# =========================================
# 4. Model Building
# =========================================
X = df[["distance_km", "passenger_count", "hour", "month"]]
y = df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
pred_ridge = ridge.predict(X_test)

# Lasso Regression
lasso = Lasso(alpha=0.001)
lasso.fit(X_train, y_train)
pred_lasso = lasso.predict(X_test)


# =========================================
# 5. Evaluation (Simple, No Function)
# =========================================

# Linear Regression
r2_lr = r2_score(y_test, pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
print("Linear Regression -> R2:", r2_lr, "RMSE:", rmse_lr)

# Ridge Regression
r2_ridge = r2_score(y_test, pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge))
print("Ridge Regression -> R2:", r2_ridge, "RMSE:", rmse_ridge)

# Lasso Regression
r2_lasso = r2_score(y_test, pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, pred_lasso))
print("Lasso Regression -> R2:", r2_lasso, "RMSE:", rmse_lasso)