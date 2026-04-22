# 2. Perform the following operations using  Python on the data sets:

# Compute and display summary statistics for each feature available in the dataset. (e.g. minimum value, maximum value, mean, range, standard deviation, variance and percentiles
# Illustrate the feature distributions using histogram.
# Data cleaning, Data integration, Data transformation, Data model building (e.g. Classification) 

# Step 1: Upload Dataset in Google Colab
from google.colab import files
uploaded = files.upload()

# Step 2: Read Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Replace with your uploaded file name if different
df = pd.read_csv('Iris.csv')

# Display first 5 rows
print(df.head())

# --------------------------------------------------
# 1. Summary Statistics for Each Feature
# --------------------------------------------------

# Select only numeric columns
numeric_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

print("\nSummary Statistics:\n")

for col in numeric_columns:
    print(f"Feature: {col}")
    print("Minimum Value :", df[col].min())
    print("Maximum Value :", df[col].max())
    print("Mean          :", df[col].mean())
    print("Median        :", df[col].median())
    print("Mode          :", df[col].mode()[0])
    print("Range         :", df[col].max() - df[col].min())
    print("Standard Dev. :", df[col].std())
    print("Variance      :", df[col].var())
    print("25th Percentile:", df[col].quantile(0.25))
    print("50th Percentile:", df[col].quantile(0.50))
    print("75th Percentile:", df[col].quantile(0.75))
    print("----------------------------------------")

# You can also display all statistics together
print("\nUsing describe():")
print(df[numeric_columns].describe())

# --------------------------------------------------
# 2. Histogram for Feature Distribution
# --------------------------------------------------

plt.figure(figsize=(12, 8))
df[numeric_columns].hist(figsize=(12, 8), bins=15, edgecolor='black')
plt.suptitle("Histogram of Iris Dataset Features", fontsize=16)
plt.show()

# --------------------------------------------------
# 3. Data Cleaning
# --------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
print("\nShape before removing duplicates:", df.shape)
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)

# --------------------------------------------------
# 4. Data Integration
# --------------------------------------------------

# Combine required feature columns into X
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

# Target column
y = df['Species']

print("\nFeature Matrix (X):")
print(X.head())

print("\nTarget Variable (y):")
print(y.head())

# --------------------------------------------------
# 5. Data Transformation
# --------------------------------------------------

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

print("\nEncoded Species Values:")
print(df[['Species', 'Species_encoded']].head())

# Features and target after transformation
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[features]
y = df['Species_encoded']

# --------------------------------------------------
# 6. Train-Test Split
# --------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# --------------------------------------------------
# 7. Data Model Building (Classification)
# --------------------------------------------------

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# --------------------------------------------------
# 8. Model Evaluation
# --------------------------------------------------

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


