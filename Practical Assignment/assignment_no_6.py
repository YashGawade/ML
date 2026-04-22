# -*- coding: utf-8 -*-
"""Assignment_no.6.ipynb

Use the diabetes data set from UCI and Pima Indians Diabetes data set for performing the following: 
a. Univariate analysis: Frequency, Mean, Median, Mode, Variance, Standard Deviation, Skewness and Kurtosis 
b. Bivariate analysis: Linear and logistic regression modeling 
c. Multiple Regression analysis 
d. Also compare the results of the above analysis for the two data sets. 
Dataset link: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

**Import Libraries**
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

from google.colab import files
uploaded = files.upload()

"""**Load Dataset**"""

import pandas as pd

df = pd.read_csv('diabetes (1).csv')
df.head()

df.shape
df.info()
df.describe()

"""**Data Cleaning**"""

cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols] = df[cols].replace(0, np.nan)


df.fillna(df.median(), inplace=True)

"""**Fill Missing Values**"""

df.fillna(df.median(), inplace=True)
df.isnull().sum()

"""Frequency Distribution"""

df['Outcome'].value_counts()

Statistical Measures

univariate = pd.DataFrame({
    'Mean': df.mean(),
    'Median': df.median(),
    'Mode': df.mode().iloc[0],
    'Variance': df.var(),
    'Std Dev': df.std(),
    'Skewness': df.skew(),
    'Kurtosis': df.kurt()
})

univariate

"""**Histogram**"""

df.hist(figsize=(15,10))
plt.show()

"""**Boxplot**"""

plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

"""**PART2-BIVARIATE ANALYSIS**"""

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

"""**Linear Regression (BMI vs Glucose)**"""

X = df[['BMI']]
y = df['Glucose']

lr = LinearRegression()
lr.fit(X,y)

print("Slope:", lr.coef_[0])
print("Intercept:", lr.intercept_)

# recreate X and y again (VERY IMPORTANT)
X = df[['BMI']]
y = df['Glucose']

# train model again
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

# plot
plt.figure(figsize=(6,4))
plt.scatter(X['BMI'], y)
plt.plot(X['BMI'], lr.predict(X), color='red')
plt.xlabel("BMI")
plt.ylabel("Glucose")
plt.title("Linear Regression: BMI vs Glucose")
plt.show()

"""Logistic Regression (Predict Diabetes)"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get probability scores
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])  # Diagonal reference line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.show()

# Print AUC Score
print("AUC Score:", roc_auc_score(y_test, y_prob))

"""**MULTIPLE REGRESSION**"""

X = df[['Pregnancies','BloodPressure','SkinThickness','Insulin','BMI','Age']]
y = df['Glucose']

multi = LinearRegression()
multi.fit(X,y)

print("Coefficients:", multi.coef_)
print("Intercept:", multi.intercept_)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()