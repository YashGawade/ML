# -*- coding: utf-8 -*-
"""Assignment_no11.ipynb

11. Assignment on Classification technique. Every year many students give the GRE exam to get admission in foreign Universities. 
The data set contains GRE Scores (out of 340), TOEFL Scores (out of 120), University Rating (out of 5), Statement of Purpose strength (out of 5), '
Letter of Recommendation strength (out of 5), Undergraduate GPA (out of 10), Research Experience (0=no, 1=yes), 
Admitted (0=no, 1=yes). Admitted is the target variable. 

Data Set: https://www.kaggle.com/mohansacharya/graduate-admissions

The counsellor of the firm is supposed check whether the student will get an admission or not based on his/her GRE score and Academic Score. 
So to help the counsellor to make appropriate decisions, 
build a machine learning model classifier using a Decision tree to predict whether a student will get admission or not. 
Apply Data pre-processing (Label Encoding, Data Transformation….) techniques if necessary. 

Perform data-preparation (Train-Test Split)
Apply Machine Learning Algorithm
Evaluate Model.

**Import Libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

"""**Load Dataset**"""

df = pd.read_csv('/content/Admission_Predict.csv')  # or your file name
df.head()

"""**Fix Column Names**"""

df.columns = df.columns.str.strip()
print(df.columns)

"""**Drop Unnecessary Column**"""

df = df.drop(columns=['Serial No.'])

"""**Create Target Variable**"""

# Convert to classification (0 or 1)
df['Admitted'] = (df['Chance of Admit'] >= 0.5).astype(int)

# Drop original column
df = df.drop(columns=['Chance of Admit'])

"""**Fix Column Names**"""

X = df[['GRE Score', 'TOEFL Score', 'University Rating',
        'SOP', 'LOR', 'CGPA', 'Research']]

y = df['Admitted']

"""**Train-Test Split**"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

"""**Decision Tree Model**"""

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

"""**Predictions**"""

y_pred = model.predict(X_test)

"""**Evaluation**"""

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""**Visualize Tree**"""

plt.figure(figsize=(15,10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

"""**Test Custom Student**"""

sample = [[320, 110, 4, 4, 4, 9.0, 1]]

pred = model.predict(sample)

if pred[0] == 1:
    print("Admitted ")
else:
    print("Not Admitted ")